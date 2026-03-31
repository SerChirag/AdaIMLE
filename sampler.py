from math import ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from LPNet import LPNet
from helpers.utils import is_main_process, get_world_size, get_rank, safe_barrier
from models import parse_layer_string
from torch import autocast
import faiss
import faiss.contrib.torch_utils
from tqdm import tqdm
from helpers.autoencoder import load_autoencoder, encode_images_to_latents, decode_latents_to_images
from helpers.cache_utils import latent_cache_key, load_latent_cache, save_latent_cache

class Sampler:
    def __init__(self, H, sz, preprocess_fn, autoencoder=None):
        
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.world_size = get_world_size()
        self.rank = get_rank()

        self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduction='none').to(self.device)
        self.l1_loss = torch.nn.L1Loss(reduction='none').to(self.device)
        self.H = H
        self.latent_lr = H.latent_lr
        self.sz = sz
        self.unique_indices = 0
        self.entire_ds = torch.arange(sz)
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.last_selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)

        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
        self.latent_spatial_size = int(getattr(H, 'latent_spatial_size', max(self.block_res)))

        self.selected_dists = torch.empty([sz], dtype=torch.float32)
        self.selected_dists[:] = np.inf
        self.selected_dists_tmp = torch.empty([sz], dtype=torch.float32)

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)

        self.pool_latents = None

        self.decode_for_metrics = bool(getattr(H, 'autoencoder_decode_for_metrics', True))
        self.autoencoder = autoencoder if autoencoder is not None else load_autoencoder(H, self.device)
        if is_main_process():
            ae_name = type(self.autoencoder).__name__
            ae_source = getattr(getattr(self.autoencoder, 'config', None), '_name_or_path', 'unknown')
            print(f'\n[autoencoder] Loaded {ae_name} from {ae_source}\n')
        # Do not compile the autoencoder — it runs in frozen inference-only mode rarely
        # (FID, visualization). CUDA graph capture for the VAE decoder is expensive and
        # its graph memory stays resident permanently, causing VRAM spikes after FID runs.

        self.autoencoder_native_latent_size = None
        fake_rgb = torch.zeros(1, 3, H.image_size, H.image_size, device=self.device)
        native_latents = encode_images_to_latents(self.autoencoder, fake_rgb, target_spatial=None)
        self.autoencoder_native_latent_size = (native_latents.shape[-2], native_latents.shape[-1])

        if H.search_type != 'l2':
            raise ValueError('This branch expects search_type=l2.')

        self.nn_search_batch = H.nn_search_batch

        self.l2_projection = None

        fake = torch.zeros(1, H.image_channels, self.latent_spatial_size, self.latent_spatial_size, device=self.device)

        safe_barrier()

        if(H.search_type == 'l2'):
            interpolated = fake.reshape(fake.shape[0],-1)
            sum_dims = interpolated.shape[1]

        else:
            exit()

        self.dci_dim = sum_dims
        self.latent_channels = H.image_channels
        self.dataset_proj_torch = torch.empty([sz, sum_dims], dtype=torch.float32, device='cpu')
        self.dataset_proj = None
        self.pool_samples_proj = None
        self._dataset_proj_gpu = None
        self._local_pool_latents = None
        self._local_pool_proj = None
        self._local_pool_combined = None
        self._gathered_combined_main = None
        self._full_combined_main = None

        self.dataset_size = sz
        self.db_iter = 0
        self.generator_seed = torch.Generator(device=self.device)         
        self.generator_seed.manual_seed(H.seed + self.rank)

        self.compress_comm = bool(getattr(H, 'compress_comm', True))
        if self.compress_comm:
            self._comm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self._comm_dtype = torch.float32

        self.faiss_res = None

        self.faiss_res = faiss.StandardGpuResources()  # one per process
        # FlatL2 needs no temp memory; cap it to avoid competing with PyTorch's allocator.
        self.faiss_res.setTempMemory(64 * 1024 * 1024)  # 64 MB
        index_flat = faiss.IndexFlatL2(self.dci_dim)
        dev_id = torch.cuda.current_device()
        self.faiss_index_flat = faiss.index_cpu_to_gpu(self.faiss_res, dev_id, index_flat)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    
    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)

        interpolated = inp.reshape(inp.shape[0],-1)
        # interpolated = F.normalize(interpolated, p=2, dim=1)
        return interpolated


    def init_projection(self, dataset):

        use_cache = bool(getattr(self.H, 'use_cache', True))
        cache_dir = getattr(self.H, 'cache_dir', './cache')

        cached = None
        if use_cache:
            key = latent_cache_key(
                data_root=self.H.data_root,
                dataset_type=self.H.dataset,
                image_size=self.H.image_size,
                latent_spatial_size=self.latent_spatial_size,
                autoencoder_type=getattr(self.H, 'autoencoder_type', 'kl'),
                autoencoder_name_or_path=getattr(self.H, 'autoencoder_name_or_path', ''),
                image_channels=self.latent_channels,
            )
            cached = load_latent_cache(cache_dir, key, expected_size=self.sz)

        if cached is not None:
            if is_main_process():
                print(f"[cache] Loaded latent projections from cache "
                      f"({cached.shape[0]} samples, dim={cached.shape[1]}).")
            self.dataset_proj_torch.copy_(cached)
        else:
            ae_batch = getattr(self.H, 'ae_batch', self.H.imle_batch)
            dataloader = DataLoader(
                dataset,
                batch_size=ae_batch,
            )

            if is_main_process():
                print("Starting Initialization")

            with torch.inference_mode():
                for ind, x in tqdm(enumerate(dataloader), total=len(dataloader), desc="Initializing"):
                    batch_slice = slice(ind * ae_batch, ind * ae_batch + x[0].shape[0])
                    if self.H.search_type == 'l2':
                        self.dataset_proj_torch[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[1]).cpu()
                    else:
                        exit()

            if use_cache and is_main_process():
                save_latent_cache(cache_dir, key, self.dataset_proj_torch)

        # Pin host memory so per-batch H2D copies can use faster async transfer.
        if torch.cuda.is_available() and not self.dataset_proj_torch.is_pinned():
            try:
                self.dataset_proj_torch = self.dataset_proj_torch.pin_memory()
            except RuntimeError as e:
                if is_main_process():
                    print(f"Warning: could not pin dataset_proj_torch ({e}); continuing without pinned cache.")

        # Keep a torch tensor for fast indexed target lookup in training,
        # and a NumPy view for FAISS nearest-neighbor search.
        self.dataset_proj = self.dataset_proj_torch.numpy()

        # Build a persistent GPU cache of query features on rank 0 to avoid
        # re-uploading the full table every resample.
        if is_main_process():
            self._dataset_proj_gpu = self.dataset_proj_torch.to(self.device, non_blocking=True)

    def sample(self, latents, gen, snoise=None):
        with torch.inference_mode():
            with autocast(device_type='cuda', dtype=self.H.amp_dtype_torch):
                latents = latents.to(self.device)
                px_z = gen(latents, None)
                if self.decode_for_metrics:
                    px_z = decode_latents_to_images(self.autoencoder, px_z, self.autoencoder_native_latent_size)

                if px_z.shape[1] == 1:
                    px_z = px_z.repeat(1, 3, 1, 1)
                elif px_z.shape[1] >= 3:
                    px_z = px_z[:, :3]

                px_z = px_z.permute(0, 2, 3, 1)
                xhat = (px_z + 1.0) * 127.5
                xhat = xhat.detach().cpu().numpy()
                xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
                return xhat

    
    def pseudo_huber(self, diff):
        return 2.0 * self.H.huber_delta**2 * (torch.sqrt(1 + (diff / (self.H.huber_delta)**2)) - 1)

    def calc_loss(self, inp, tar, use_mean=True, logging=False):
        if self.H.loss_type == 'huber':
            per_elem = self.pseudo_huber((inp - tar) ** 2)
        elif self.H.loss_type == 'pseudo_l1':
            per_elem = self.l1_loss(inp, tar) * self.H.huber_delta
        elif self.H.loss_type == 'mclure':
            residual = inp - tar
            per_elem = (residual ** 2) / (self.H.loss_scale**2 + residual ** 2)
        elif self.H.loss_type == 'welsch':
            residual = inp - tar
            per_elem = (1 - torch.exp(-(residual / self.H.loss_scale)**2))
        elif self.H.loss_type == 'rmse':
            l2_loss = (inp - tar).pow(2).flatten(1).mean(dim=1)
            per_elem = torch.sqrt(l2_loss + 1e-8)
        elif self.H.loss_type == 'cauchy':
            per_elem = torch.log(1 + 0.5 * ((inp - tar) / self.H.loss_scale)**2)
        else:
            per_elem = self.l2_loss(inp, tar)

        return per_elem.mean()
    
    def resample_pool(self, gen):

        gen.eval()

        # Determine local pool size
        local_pool_size = ceil(self.pool_size / self.world_size)

        # Reuse local buffers across resamples to avoid repeated allocations.
        if self._local_pool_latents is None or self._local_pool_latents.shape[0] != local_pool_size:
            self._local_pool_latents = torch.empty((local_pool_size, self.H.latent_dim), device=self.device)
            self._local_pool_combined = torch.empty(
                (local_pool_size, self.H.latent_dim + self.dci_dim),
                device=self.device,
                dtype=self._comm_dtype,
            )
            if self.rank == 0:
                self._gathered_combined_main = [torch.empty_like(self._local_pool_combined) for _ in range(self.world_size)]

        # Preserve existing behavior: regenerate the entire local pool each resample.
        self._local_pool_latents.normal_(mean=0.0, std=1.0, generator=self.generator_seed)

        # Process local chunk in batches, including the tail batch.
        with torch.inference_mode():
            for start in range(0, local_pool_size, self.H.imle_batch):
                end = min(start + self.H.imle_batch, local_pool_size)
                batch_slice = slice(start, end)
                cur_latents = self._local_pool_latents[batch_slice]
                self._local_pool_combined[batch_slice, :self.H.latent_dim].copy_(cur_latents.to(self._comm_dtype))
                with autocast(device_type='cuda', dtype=self.H.amp_dtype_torch):
                    outputs = gen(cur_latents)
                    if self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    else:
                        exit()
                    self._local_pool_combined[batch_slice, self.H.latent_dim:].copy_(proj.to(self._comm_dtype))

        # One collective for both latents and projections to reduce comm overhead.
        if self.rank == 0:
            if self._gathered_combined_main is None or len(self._gathered_combined_main) != self.world_size:
                self._gathered_combined_main = [torch.empty_like(self._local_pool_combined) for _ in range(self.world_size)]
            torch.distributed.gather(self._local_pool_combined, gather_list=self._gathered_combined_main, dst=0)
        else:
            torch.distributed.gather(self._local_pool_combined, gather_list=None, dst=0)

        gen.train()

        # Aggregate the full pool latents and projected features.
        # cat into a pre-allocated combined buffer (one GPU kernel), then take contiguous
        # slices — no extra allocation beyond the buffer itself.
        if self.rank == 0:
            full_pool_size = local_pool_size * self.world_size
            combined_dim = self.H.latent_dim + self.dci_dim
            if (self._full_combined_main is None or self._full_combined_main.shape[0] != full_pool_size):
                self._full_combined_main = torch.empty((full_pool_size, combined_dim), dtype=torch.float32, device=self.device)
            torch.cat([c.to(torch.float32) for c in self._gathered_combined_main], dim=0, out=self._full_combined_main)
            self.pool_latents = self._full_combined_main[:, :self.H.latent_dim]
            self.pool_samples_proj = self._full_combined_main[:, self.H.latent_dim:]
    

    def nn_search_batched(self, queries, dataset):
        """
        Hard-first greedy Top-K matching (unique when possible).
        Returns:
            distances: (Nq,) torch.float32  # squared L2
            indices:   (Nq,) torch.long
        """
        topk = self.H.imle_db_topk
        Nq = queries.shape[0]
        Nd = dataset.shape[0]

        if Nq == 0:
            return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)

        queries_np = np.ascontiguousarray(queries.detach().cpu().numpy() if isinstance(queries, torch.Tensor) else queries, dtype=np.float32)
        dataset_np = np.ascontiguousarray(dataset.detach().cpu().numpy() if isinstance(dataset, torch.Tensor) else dataset, dtype=np.float32)

        topk = int(min(max(1, topk), Nd))

        self.faiss_index_flat.reset()
        self.faiss_index_flat.add(dataset_np)

        # Sort queries hard-first by 1-NN distance (proxy for hardness)
        if Nd >= 2:
            D2, _ = self.faiss_index_flat.search(queries_np, 2)
            margin = D2[:, 0]
        else:
            margin = np.zeros(Nq, dtype=np.float32)

        perm = np.random.permutation(Nq)
        order = perm[np.argsort(margin[perm], kind="stable")]

        # Top-K candidates for all queries at once
        D, I = self.faiss_index_flat.search(queries_np, topk)  # (Nq, K)

        # Greedy unique assignment in hard-first order
        used = np.zeros(Nd, dtype=bool)
        out_idx = np.empty(Nq, dtype=np.int64)
        out_dst = np.empty(Nq, dtype=np.float32)

        for qi in order:
            cand, cd = I[qi], D[qi]
            chosen = -1
            for k in range(topk):
                j = int(cand[k])
                if not used[j]:
                    chosen = j
                    used[j] = True
                    out_dst[qi] = float(cd[k])
                    break
            if chosen == -1:  # all top-K taken, allow collision
                chosen = int(cand[0])
                out_dst[qi] = float(cd[0])
            out_idx[qi] = chosen

        self.faiss_index_flat.reset()
        return torch.from_numpy(out_dst), torch.from_numpy(out_idx)


    def imle_sample_force(self, gen, to_update=None):
        """
        Optimized force resampling routine using FAISS for batched nearest-neighbor search.
        In a DDP setting, each process handles a different subset of the dataset features,
        performs NN search locally, and then the results are merged and broadcast so that
        all processes end up with the complete global results.
        """
        if is_main_process():
            t1 = time.time()
            print("Starting pool resampling...")

        # Resample pool first (each process contributes its part);
        # this updates self.pool_samples_proj and self.pool_latents.
        self.resample_pool(gen)

        if(is_main_process()):
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")
        
        torch.cuda.empty_cache()

        self.selected_dists_tmp[:] = np.inf

        with torch.inference_mode():

            if(is_main_process()):
                if (
                    self._dataset_proj_gpu is None
                    or self._dataset_proj_gpu.shape != self.dataset_proj_torch.shape
                    or self._dataset_proj_gpu.device != self.device
                ):
                    self._dataset_proj_gpu = self.dataset_proj_torch.to(self.device, non_blocking=True)

                local_ds_feats = self._dataset_proj_gpu

                # Pool features (as computed from resample_pool).
                pool_feats = self.pool_samples_proj

                # Perform NN search for the local chunk. Returns arrays of shape (local_size, 1).
                local_distances, local_indices = self.nn_search_batched(local_ds_feats, pool_feats)

                # get count of unique indices for logging
                self.unique_indices = torch.unique(local_indices).numel() / self.sz

                local_indices = local_indices.to(device=self.pool_latents.device, non_blocking=True)
                new_latents = self.pool_latents.index_select(0, local_indices)

            if is_main_process():
                full_updated_latents = new_latents
                perturbation = self.H.imle_perturb_coef * torch.randn(
                    (self.sz, self.H.latent_dim),
                    device=self.device,
                    generator=self.generator_seed)
                full_updated_latents += perturbation
                comm_latents = full_updated_latents.to(self._comm_dtype)
            else:
                comm_latents = torch.empty(self.sz, self.H.latent_dim, dtype=self._comm_dtype, device=self.device)

            torch.distributed.broadcast(comm_latents, src=0)
            full_updated_latents = comm_latents.to(torch.float32)

            # Update last and current selected latents on all processes.
            self.last_selected_latents.copy_(self.selected_latents)
            self.selected_latents.copy_(full_updated_latents)

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        self.faiss_index_flat.reset()
