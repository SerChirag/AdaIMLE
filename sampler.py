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
        self._dataset_proj_gpu = None
        self._local_pool_latents = None
        self._local_pool_combined = None
        self.local_pool_proj = None
        self.local_pool_offset = 0

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

        # Regenerate the entire local pool each resample.
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
                    proj = self.get_l2_feature(outputs, False)
                    self._local_pool_combined[batch_slice, self.H.latent_dim:].copy_(proj.to(self._comm_dtype))

        gen.train()

        # Each rank keeps its own local projection slice for distributed NN search.
        self.local_pool_proj = self._local_pool_combined[:, self.H.latent_dim:].to(torch.float32)
        self.local_pool_offset = self.rank * local_pool_size

        # All-gather latents so every rank can look up the winning latent after sync.
        local_latents_f32 = self._local_pool_combined[:, :self.H.latent_dim].to(torch.float32).contiguous()
        gathered_latents = [torch.empty_like(local_latents_f32) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered_latents, local_latents_f32)
        self.pool_latents = torch.cat(gathered_latents, dim=0)  # [full_pool_size, latent_dim]
    

    def nn_search_batched(self, queries_t, dataset_t):
        """k=1 exact L2 nearest-neighbour search via FAISS on the local pool slice."""
        queries_t = queries_t.contiguous()
        dataset_t = dataset_t.contiguous()
        self.faiss_index_flat.reset()
        self.faiss_index_flat.add(dataset_t)
        D1, I1 = self.faiss_index_flat.search(queries_t, 1)
        self.faiss_index_flat.reset()
        return D1.squeeze(1).to(torch.float32), I1.squeeze(1).to(torch.long)


    def imle_sample_force(self, gen, to_update=None):
        """
        Force resampling with distributed NN search: each rank searches its local pool
        slice, results are synced via all_gather, and the global best match is selected.
        """
        t1 = time.time()
        if is_main_process():
            print("Starting pool resampling...")

        # Each rank generates its local pool slice and all-gathers latents.
        self.resample_pool(gen)

        if is_main_process():
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")

        self.selected_dists_tmp[:] = np.inf

        with torch.inference_mode():

            # All ranks need the dataset projections on GPU.
            if (
                self._dataset_proj_gpu is None
                or self._dataset_proj_gpu.shape != self.dataset_proj_torch.shape
                or self._dataset_proj_gpu.device != self.device
            ):
                self._dataset_proj_gpu = self.dataset_proj_torch.to(self.device, non_blocking=True)

            # Each rank searches its local pool slice.
            local_dist, local_idx = self.nn_search_batched(self._dataset_proj_gpu, self.local_pool_proj)

            # Map local indices to global pool indices.
            local_global_idx = local_idx + self.local_pool_offset

            # All-gather distances and global indices from all ranks.
            all_dists = [torch.empty(self.sz, dtype=torch.float32, device=self.device) for _ in range(self.world_size)]
            all_idxs  = [torch.empty(self.sz, dtype=torch.long,    device=self.device) for _ in range(self.world_size)]
            torch.distributed.all_gather(all_dists, local_dist)
            torch.distributed.all_gather(all_idxs,  local_global_idx)

            # Pick the closest match across all ranks.
            all_dists_t = torch.stack(all_dists, dim=0)           # [world_size, sz]
            all_idxs_t  = torch.stack(all_idxs,  dim=0)           # [world_size, sz]
            winner      = all_dists_t.argmin(dim=0)                # [sz]
            best_global_idx = all_idxs_t.gather(0, winner.unsqueeze(0)).squeeze(0)  # [sz]

            # Log unique coverage (rank 0 only).
            if is_main_process():
                self.unique_indices = torch.unique(best_global_idx).numel() / self.sz

            # Every rank has pool_latents, so lookup works everywhere.
            new_latents = self.pool_latents.index_select(0, best_global_idx)

            perturbation = self.H.imle_perturb_coef * torch.randn(
                (self.sz, self.H.latent_dim), device=self.device, generator=self.generator_seed
            )
            full_updated_latents = new_latents + perturbation

            # Update last and current selected latents on all processes.
            self.last_selected_latents.copy_(self.selected_latents)
            self.selected_latents.copy_(full_updated_latents)

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")
