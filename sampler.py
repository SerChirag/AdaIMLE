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
        self.selected_latents_tmp = torch.empty([sz, H.latent_dim], dtype=torch.float32)

        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
        self.latent_spatial_size = int(getattr(H, 'latent_spatial_size', max(self.block_res)))

        self.selected_dists = torch.empty([sz], dtype=torch.float32)
        self.selected_dists[:] = np.inf
        self.selected_dists_tmp = torch.empty([sz], dtype=torch.float32)

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.latent_spatial_size, self.latent_spatial_size],
                                        dtype=torch.float32)

        self.pool_latents = None

        self.decode_for_metrics = bool(getattr(H, 'autoencoder_decode_for_metrics', True))
        self.autoencoder = autoencoder if autoencoder is not None else load_autoencoder(H, self.device)
        if is_main_process():
            ae_name = type(self.autoencoder).__name__
            ae_source = getattr(getattr(self.autoencoder, 'config', None), '_name_or_path', 'unknown')
            print(f'\n[autoencoder] Loaded {ae_name} from {ae_source}\n')
        if(H.compile):
            self.autoencoder = torch.compile(self.autoencoder) 

        self.autoencoder_native_latent_size = None
        fake_rgb = torch.zeros(1, 3, H.image_size, H.image_size, device=self.device)
        native_latents = encode_images_to_latents(self.autoencoder, fake_rgb, target_spatial=None)
        self.autoencoder_native_latent_size = (native_latents.shape[-2], native_latents.shape[-1])

        if H.search_type != 'l2':
            raise ValueError('This branch expects search_type=l2.')

        self.nn_search_batch = H.nn_search_batch

        self.l2_projection = None
        self.total_excluded = 0
        self.total_excluded_percentage = 0.0
        self.rs_current_radius = float(getattr(H, 'rs_radius', 100.0))
        self.rs_reject_ema_beta = float(getattr(H, 'rs_reject_ema_beta', 0.9))
        self.rs_reject_ema = 0.0
        self.rs_reject_ema_steps = 0
        self.rs_reject_ema_corrected = 0.0
        self.rs_radius_anneal_cooldown_rounds = int(getattr(H, 'rs_radius_anneal_cooldown_rounds', 20))
        self.rs_radius_anneal_cooldown_left = 0

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

        self.knn_ignore = H.knn_ignore
        self.ignore_radius = H.ignore_radius
        self.resample_angle = H.resample_angle

        self.total_excluded = 0
        self.total_excluded_percentage = 0

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
        index_flat = faiss.IndexFlatL2(self.dci_dim)
        dev_id = torch.cuda.current_device()
        self.faiss_index_flat = faiss.index_cpu_to_gpu(self.faiss_res, dev_id, index_flat)

    def state_dict(self):
        return {
            'total_excluded': int(self.total_excluded),
            'total_excluded_percentage': float(self.total_excluded_percentage),
            'rs_current_radius': float(self.rs_current_radius),
            'rs_reject_ema_beta': float(self.rs_reject_ema_beta),
            'rs_reject_ema': float(self.rs_reject_ema),
            'rs_reject_ema_steps': int(self.rs_reject_ema_steps),
            'rs_reject_ema_corrected': float(self.rs_reject_ema_corrected),
            'rs_radius_anneal_cooldown_rounds': int(self.rs_radius_anneal_cooldown_rounds),
            'rs_radius_anneal_cooldown_left': int(self.rs_radius_anneal_cooldown_left),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            return
        self.total_excluded = int(state.get('total_excluded', self.total_excluded))
        self.total_excluded_percentage = float(state.get('total_excluded_percentage', self.total_excluded_percentage))
        self.rs_current_radius = float(state.get('rs_current_radius', self.rs_current_radius))
        self.rs_reject_ema_beta = float(state.get('rs_reject_ema_beta', self.rs_reject_ema_beta))
        self.rs_reject_ema = float(state.get('rs_reject_ema', self.rs_reject_ema))
        self.rs_reject_ema_steps = int(state.get('rs_reject_ema_steps', self.rs_reject_ema_steps))
        self.rs_reject_ema_corrected = float(state.get('rs_reject_ema_corrected', self.rs_reject_ema_corrected))
        self.rs_radius_anneal_cooldown_rounds = int(state.get('rs_radius_anneal_cooldown_rounds', self.rs_radius_anneal_cooldown_rounds))
        self.rs_radius_anneal_cooldown_left = int(state.get('rs_radius_anneal_cooldown_left', self.rs_radius_anneal_cooldown_left))

    def _update_rs_rejection_ema(self, rejection_pct):
        # Bias-corrected EMA:
        # m_t = beta * m_{t-1} + (1-beta) * x_t
        # m_hat_t = m_t / (1 - beta^t)
        beta = min(max(self.rs_reject_ema_beta, 0.0), 0.999999)
        self.rs_reject_ema = beta * self.rs_reject_ema + (1.0 - beta) * float(rejection_pct)
        self.rs_reject_ema_steps += 1
        bias_correction = 1.0 - (beta ** self.rs_reject_ema_steps)
        if bias_correction <= 0.0:
            self.rs_reject_ema_corrected = self.rs_reject_ema
            return self.rs_reject_ema
        self.rs_reject_ema_corrected = self.rs_reject_ema / bias_correction
        return self.rs_reject_ema_corrected

    
    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)

        if inp.shape[1] == 3:
            inp = encode_images_to_latents(self.autoencoder, inp, target_spatial=(self.latent_spatial_size, self.latent_spatial_size))

        interpolated = inp.reshape(inp.shape[0],-1)
        # interpolated = F.normalize(interpolated, p=2, dim=1)
        return interpolated


    def init_projection(self, dataset):

        dataloader = DataLoader(
            dataset,
            batch_size=self.H.imle_batch
        )

        if(is_main_process()):
            print("Starting Initialization")

        with torch.inference_mode():

            for ind, x in tqdm(enumerate(dataloader), total=len(dataloader), desc="Initializing"):
                batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + x[0].shape[0])
                if(self.H.search_type == 'l2'):
                    self.dataset_proj_torch[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[1]).cpu()
                else:
                    exit()

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

        # Aggregate the full pool latents and projected features
        if self.rank == 0:
            full_combined = torch.cat(self._gathered_combined_main, dim=0)
            # Keep both latents and projections on GPU on rank 0.
            self.pool_latents = full_combined[:, :self.H.latent_dim].to(torch.float32)
            self.pool_samples_proj = full_combined[:, self.H.latent_dim:].to(torch.float32)
    

    def nn_search_batched(self, queries, dataset):
        """
        RS-IMLE search (when enabled):
        1) Query top-k (k = rs_knn_ignore).
        2) Remove samples inside rs_radius of any query.
        3) Query k=1 on the remaining samples.

        Non-RS path (use_rs_imle=False):
        Direct k=1 search on the full sample set.
        """
        if isinstance(queries, np.ndarray):
            queries_t = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to(self.device)
        else:
            queries_t = queries.to(self.device)

        if isinstance(dataset, np.ndarray):
            dataset_t = torch.from_numpy(np.ascontiguousarray(dataset, dtype=np.float32)).to(self.device)
        else:
            dataset_t = dataset.to(self.device)

        queries_t = queries_t.contiguous()
        dataset_t = dataset_t.contiguous()

        Nq = int(queries_t.shape[0])
        Nd = int(dataset_t.shape[0])
        if Nq == 0 or Nd == 0:
            return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)

        use_rs = bool(getattr(self.H, 'use_rs_imle', False))

        # Fast fallback path when RS-IMLE filtering is disabled.
        if not use_rs:
            self.total_excluded = 0
            self.total_excluded_percentage = 0.0
            self.rs_reject_ema = 0.0
            self.rs_reject_ema_steps = 0
            self.rs_reject_ema_corrected = 0.0
            self.faiss_index_flat.reset()
            self.faiss_index_flat.add(dataset_t)
            D1, I1 = self.faiss_index_flat.search(queries_t, 1)
            self.faiss_index_flat.reset()
            return D1.squeeze(1).to(torch.float32), I1.squeeze(1).to(torch.long)

        topk = int(min(max(1, int(getattr(self.H, 'rs_knn_ignore', 10))), Nd))
        epsilon = float(self.rs_current_radius)

        # Step 1: top-k search on the full sample set.
        self.faiss_index_flat.reset()
        self.faiss_index_flat.add(dataset_t)
        Dk, Ik = self.faiss_index_flat.search(queries_t, topk)

        # Step 2: reject any sample that appears within epsilon of any query.
        keep_mask = torch.ones(Nd, dtype=torch.bool, device=dataset_t.device)
        close_indices = None
        close_distances = None
        if epsilon > 0.0:
            reject_mask = torch.zeros(Nd, dtype=torch.bool, device=dataset_t.device)
            close_indices = Ik[Dk < epsilon]
            close_distances = Dk[Dk < epsilon]
            if close_indices.numel() > 0:
                reject_mask[close_indices.long()] = True
                keep_mask = ~reject_mask

        kept_count = int(keep_mask.sum().item())
        original_indices_map = None

        # If everything was rejected, keep one least-close sample so FAISS index is valid.
        if kept_count == 0 and close_indices is not None and close_indices.numel() > 0:
            min_dist = torch.full((Nd,), float('inf'), dtype=torch.float32, device=dataset_t.device)
            min_dist.scatter_reduce_(0, close_indices.long(), close_distances.float(), reduce='amin', include_self=True)
            recover_idx = torch.argmax(min_dist).item()
            keep_mask[recover_idx] = True
            kept_count = 1

        # Keep at least one candidate to avoid an empty FAISS index.
        if kept_count > 0 and kept_count < Nd:
            self.total_excluded = Nd - kept_count
            self.total_excluded_percentage = (self.total_excluded * 100.0) / Nd
            dataset_kept = dataset_t[keep_mask]
            original_indices_map = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        else:
            self.total_excluded = 0
            self.total_excluded_percentage = 0.0
            dataset_kept = dataset_t

        # Update rejection EMA and anneal epsilon if the EMA crosses threshold.
        reject_ema = self._update_rs_rejection_ema(self.total_excluded_percentage)
        anneal_threshold = float(getattr(self.H, 'rs_reject_ema_threshold', 50.0))
        anneal_factor = float(getattr(self.H, 'rs_radius_anneal_factor', 0.9))
        min_radius = float(getattr(self.H, 'rs_radius_min', 0.0))
        if self.rs_radius_anneal_cooldown_left > 0:
            self.rs_radius_anneal_cooldown_left -= 1
        elif reject_ema >= anneal_threshold:
            new_radius = max(min_radius, self.rs_current_radius * anneal_factor)
            if new_radius < self.rs_current_radius:
                self.rs_current_radius = new_radius
                # Reset EMA after anneal to avoid immediate repeated annealing.
                self.rs_reject_ema = 0.0
                self.rs_reject_ema_steps = 0
                self.rs_reject_ema_corrected = 0.0
                self.rs_radius_anneal_cooldown_left = self.rs_radius_anneal_cooldown_rounds

        # Step 3: final k=1 search on the remaining sample set.
        self.faiss_index_flat.reset()
        self.faiss_index_flat.add(dataset_kept)
        D1, I1 = self.faiss_index_flat.search(queries_t, 1)

        out_dst = D1.squeeze(1).to(torch.float32)
        out_idx = I1.squeeze(1).to(torch.long)

        if original_indices_map is not None:
            out_idx = original_indices_map.index_select(0, out_idx)

        self.faiss_index_flat.reset()
        return out_dst, out_idx


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

            # Reuse the preallocated CPU buffer instead of allocating a new tensor every resample.
            self.selected_latents_tmp.copy_(full_updated_latents)

            # Update last and current selected latents on all processes.
            self.last_selected_latents.copy_(self.selected_latents)
            self.selected_latents.copy_(self.selected_latents_tmp)

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        self.faiss_index_flat.reset()
