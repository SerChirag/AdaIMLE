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
            batch_size=self.H.imle_batch,      # Get 32 samples per batch
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

        # NumPy view for any CPU-side FAISS operations.
        self.dataset_proj = self.dataset_proj_torch.numpy()

    def sample(self, latents, gen, snoise=None):
        with torch.inference_mode():
            with autocast(device_type='cuda'):
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
            # Projections stored in reduced precision to save VRAM.
            self._local_pool_proj = torch.empty(
                (local_pool_size, self.dci_dim),
                device=self.device,
                dtype=self._comm_dtype,
            )

        # Regenerate the entire local pool each resample.
        self._local_pool_latents.normal_(mean=0.0, std=1.0, generator=self.generator_seed)

        # Process local chunk in batches.
        with torch.inference_mode():
            for start in range(0, local_pool_size, self.H.imle_batch):
                end = min(start + self.H.imle_batch, local_pool_size)
                batch_slice = slice(start, end)
                cur_latents = self._local_pool_latents[batch_slice]
                with autocast(device_type='cuda'):
                    outputs = gen(cur_latents, None)
                    if self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    else:
                        exit()
                    self._local_pool_proj[batch_slice].copy_(proj.to(self._comm_dtype))

        gen.train()
    

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
        Distributed force resampling: each rank generates and searches its own local
        pool against the full dataset. An all-gather of distances/indices finds the
        globally nearest pool sample per dataset point, eliminating the O(N*dci_dim)
        gather that previously centralised everything on rank 0.
        """
        if is_main_process():
            t1 = time.time()
            print("Starting pool resampling...")

        self.resample_pool(gen)

        if is_main_process():
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")

        self.selected_dists_tmp[:] = np.inf

        N = self.sz
        local_pool_size = self._local_pool_latents.shape[0]

        with torch.inference_mode():
            # Upload dataset features to GPU transiently (each rank does its own search).
            dataset_gpu = self.dataset_proj_torch.to(self.device)

            # Each rank: find nearest pool sample for every dataset point.
            local_distances, local_pool_indices = self.nn_search_batched(
                dataset_gpu, self._local_pool_proj.float()
            )
            del dataset_gpu

            # All-gather: every rank gets (distances, indices) from all other ranks.
            all_distances = [torch.empty(N, dtype=torch.float32, device=self.device)
                             for _ in range(self.world_size)]
            all_pool_indices = [torch.empty(N, dtype=torch.long, device=self.device)
                                for _ in range(self.world_size)]
            torch.distributed.all_gather(all_distances, local_distances.float())
            torch.distributed.all_gather(all_pool_indices, local_pool_indices.long())

            stacked_distances = torch.stack(all_distances, dim=0)   # (world_size, N)
            stacked_indices   = torch.stack(all_pool_indices, dim=0) # (world_size, N)

            # Global nearest: which rank and which local-pool index wins per dataset point.
            best_rank      = stacked_distances.argmin(dim=0)                             # (N,)
            arange_N       = torch.arange(N, device=self.device)
            best_local_idx = stacked_indices[best_rank, arange_N]                        # (N,)

            # Unique-usage fraction for logging (global pool indices are rank*size + local).
            global_pool_indices = best_rank.long() * local_pool_size + best_local_idx
            self.unique_indices = torch.unique(global_pool_indices).numel() / N

            # Each rank fills in the latents it owns; all_reduce SUM consolidates.
            selected = torch.zeros(N, self.H.latent_dim, dtype=torch.float32, device=self.device)
            my_mask = (best_rank == self.rank)
            my_ds_indices   = my_mask.nonzero(as_tuple=True)[0]
            if my_ds_indices.numel() > 0:
                my_pool_indices = best_local_idx[my_ds_indices]
                selected[my_ds_indices] = self._local_pool_latents[my_pool_indices].float()

            torch.distributed.all_reduce(selected, op=torch.distributed.ReduceOp.SUM)

            # Add perturbation (same generator state on all ranks → same noise).
            perturbation = self.H.imle_perturb_coef * torch.randn(
                N, self.H.latent_dim,
                device=self.device,
                generator=self.generator_seed,
            )
            selected.add_(perturbation)

            # Update CPU latent tables, reusing pre-allocated buffers.
            self.last_selected_latents.copy_(self.selected_latents)
            self.selected_latents_tmp.copy_(selected)
            self.selected_latents.copy_(self.selected_latents_tmp)

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        self.faiss_index_flat.reset()

