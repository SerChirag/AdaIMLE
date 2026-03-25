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

class Sampler:
    def __init__(self, H, sz, preprocess_fn):
        
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
        self.autoencoder = load_autoencoder(H, self.device)
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

        self.faiss_use_cpu = bool(getattr(H, 'faiss_use_cpu', False))
        self.faiss_res = None
        if self.faiss_use_cpu:
            self.faiss_index_flat = faiss.IndexFlatL2(self.dci_dim)
        else:
            self.faiss_res = faiss.StandardGpuResources()  # one per process
            index_flat = faiss.IndexFlatL2(self.dci_dim)
            dev_id = torch.cuda.current_device()
            self.faiss_index_flat = faiss.index_cpu_to_gpu(self.faiss_res, dev_id, index_flat)

    
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

        # Keep a torch tensor for fast indexed target lookup in training,
        # and a NumPy view for FAISS nearest-neighbor search.
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
            self._local_pool_proj = torch.empty((local_pool_size, self.dci_dim), device=self.device)
            self._local_pool_combined = torch.empty((local_pool_size, self.H.latent_dim + self.dci_dim), device=self.device)
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
                with autocast(device_type='cuda'):
                    outputs = gen(cur_latents, None)
                    if self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    else:
                        exit()
                    self._local_pool_proj[batch_slice] = proj

        # One collective for both latents and projections to reduce comm overhead.
        self._local_pool_combined[:, :self.H.latent_dim].copy_(self._local_pool_latents)
        self._local_pool_combined[:, self.H.latent_dim:].copy_(self._local_pool_proj)
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
            self.pool_latents = full_combined[:, :self.H.latent_dim].cpu()
            self.pool_samples_proj = full_combined[:, self.H.latent_dim:].cpu()
    

    def nn_search_batched(self, queries, dataset):
        """
        Hard-first greedy Top-K matching (unique when possible) using self.faiss_index_flat.

        RS-IMLE Logic:
        Prior to matching, if self.ignore_radius > 0, we identify dataset samples (pool latents)
        that are too close to queries (dataset latents). We drop these from consideration so they
        are not selected. To ensure we can assign one latent per query, we guarantee that the dataset
        retains at least Nq samples (dropping the closest ones first).

        Returns:
            distances: (Nq,) torch.float32   # squared L2 from FAISS
            indices:   (Nq,) torch.long
        """
        topk = self.H.imle_db_topk
        tie_shuffle = True  # avoid ordering bias for equal/near-equal margins

        Nq = queries.shape[0]
        Nd = dataset.shape[0]
        if Nq == 0:
            return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)

        if isinstance(dataset, torch.Tensor):
            dataset_np = np.ascontiguousarray(dataset.detach().cpu().numpy(), dtype=np.float32)
        else:
            dataset_np = np.ascontiguousarray(dataset, dtype=np.float32)

        if isinstance(queries, torch.Tensor):
            queries_np = np.ascontiguousarray(queries.detach().cpu().numpy(), dtype=np.float32)
        else:
            queries_np = np.ascontiguousarray(queries, dtype=np.float32)


        # ---- Build index once on the full dataset ----
        self.faiss_index_flat.reset()
        self.faiss_index_flat.add(dataset_np)

        # ---- RS-IMLE logic: reject dataset samples that are too close to queries ----
        original_indices_map = None
        if getattr(self.H, 'use_rs_imle', False):
            k_ignore = int(min(max(1, getattr(self.H, 'rs_knn_ignore', 10)), Nd))
            rs_radius = getattr(self.H, 'rs_radius', 10.0)
            distances, indices = self.faiss_index_flat.search(queries_np, k_ignore)
            easy_mask = distances < rs_radius
            
            flat_indices = indices[easy_mask]
            flat_distances = distances[easy_mask]
            
            if len(flat_indices) > 0:
                min_dist = np.full(Nd, np.inf, dtype=np.float32)
                np.minimum.at(min_dist, flat_indices, flat_distances)
                
                too_close_indices = np.where(min_dist < np.inf)[0]
                max_drops = max(0, Nd - Nq)
                
                if len(too_close_indices) > max_drops:
                    # Drop the `max_drops` closest ones
                    sorted_by_dist = too_close_indices[np.argsort(min_dist[too_close_indices])]
                    drop_indices = sorted_by_dist[:max_drops]
                else:
                    drop_indices = too_close_indices

                if len(drop_indices) > 0:
                    self.total_excluded = len(drop_indices)
                    self.total_excluded_percentage = self.total_excluded / Nd

                    keep_mask = np.ones(Nd, dtype=bool)
                    keep_mask[drop_indices] = False
                    
                    dataset_np = dataset_np[keep_mask]
                    original_indices_map = np.where(keep_mask)[0]
                    Nd = dataset_np.shape[0]
                    
                    self.faiss_index_flat.reset()
                    self.faiss_index_flat.add(dataset_np)

        topk = int(min(max(1, topk), Nd))

        # ---- 1) Hardness (margin = d2 - d1) ----
        # Need k=2 even if topk==1, to get a margin; if Nd==1 margin is 0.
        if Nd >= 2:
            D2, _ = self.faiss_index_flat.search(queries_np, 2)  # (Nq,2)
            margin = D2[:, 0]
        else:
            margin = np.zeros(Nq, dtype=np.float32)

        if tie_shuffle:
            perm = np.random.permutation(Nq)
            order = perm[np.argsort(margin[perm], kind="stable")]
        else:
            order = np.argsort(margin, kind="stable")

        # ---- 2) Get Top-K candidate lists for all queries ----
        D, I = self.faiss_index_flat.search(queries_np, topk)  # (Nq,K), squared L2 + indices

        # ---- 3) Greedy unique assignment in hard-first order ----
        used = np.zeros(Nd, dtype=bool)

        out_idx = np.full(Nq, -1, dtype=np.int64)
        out_dst = np.full(Nq, np.inf, dtype=np.float32)

        for qi in order:
            cand = I[qi]   # (K,)
            cd   = D[qi]   # (K,)

            chosen = -1
            chosen_d = None

            # First unused among top-K (already sorted by distance)
            for k in range(topk):
                j = int(cand[k])
                if not used[j]:
                    chosen = j
                    chosen_d = float(cd[k])
                    used[j] = True
                    break

            # If no unused candidate exists, return 1-NN anyway (collision allowed)
            if chosen == -1:
                chosen = int(cand[0])
                chosen_d = float(cd[0])

            # Map the local pool index back to the original index if we dropped elements
            if original_indices_map is not None:
                chosen = int(original_indices_map[chosen])

            out_idx[qi] = chosen
            out_dst[qi] = chosen_d

        # ---- Cleanup ----
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

                local_ds_feats = self.dataset_proj

                # Pool features (as computed from resample_pool).
                pool_feats = self.pool_samples_proj.numpy()

                # Perform NN search for the local chunk. Returns arrays of shape (local_size, 1).
                local_distances, local_indices = self.nn_search_batched(local_ds_feats, pool_feats)

                # get count of unique indices for logging
                self.unique_indices = torch.unique(local_indices).numel() / self.sz

                new_latents = self.pool_latents[local_indices]

            if is_main_process():
                full_updated_latents = new_latents.to(self.device)
                perturbation = self.H.imle_perturb_coef * torch.randn(
                    (self.sz, self.H.latent_dim), 
                    device=self.device,
                    generator=self.generator_seed)
                full_updated_latents += perturbation
            else:
                full_updated_latents = torch.empty(self.sz, self.H.latent_dim, dtype=torch.float32, device=self.device)

            torch.distributed.broadcast(full_updated_latents, src=0)

            # Move the broadcasted results to CPU if desired.
            self.selected_latents_tmp = full_updated_latents.cpu()

            # Update last and current selected latents on all processes.
            self.last_selected_latents.copy_(self.selected_latents)
            self.selected_latents.copy_(self.selected_latents_tmp)

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        self.faiss_index_flat.reset()

