from curses import update_lines_cols
from math import comb, ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModel

from LPNet import LPNet
from helpers.utils import is_main_process, get_world_size, get_rank
from models import parse_layer_string
from helpers.angle_sampler import Angle_Generator
from torch import autocast
from diffusers import AutoencoderKL
import faiss

class Sampler:
    def __init__(self, H, sz, preprocess_fn):
        
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.world_size = get_world_size()
        self.rank = get_rank()

        self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduce=False).to(self.device)
        self.H = H
        self.latent_lr = H.latent_lr
        self.sz = sz
        self.entire_ds = torch.arange(sz)
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.last_selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.selected_latents_tmp = torch.empty([sz, H.latent_dim], dtype=torch.float32)

        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))

        self.selected_dists = torch.empty([sz], dtype=torch.float32)
        self.selected_dists[:] = np.inf
        self.selected_dists_tmp = torch.empty([sz], dtype=torch.float32)

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float32)

        self.pool_latents = None

        self.l2_projection = None

        self.vae = AutoencoderKL.from_pretrained("zelaki/eq-vae")
        self.vae.to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        fake = torch.zeros(1, 3, H.image_size, H.image_size, device=self.device)

        torch.distributed.barrier()

        if(H.search_type == 'l2'):
            with autocast(device_type='cuda'):
                fake = self.vae.encode(fake).latent_dist.sample()
                fake = fake * self.vae.config.scaling_factor

            interpolated = fake.reshape(fake.shape[0],-1)
            # self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim, device=self.device), p=2, dim=1)
            sum_dims = interpolated.shape[1]

        else:
            exit()

        self.dci_dim = sum_dims

        self.dataset_proj = torch.empty([sz, sum_dims], dtype=torch.float32, device='cpu')
        self.pool_samples_proj = None

        self.knn_ignore = H.knn_ignore
        self.ignore_radius = H.ignore_radius
        self.resample_angle = H.resample_angle

        self.total_excluded = 0
        self.total_excluded_percentage = 0
        self.ema_raw = 0.0
        self.ema_factor = 0.99
        self.ema_counter = 0
        self.mean_distance_nn = None

        self.dataset_size = sz
        self.db_iter = 0
        self.generator_seed = torch.Generator(device=self.device)         
        self.generator_seed.manual_seed(H.seed + self.rank)
        self.first_time = True

        self.faiss_res = faiss.StandardGpuResources()  # one per process
        index_flat = faiss.IndexFlatL2(self.dci_dim)  # identical API to IndexFlatL2
        self.gpu_index_flat = faiss.index_cpu_to_gpu(self.faiss_res, self.rank, index_flat)

    def preprocess_dino_tensor(self, inp):
        # x: [B, C, H, W], range [0, 1]

        x = (inp + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)

        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return (x - self.dino_mean) / self.dino_std

    
    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        interpolated = inp.reshape(inp.shape[0],-1)
        # interpolated = torch.mm(interpolated, self.l2_projection)
        # interpolated = F.normalize(interpolated, p=2, dim=1)
        return interpolated
    
    def get_image_feature(self, inp):
        inp = inp.permute(0, 3, 1, 2)
        interpolated = self.vae.encode(inp).latent_dist.sample()
        interpolated = interpolated * self.vae.config.scaling_factor
        return interpolated

    def init_projection(self, dataset):

        dataloader = DataLoader(
            dataset,
            batch_size=self.H.imle_batch,      # Get 32 samples per batch
            shuffle=False,  # No need to shuffle for projection
            num_workers=4      # Adjust based on your CPU
        )

        for ind, x in enumerate(dataloader):
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + x[0].shape[0])
            with autocast(device_type='cuda'):
                with torch.no_grad():
                    if(self.H.search_type == 'l2'):
                        inp = self.preprocess_fn(x)[1]
                        interpolated = self.get_image_feature(inp)
                        interpolated = interpolated.reshape(interpolated.shape[0],-1)
                        # interpolated = torch.mm(interpolated, self.l2_projection)
                        self.dataset_proj[batch_slice] = interpolated.cpu()
                    else:
                        exit()

        self.dataset_proj = self.dataset_proj.cpu().numpy().astype(np.float32)

    def sample(self, latents, gen, snoise=None):
        with torch.no_grad():
            with autocast(device_type='cuda'):
                latents = latents.to(self.device)
                px_z = gen(latents, None)
                x_hat = self.vae.decode(px_z / self.vae.config.scaling_factor).sample
                x_hat = x_hat.permute(0, 2, 3, 1)
                xhat = (x_hat + 1.0) * 127.5
                xhat = xhat.detach().cpu().numpy()
                xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
                return xhat

    def calc_loss(self, inp, tar, use_mean=True, logging=False):

        if use_mean:       
            l2_loss = torch.mean(self.l2_loss(inp, tar))
            return l2_loss

        else:
            l2_loss = torch.mean(self.l2_loss(inp, tar), dim=[1, 2, 3])
            return l2_loss
            

    def resample_pool(self, gen):

        gen.eval()   

        # Determine local pool size
        local_pool_size = ceil(self.pool_size / self.world_size)


        # Generate local pool latents and prepare container for projected features
        local_pool_latents = torch.randn((local_pool_size, self.H.latent_dim), 
                                         device=self.device, 
                                         generator=self.generator_seed)
        # Assuming pool_samples_proj is preallocated with shape (self.pool_size, projection_dim)

        local_pool_proj = torch.empty((local_pool_size, self.dci_dim), device=self.device)

        # Process local chunk in batches
        for j in range(local_pool_size // self.H.imle_batch):
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
            cur_latents = local_pool_latents[batch_slice]
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    outputs = gen(cur_latents, None)
                    if self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    elif self.H.search_type == 'combined':
                        proj = self.get_combined_feature(outputs, False)
                    else:
                        proj = self.get_combined_feature(outputs, False)
                    local_pool_proj[batch_slice] = proj

        torch.distributed.barrier()

        gathered_latents = [torch.empty_like(local_pool_latents) for _ in range(self.world_size)]
        gathered_proj = [torch.empty_like(local_pool_proj) for _ in range(self.world_size)]

        torch.distributed.all_gather(gathered_latents, local_pool_latents)
        torch.distributed.all_gather(gathered_proj, local_pool_proj)

        gen.train()

        torch.distributed.barrier()

        # Aggregate the full pool latents and projected features
        self.pool_latents = torch.cat(gathered_latents, dim=0).to('cpu')
        self.pool_samples_proj = torch.cat(gathered_proj, dim=0).to('cpu')

    def _sync_union_indices(self, local_tensor: torch.Tensor) -> torch.Tensor:
        """Synchronize a union of indices across ranks — works correctly under NCCL by broadcasting size and values separately."""
        send = local_tensor.cpu().tolist()

        if is_main_process():
            gathered = [None for _ in range(self.world_size)]
        else:
            gathered = None

        torch.distributed.gather_object(send, gathered, dst=0)

        if is_main_process():
            union_set = set()
            for item in gathered:
                union_set.update(item)
            sorted_union = sorted(union_set)
            global_len = torch.tensor([len(sorted_union)], dtype=torch.long, device=self.device)
            global_union = torch.tensor(sorted_union, dtype=torch.long, device=self.device)
        else:
            global_len = torch.empty(1, dtype=torch.long, device=self.device)
            global_union = None  # will be allocated after length broadcast

        # Step 1: Broadcast length
        torch.distributed.broadcast(global_len, src=0)
        union_size = global_len.item()

        # Step 2: Broadcast tensor
        if not is_main_process():
            global_union = torch.empty(union_size, dtype=torch.long, device=self.device)

        torch.distributed.broadcast(global_union, src=0)
        torch.distributed.barrier()  # Optional sync point

        return global_union

    def interpolate_latents(self, old_latents, new_latents, step=0.1):
        latents_interpolate = (1 - step) * old_latents.to(self.device) + step * new_latents
        normalized_latents = F.normalize(latents_interpolate, p=2, dim=1)
        random_gaussian_rvs = torch.randn_like(latents_interpolate)
        norms = torch.norm(random_gaussian_rvs, p=2, dim=1, keepdim=True)
        normalized_latents = normalized_latents * norms
        return normalized_latents

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
        torch.distributed.barrier()  # Ensure all processes complete the pool resample

        if(is_main_process()):
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")
        
        torch.cuda.empty_cache()

        self.selected_dists_tmp[:] = np.inf

        with torch.no_grad():
            # Total number of dataset samples.
            total_datapoints = self.dataset_proj.shape[0]

            # --------------------
            # Partition the dataset features so each process works on a different chunk.
            chunk_size = total_datapoints // self.world_size
            remainder = total_datapoints % self.world_size
            if self.rank < remainder:
                local_size = chunk_size + 1
                local_start = self.rank * local_size
            else:
                local_size = chunk_size
                local_start = self.rank * local_size + remainder
            local_end = min(local_start + local_size, self.sz)

            # Obtain the full dataset features (on CPU) and then slice locally.
            local_ds_feats = self.dataset_proj[local_start:local_end]

            # Pool features (as computed from resample_pool).
            pool_feats = self.pool_samples_proj.cpu().numpy().astype(np.float32)
            feature_dim = pool_feats.shape[1]

            # --------------------
            # Build FAISS index on global pool features.

            self.gpu_index_flat.add(pool_feats)  # add entire pool

            if(self.H.use_rsimle):
                # If using RSIMLE, we need to reset the index to avoid accumulating entries.
                distances, indices = self.gpu_index_flat.search(local_ds_feats, self.H.rs_knn_ignore)
                local_distances = torch.from_numpy(distances).squeeze(1)  # (local_size,)
                local_indices   = torch.from_numpy(indices).squeeze(1)    # (local_size,)
                easy_mask = local_distances < self.H.rs_radius
                local_easy = torch.unique(local_indices[easy_mask])  # 1‑D tensor of pool indices to drop
                # if is_main_process():
                torch.distributed.barrier()  # Ensure all processes complete the search

                global_easy = self._sync_union_indices(local_easy)

                percent_curr = global_easy.numel() / self.pool_latents.shape[0]
                self.ema_raw = self.ema_factor * self.ema_raw + (1 - self.ema_factor) * percent_curr
                self.total_excluded_percentage = self.ema_raw / (1 - self.ema_factor ** (self.ema_counter + 1))  # apply correction only here
                self.ema_counter += 1

                if global_easy.numel() > 0:
                    keep_mask = torch.ones(pool_feats.shape[0], dtype=torch.bool)
                    keep_mask[global_easy] = False
                    pool_feats = pool_feats[keep_mask]
                    self.pool_samples_proj = self.pool_samples_proj[keep_mask]
                    self.pool_latents = self.pool_latents[keep_mask]
                    self.gpu_index_flat.reset()  # Reset the index to avoid accumulating entries
                    self.gpu_index_flat.add(pool_feats)
                
                torch.distributed.barrier()  # Ensure synchronization before leaving the function


            # Perform NN search for the local chunk. Returns arrays of shape (local_size, 1).
            distances, indices = self.gpu_index_flat.search(local_ds_feats, 1)
            local_distances = torch.from_numpy(distances).squeeze(1)  # (local_size,)
            local_indices   = torch.from_numpy(indices).squeeze(1)    # (local_size,)
            self.mean_distance_nn = local_distances.mean().item()

            # if is_main_process():
            #     print(f"Mean distance NN: {self.mean_distance_nn:.4f}")
            #     print(f"Min distance NN: {local_distances.min().item():.4f}")
            #     print(f"Min distance NN: {local_distances.max().item():.4f}")
            #     print(f"Total excluded percentage: {self.total_excluded_percentage:.4f}")


            # Get current temporary distances for the local slice.
            local_current_dists = self.selected_dists_tmp[local_start:local_end].clone()
            # Determine which samples need update.
            need_update = local_distances < local_current_dists

            # Prepare local updated arrays.
            local_updated_dists = local_current_dists.clone()
            local_updated_latents = self.selected_latents_tmp[local_start:local_end].clone()

            if need_update.sum().item() > 0:
                # Fetch new latents from the pool for samples that need update.
                new_latents = self.pool_latents[local_indices[need_update]].clone()
                # Add random perturbation.
            
                local_updated_dists[need_update] = local_distances[need_update]
                local_updated_latents[need_update] = new_latents
                
            if is_main_process():
                gathered_dists = [None for _ in range(self.world_size)]
                gathered_latents = [None for _ in range(self.world_size)]
            else:
                gathered_dists = None
                gathered_latents = None

            torch.distributed.gather_object(local_updated_dists, gathered_dists, dst=0)
            torch.distributed.gather_object(local_updated_latents, gathered_latents, dst=0)

            torch.distributed.barrier()  # Ensure all processes complete the gather

            if is_main_process():
                full_updated_dists = torch.cat(gathered_dists, dim=0).to(self.device)
                full_updated_latents = torch.cat(gathered_latents, dim=0).to(self.device)
                perturbation = self.H.imle_perturb_coef * torch.randn(
                    (self.sz, self.H.latent_dim), 
                    device=self.device,
                    generator=self.generator_seed)
                
                if(self.H.use_interpolate_latents):
                    if(self.first_time):
                        self.first_time = False
                    else:
                        full_updated_latents = self.interpolate_latents(self.last_selected_latents, full_updated_latents, step=self.H.latent_interpolate_step)

                full_updated_latents += perturbation

            else:
                full_updated_dists = torch.empty(self.sz, dtype=torch.float32, device=self.device)
                full_updated_latents = torch.empty(self.sz, self.H.latent_dim, dtype=torch.float32, device=self.device)

            torch.distributed.barrier()

            torch.distributed.broadcast(full_updated_dists, src=0)
            torch.distributed.broadcast(full_updated_latents, src=0)

            torch.distributed.barrier()


            # Move the broadcasted results to CPU if desired.
            self.selected_dists_tmp = full_updated_dists.cpu()
            self.selected_latents_tmp = full_updated_latents.cpu()

            # Update last and current selected latents on all processes.
            self.last_selected_latents = self.selected_latents.clone()
            self.selected_latents = self.selected_latents_tmp.clone()

        torch.distributed.barrier()  # Ensure synchronization before leaving the function
        self.gpu_index_flat.reset()
