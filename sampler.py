from curses import update_lines_cols
from math import comb, ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from helpers.utils import is_dist_avail_and_initialized, is_main_process, get_world_size, get_rank, safe_barrier
from models import parse_layer_string
from helpers.angle_sampler import Angle_Generator
from torch import autocast
import faiss
from tqdm import tqdm
from diffusers import AutoencoderKL

class Sampler:
    def __init__(self, H, sz, preprocess_fn):
        
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.nn_search_batch = H.nn_search_batch

        self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size
        self.reverse_pool_size = ceil(int(H.reverse_force_factor * sz) / H.imle_db_size) * H.imle_db_size
        self.total_reverse_count = int(H.reverse_force_factor * sz)
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduce='mean').to(self.device)
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
        self.reverse_pool_latents = None
        self.reverse_indices = None

        self.projections = []
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae.to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        if(H.compile):
            self.vae = torch.compile(self.vae)


        self.l2_projection = None

        fake = torch.randn(1, 3, H.image_size, H.image_size, device=self.device)

        safe_barrier()

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
        self.reverse_pool_samples_proj = None

        self.knn_ignore = H.knn_ignore
        self.ignore_radius = H.ignore_radius
        self.resample_angle = H.resample_angle

        self.total_excluded = 0
        self.total_excluded_percentage = 0

        self.dataset_size = sz
        self.db_iter = 0
        self.generator_seed = torch.Generator(device=self.device)         
        self.generator_seed.manual_seed(H.seed + self.rank)

        self.faiss_res = faiss.StandardGpuResources()  # one per process
        index_flat = faiss.IndexFlatL2(self.dci_dim)  # identical API to IndexFlatL2
        self.gpu_index_flat = index_flat
    
    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        interpolated = inp.reshape(inp.shape[0],-1)
        return interpolated

    def get_image_feature(self, inp):
        with autocast(device_type='cuda'):
            with torch.no_grad():
                inp = inp.permute(0, 3, 1, 2)
                interpolated = self.vae.encode(inp).latent_dist.sample()
                interpolated = interpolated * self.vae.config.scaling_factor
                return interpolated

    def init_projection(self, dataset):

        dataloader = DataLoader(
            dataset,
            batch_size=self.H.imle_batch,      # Get 32 samples per batch
            shuffle=False,
        )

        if(is_main_process()):
            print("Starting Initialization")

        for ind, x in tqdm(enumerate(dataloader), total=len(dataloader), desc="Initializing"):
            batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + x[0].shape[0])
            if(self.H.search_type == 'l2'):
                inp = self.preprocess_fn(x)[1]
                interpolated = self.get_image_feature(inp)
                interpolated = interpolated.reshape(interpolated.shape[0],-1)
                self.dataset_proj[batch_slice] = interpolated.cpu()
            else:
                exit()

        self.dataset_proj = self.dataset_proj.cpu().detach().numpy().astype(np.float32)
        self.init_pca()
    
    def init_pca(self):
        flattened_data = torch.from_numpy(self.dataset_proj).to(self.device)
        flattened_data = flattened_data.view(flattened_data.shape[0], -1)
        data_mean = torch.mean(flattened_data, dim=0)
        centered = flattened_data - data_mean
        cov = centered.T @ centered / (centered.shape[0] - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = torch.argsort(eigenvalues, descending=True)
        self.pca_components = eigenvectors[:, idx].T        
        self.pca_mean = data_mean
        self.pca_eigenvalues = eigenvalues[idx].clamp(min=0)



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

    def calc_loss(self, inp, tar, logging=False):

        # l2_loss = self.l2_loss(inp.reshape(inp.shape[0], -1), tar).mean()
        if(self.H.loss_type == 'l2'):
            input_reshaped = inp.reshape(inp.shape[0], -1)
            l2_loss = self.l2_loss(input_reshaped, tar)
            return l2_loss
        elif(self.H.loss_type == 'pca'):
            pca_loss = self.pca_loss(inp, tar)
            return pca_loss
        else:
            exit()


    def pca_loss(self, inp, tar):
        inp_reshaped = inp.reshape(inp.shape[0], -1)
        tar_reshaped = tar.reshape(tar.shape[0], -1)
        inp_centered = inp_reshaped - self.pca_mean
        tar_centered = tar_reshaped - self.pca_mean
        inp_pca = inp_centered @ self.pca_components    
        tar_pca = tar_centered @ self.pca_components
        weights = self.pca_eigenvalues / self.pca_eigenvalues.sum()
        weighted_diff = ((inp_pca - tar_pca) ** 2 * weights).sum(dim=1)
        return weighted_diff.mean()

            
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

        num_batches = (local_pool_size + self.H.imle_batch - 1) // self.H.imle_batch

        # Process local chunk in batches
        for j in range(num_batches):
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
            cur_latents = local_pool_latents[batch_slice]
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    outputs = gen(cur_latents, None)
                    if self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    else:
                        exit()

                    local_pool_proj[batch_slice] = proj.to('cpu')

        safe_barrier()
        gathered_latents = [torch.empty_like(local_pool_latents) for _ in range(self.world_size)]
        gathered_proj = [torch.empty_like(local_pool_proj) for _ in range(self.world_size)]

        torch.distributed.all_gather(gathered_latents, local_pool_latents)
        torch.distributed.all_gather(gathered_proj, local_pool_proj)

        gen.train()

        safe_barrier()
        # Aggregate the full pool latents and projected features
        self.pool_latents = torch.cat(gathered_latents, dim=0).to('cpu')
        self.pool_samples_proj = torch.cat(gathered_proj, dim=0).to('cpu')
    
    def resample_reverse_pool(self, gen):

        gen.eval()   

        # Determine local pool size
        local_reverse_pool_size = ceil(self.reverse_pool_size / self.world_size)


        # Generate local pool latents and prepare container for projected features
        local_reverse_pool_latents = torch.randn((local_reverse_pool_size, self.H.latent_dim), 
                                         device=self.device, 
                                         generator=self.generator_seed)
        # Assuming pool_samples_proj is preallocated with shape (self.pool_size, projection_dim)

        local_reverse_pool_proj = torch.empty((local_reverse_pool_size, self.dci_dim), device=self.device)

        # print(f'Local reverse pool size is {local_reverse_pool_size}')
        # print(f'Reverse pool size is {self.reverse_pool_size}')

        num_batches = (local_reverse_pool_size + self.H.imle_batch - 1) // self.H.imle_batch

        for j in range(num_batches):
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
            cur_latents = local_reverse_pool_latents[batch_slice]

            with torch.no_grad():
                with autocast(device_type='cuda'):
                    outputs = gen(cur_latents, None)
                    if self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    else:
                        exit()
                    local_reverse_pool_proj[batch_slice] = proj

        safe_barrier()
        gathered_latents = [torch.empty_like(local_reverse_pool_latents) for _ in range(self.world_size)]
        gathered_proj = [torch.empty_like(local_reverse_pool_proj) for _ in range(self.world_size)]

        torch.distributed.all_gather(gathered_latents, local_reverse_pool_latents)
        torch.distributed.all_gather(gathered_proj, local_reverse_pool_proj)

        gen.train()

        safe_barrier()
        # Aggregate the full pool latents and projected features
        self.reverse_pool_latents = torch.cat(gathered_latents, dim=0).to('cpu')
        self.reverse_pool_samples_proj = torch.cat(gathered_proj, dim=0).to('cpu')

        self.reverse_pool_latents = self.reverse_pool_latents[:self.total_reverse_count]
        self.reverse_pool_samples_proj = self.reverse_pool_samples_proj[:self.total_reverse_count]

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
        safe_barrier()  # Ensure all processes complete before returning

        return global_union


    def nn_search_batched(self, queries, dataset):
        """
        Perform nearest-neighbor search in batches using self.gpu_index_flat (FAISS).
        Each dataset sample can be matched only once (greedy removal).
        Random permutation is applied to remove ordering bias.

        Args:
            queries: (Nq, D) torch.Tensor of query vectors (CPU or CUDA).
            dataset: (Nd, D) torch.Tensor of database vectors (CPU or CUDA).
            batch_size: int, number of queries per batch.

        Returns:
            distances: torch.FloatTensor (Nq,)
            indices: torch.LongTensor (Nq,)
        """

        # Convert to NumPy float32 arrays for FAISS
        q_np = np.ascontiguousarray(queries, dtype=np.float32)
        db_np = np.ascontiguousarray(dataset, dtype=np.float32)
        Nd = db_np.shape[0]

        # Prepare result buffers
        Nq = q_np.shape[0]
        all_dists = np.full(Nq, np.inf, dtype=np.float32)
        all_indices = np.full(Nq, -1, dtype=np.int64)

        # Track availability
        available = np.ones(Nd, dtype=bool)

        # Randomize query order to remove bias
        perm = np.random.permutation(Nq)

        self.gpu_index_flat.reset()

        for start in range(0, Nq, self.nn_search_batch):
            end = min(start + self.nn_search_batch, Nq)
            batch_ids = perm[start:end]
            q_batch = q_np[batch_ids]

            # Restrict search to remaining dataset samples
            valid_mask = np.flatnonzero(available)
            if valid_mask.size == 0:
                break

            db_valid = db_np[valid_mask]
            dim = db_valid.shape[1]

            # Rebuild FAISS index for current available set
            self.gpu_index_flat.reset()
            self.gpu_index_flat.add(db_valid)

            # Perform search
            D, I = self.gpu_index_flat.search(q_batch, 1)
            D = D.squeeze(1)
            I = I.squeeze(1)
            global_idx = valid_mask[I]

            # Record results
            all_dists[batch_ids] = D
            all_indices[batch_ids] = global_idx

            # Remove matched samples from availability mask
            available[global_idx] = False

        # Return as torch tensors
        self.gpu_index_flat.reset()

        distances = torch.from_numpy(all_dists)
        indices = torch.from_numpy(all_indices)
        return distances, indices



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
        safe_barrier()  # Ensure all processes complete the pool resample

        if(is_main_process()):
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")
        
        torch.cuda.empty_cache()

        self.selected_dists_tmp[:] = np.inf

        with torch.no_grad():

            if(is_main_process()):

                local_ds_feats = np.ascontiguousarray(self.dataset_proj, dtype=np.float32)

                # Pool features (as computed from resample_pool).
                pool_feats = np.ascontiguousarray(self.pool_samples_proj.cpu().numpy().astype(np.float32), dtype=np.float32)

                # Perform NN search for the local chunk. Returns arrays of shape (local_size, 1).
                local_distances, local_indices = self.nn_search_batched(local_ds_feats, pool_feats)

                new_latents = self.pool_latents[local_indices].clone()
            
            safe_barrier()  # Ensure all processes complete the gather

            if is_main_process():
                full_updated_latents = new_latents.to(self.device)
                perturbation = self.H.imle_perturb_coef * torch.randn(
                    (self.sz, self.H.latent_dim), 
                    device=self.device,
                    generator=self.generator_seed)
                full_updated_latents += perturbation
            else:
                full_updated_latents = torch.empty(self.sz, self.H.latent_dim, dtype=torch.float32, device=self.device)

            safe_barrier()

            torch.distributed.broadcast(full_updated_latents, src=0)

            safe_barrier()

            # Move the broadcasted results to CPU if desired.
            self.selected_latents_tmp = full_updated_latents.cpu()

            # Update last and current selected latents on all processes.
            self.last_selected_latents = self.selected_latents.clone()
            self.selected_latents = self.selected_latents_tmp.clone()

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        safe_barrier()  # Ensure synchronization before leaving the function
        self.gpu_index_flat.reset()


    def imle_sample_force_reverse(self, gen, to_update=None):
        """
        Optimized force resampling routine using FAISS for batched nearest-neighbor search.
        In a DDP setting, each process handles a different subset of the dataset features,
        performs NN search locally, and then the results are merged and broadcast so that
        all processes end up with the complete global results.
        """
        if is_main_process():
            t1 = time.time()
            print("Starting reverse pool resampling...")

        # Resample pool first (each process contributes its part);
        # this updates self.pool_samples_proj and self.pool_latents.
        self.resample_reverse_pool(gen)
        safe_barrier()  # Ensure all processes complete the pool resample
        # self.reverse_pool_latents = self.pool_latents
        # self.reverse_pool_samples_proj = self.pool_samples_proj

        if(is_main_process()):
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")
        
        torch.cuda.empty_cache()

        with torch.no_grad():

            if(is_main_process()):

                # Obtain the full dataset features (on CPU) and then slice locally.
                local_query_feats = np.ascontiguousarray(
                    self.reverse_pool_samples_proj.cpu().numpy(),
                    dtype=np.float32
                )

                # Pool features (as computed from resample_pool).
                dataset_feats = np.ascontiguousarray(self.dataset_proj.copy(), dtype=np.float32)

                distances, local_indices = self.nn_search_batched(local_query_feats, dataset_feats)
            
            safe_barrier()  # Ensure all processes complete the gather

            if(is_main_process()):
                global_indices = local_indices.to(self.device)
            else:
                global_indices = torch.empty(self.reverse_pool_samples_proj.shape[0], dtype=torch.long, device=self.device)

            safe_barrier()

            torch.distributed.broadcast(global_indices, src=0)

            safe_barrier()

            self.reverse_indices = global_indices.cpu()

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")
                print(f"Unique indices count: {self.reverse_indices.unique().shape[0]}")

        safe_barrier()  # Ensure synchronization before leaving the function
        self.gpu_index_flat.reset()
