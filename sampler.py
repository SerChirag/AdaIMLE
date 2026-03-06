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
        self.l2_loss = torch.nn.MSELoss(reduce=False).to(self.device)
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

        self.dataset_proj = torch.empty([sz, sum_dims], dtype=torch.float32, device='cpu')
        self.pool_samples_proj = None

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
        dev_id = torch.cuda.current_device()
        self.gpu_index_flat = faiss.index_cpu_to_gpu(self.faiss_res, dev_id, index_flat)

    
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

        for ind, x in tqdm(enumerate(dataloader), total=len(dataloader), desc="Initializing"):
            batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + x[0].shape[0])
            if(self.H.search_type == 'l2'):
                self.dataset_proj[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[1]).cpu()
            else:
                exit()

        self.dataset_proj = self.dataset_proj.cpu().numpy().astype(np.float32)

    def sample(self, latents, gen, snoise=None):
        with torch.no_grad():
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

    def get_lpips_loss(self, inp, tar, use_mean=True):
        res = 0
        if(inp.shape[2] < 32):
            inp_interpolated = F.interpolate(inp, size=(32,32), mode='bicubic')
            tar_interpolated = F.interpolate(tar, size=(32,32), mode='bicubic')
        else:
            inp_interpolated = inp
            tar_interpolated = tar
        inp_feat, inp_shape = self.lpips_net(inp_interpolated)
        tar_feat, _ = self.lpips_net(tar_interpolated)
        for i, g_feat in enumerate(inp_feat):
            lpips_feature_loss = (g_feat - tar_feat[i]) ** 2

            # if(self.H.use_eps_ignore and self.H.use_eps_ignore_advanced):
            #     lpips_feature_loss[bool_mask] = 0.0

            res += torch.sum(lpips_feature_loss, dim=1) / (inp_shape[i] ** 2)
        
        if use_mean:
            return res.mean()
        else:
            return res
    
    def get_dino_loss(self, inp, tar, use_mean=True):
        dino_feat = self.get_dino_features(inp, scale_factor=1, permute=False)
        tar_feat = self.get_dino_features(tar, scale_factor=1, permute=False)
        dino_loss = self.l2_loss(dino_feat, tar_feat)
        if use_mean:
            return dino_loss.mean()
        else:
            return dino_loss
    
    def pseudo_huber(self, diff):
        return 2.0 * self.H.huber_delta**2 * (torch.sqrt(1 + (diff / (self.H.huber_delta)**2)) - 1)

    def calc_loss(self, inp, tar, use_mean=True, logging=False):
        if self.H.loss_type == 'huber':
            per_elem = self.pseudo_huber((inp - tar) ** 2)
        else:
            per_elem = self.l2_loss(inp, tar)

        if use_mean:
            return per_elem.mean()
        return per_elem.reshape(per_elem.shape[0], -1).mean(dim=1)
    
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
                    else:
                        exit()
                    local_pool_proj[batch_slice] = proj

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
    

    def nn_search_batched(self, queries, dataset):
        """
        Hard-first greedy Top-K matching (unique when possible) using self.gpu_index_flat.

        Logic:
        1) Build FAISS index once on the FULL dataset.
        2) Compute hardness score per query using k=2 margin (d2 - d1).
        3) Process queries in hard-first order (small margin first).
        4) For each query, pick the nearest *unused* dataset element from its Top-K list.
        5) If all Top-K are used, fall back to 1-NN (collision allowed) so it always returns.

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

        topk = int(min(max(1, topk), Nd))

        # ---- Build index once on the full dataset ----
        self.gpu_index_flat.reset()
        self.gpu_index_flat.add(dataset)

        # ---- 1) Hardness (margin = d2 - d1) ----
        # Need k=2 even if topk==1, to get a margin; if Nd==1 margin is 0.
        if Nd >= 2:
            D2, _ = self.gpu_index_flat.search(queries, 2)  # (Nq,2)
            margin = D2[:, 0]
        else:
            margin = np.zeros(Nq, dtype=np.float32)

        if tie_shuffle:
            perm = np.random.permutation(Nq)
            order = perm[np.argsort(margin[perm], kind="stable")]
        else:
            order = np.argsort(margin, kind="stable")

        # ---- 2) Get Top-K candidate lists for all queries ----
        D, I = self.gpu_index_flat.search(queries, topk)  # (Nq,K), squared L2 + indices

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

            out_idx[qi] = chosen
            out_dst[qi] = chosen_d

        # ---- Cleanup ----
        self.gpu_index_flat.reset()

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

                # get count of unique indices for logging
                self.unique_indices = np.unique(local_indices).size / self.sz

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

