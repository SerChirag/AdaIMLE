from math import ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

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

        self.pool_size = ceil(int(H.pool_size_per_class) / H.imle_db_size) * H.imle_db_size
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduction='none').to(self.device)
        self.l1_loss = torch.nn.L1Loss(reduction='none').to(self.device)
        self.H = H
        self.latent_lr = H.latent_lr
        self.sz = sz
        self.unique_indices = 0

        if(is_main_process()):
             print(f"Initialized Sampler with dataset size {sz}")
             
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float16)
        self.last_selected_latents = torch.empty([H.num_images_visualize, H.latent_dim], dtype=torch.float16)

        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
        self.latent_spatial_size = int(getattr(H, 'latent_spatial_size', max(self.block_res)))

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float16)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.latent_spatial_size, self.latent_spatial_size],
                                        dtype=torch.float16)

        self.pool_latents = torch.empty([self.pool_size, H.latent_dim], dtype=torch.float16, device=self.device)
        self.decode_for_metrics = bool(getattr(H, 'autoencoder_decode_for_metrics', True))
        self.autoencoder = load_autoencoder(H, self.device)
        if(self.H.compile):
            self.autoencoder = torch.compile(self.autoencoder)

        fake_rgb = torch.zeros(1, 3, H.image_size, H.image_size, device=self.device)
        native_latents = encode_images_to_latents(self.autoencoder, fake_rgb, target_spatial=None)
        self.autoencoder_native_latent_size = (native_latents.shape[-2], native_latents.shape[-1])

        if H.search_type != 'l2':
            raise ValueError('This branch expects search_type=l2.')
        
        self.nn_search_batch = H.nn_search_batch


        fake = torch.zeros(1, H.image_channels, self.latent_spatial_size, self.latent_spatial_size, device=self.device)

        safe_barrier()
        if(H.search_type == 'l2'):
            interpolated = fake.reshape(fake.shape[0], -1)
            sum_dims = interpolated.shape[1]

        else:
            exit()

        self.dci_dim = sum_dims

        self.dataset_proj = None
        self.pool_samples_proj = np.empty([self.pool_size, self.dci_dim], dtype=np.float16)

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

        self.num_classes = H.num_classes
        self.local_classes = self._distribute_classes_across_gpus()
        self.class_ranges = None
        self.viz_indices = None


    def _distribute_classes_across_gpus(self):
        """Distribute classes across GPUs for load balancing."""
        classes_per_gpu = self.num_classes // self.world_size
        remainder = self.num_classes % self.world_size
        
        if self.rank < remainder:
            local_classes_count = classes_per_gpu + 1
            start_class = self.rank * local_classes_count
        else:
            local_classes_count = classes_per_gpu
            start_class = self.rank * local_classes_count + remainder
        
        local_classes = list(range(start_class, start_class + local_classes_count))
        
        if is_main_process():
            print(f"GPU {self.rank}: handling classes {local_classes}")
        
        return local_classes


    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        if inp.shape[1] == 3:
            inp = encode_images_to_latents(
                self.autoencoder,
                inp,
                target_spatial=(self.latent_spatial_size, self.latent_spatial_size),
            )
        return inp.reshape(inp.shape[0], -1)
    
    def init_projection(self, dataset):
        # Filter indices for local classes only
        indices = [i for i, (_, y) in enumerate(dataset) if y in self.local_classes]
        subset = Subset(dataset, indices)

            # Step 2: build class_ranges directly from labels
        self.class_ranges = {}
        start = 0
        prev_cls = None

        for j, idx in enumerate(indices):
            _, y = dataset[idx]
            lbl = int(y if not isinstance(y, (list, tuple)) else y[0])  # flatten
            if lbl != prev_cls:
                if prev_cls is not None:
                    self.class_ranges[prev_cls] = (start, j)  # close previous class
                start = j
                prev_cls = lbl
        # close the last class
        if prev_cls is not None:
            self.class_ranges[prev_cls] = (start, len(indices))
        
        self.dataset_proj = torch.empty([len(indices), self.dci_dim], dtype=torch.float16, device='cpu')

        dataloader = DataLoader(
            subset,
            batch_size=self.H.imle_batch,
        )

        if is_main_process():
            print(f"Starting Initialization for classes {self.local_classes}")

        for ind, x in tqdm(enumerate(dataloader), total=len(dataloader), desc="Initializing"):
            batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + x[0].shape[0])
            self.dataset_proj[batch_slice] = self.get_l2_feature(self.preprocess_fn(x[0])[1]).cpu()

        # Convert to numpy
        self.dataset_proj = self.dataset_proj.numpy().astype(np.float16)
        # print(f"Rank {self.rank} class ranges: {self.class_ranges}")


    def sample(self, latents, labels, gen, snoise=None):
        with torch.no_grad():
            with autocast(device_type='cuda'):
                latents = latents.to(self.device)
                px_z = gen(latents, labels)
                if self.decode_for_metrics:
                    px_z = decode_latents_to_images(self.autoencoder, px_z, self.autoencoder_native_latent_size)
                px_z = px_z.permute(0, 2, 3, 1)
                xhat = (px_z + 1.0) * 127.5
                xhat = xhat.detach().cpu().numpy()
                xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
                return xhat

    def pseudo_huber(self, diff):
        delta = self.H.huber_delta
        return 2.0 * delta**2 * (torch.sqrt(1 + (diff / (delta)**2)) - 1)

    def calc_loss(self, inp, tar):
        if self.H.loss_type == 'huber':
            per_elem = self.pseudo_huber((inp - tar) ** 2)
        elif self.H.loss_type == 'pseudo_l1':
            per_elem = self.l1_loss(inp, tar) * self.H.huber_delta
        elif self.H.loss_type == 'mclure':
            residual = inp - tar
            per_elem = (residual ** 2) / (self.H.loss_scale**2 + residual ** 2)
        elif self.H.loss_type == 'welsch':
            residual = inp - tar
            per_elem = 1 - torch.exp(-(residual / self.H.loss_scale) ** 2)
        elif self.H.loss_type == 'rmse':
            residual = inp - tar
            return torch.sqrt((residual ** 2).mean() + 1e-8)
        elif self.H.loss_type == 'cauchy':
            residual = inp - tar
            per_elem = torch.log1p((residual / self.H.loss_scale) ** 2)
        else:
            per_elem = self.l2_loss(inp, tar)
        return per_elem.mean()
        
            
    def resample_pool(self, gen, class_condition):

        gen.eval()   

        self.pool_latents.normal_()

        for j in range(self.pool_size // self.H.imle_batch):
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
            cur_latents = self.pool_latents[batch_slice]
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    class_tensor = torch.full((cur_latents.shape[0],), class_condition, 
                                              dtype=torch.long, device=cur_latents.device)
                    outputs = gen(cur_latents, class_tensor)
                    proj = self.get_l2_feature(outputs, False)
                    self.pool_samples_proj[batch_slice] = proj.to('cpu').detach().cpu().numpy().astype(np.float16)

        gen.train()
    

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
            self.last_selected_latents = self.selected_latents[self.viz_indices].clone()
            t1 = time.time()
            print("Starting pool resampling...")

        all_pool_latents = []

        for i in self.local_classes:
            self.resample_pool(gen, i)
        
            torch.cuda.empty_cache()

            with torch.no_grad():

                pool_feats = np.ascontiguousarray(self.pool_samples_proj, dtype=np.float32)


                # Obtain the full dataset features (on CPU) and then slice locally.
                local_ds_feats = np.ascontiguousarray(self.dataset_proj[self.class_ranges[i][0]:self.class_ranges[i][1]], dtype=np.float32) 

                local_distances, local_indices = self.nn_search_batched(local_ds_feats, pool_feats)

                new_latents = self.pool_latents[local_indices].clone()
                all_pool_latents.append(new_latents.detach().cpu())
            
        all_pool_latents = torch.cat(all_pool_latents, dim=0)

        safe_barrier()  # Ensure all processes complete the gather
                
        if is_main_process():
            print("Done resampling. Now collecting")
            gathered_latents = [None for _ in range(self.world_size)]
        else:
            gathered_latents = None
        
        safe_barrier()  # Ensure all processes complete the gather

        torch.distributed.gather_object(all_pool_latents, gathered_latents, dst=0)

        safe_barrier()  # Ensure all processes complete the gather

        if is_main_process():
            gathered_latents = [t.cpu() for t in gathered_latents]
            full_updated_latents = torch.cat(gathered_latents, dim=0).to(self.device)
            perturbation = self.H.imle_perturb_coef * torch.randn(
                (self.sz, self.H.latent_dim), 
                device=self.device,
                generator=self.generator_seed)
            full_updated_latents += perturbation
        else:
            full_updated_latents = torch.empty(self.sz, self.H.latent_dim, dtype=torch.float16, device=self.device)

        safe_barrier()

        torch.distributed.broadcast(full_updated_latents, src=0)

        safe_barrier()

        # Move the broadcasted results to CPU if desired.
        self.selected_latents = full_updated_latents.cpu().clone()

        if is_main_process():
            print(f"Force resampling took {time.time() - t1:.2f} seconds")

        safe_barrier()  # Ensure synchronization before leaving the function
