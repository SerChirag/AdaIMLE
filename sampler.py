from math import ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from LPNet import LPNet
from helpers.utils import is_main_process, get_world_size, get_rank, safe_barrier
from helpers.utils import is_dist_avail_and_initialized
from models import parse_layer_string
from torch import autocast
import faiss
import faiss.contrib.torch_utils
from tqdm import tqdm
from helpers.cache_utils import latent_cache_key, load_latent_cache, save_latent_cache

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
        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
        self.latent_spatial_size = int(getattr(H, 'latent_spatial_size', max(self.block_res)))

        self.selected_dists = torch.empty([sz], dtype=torch.float32)
        self.selected_dists[:] = np.inf
        self.selected_dists_tmp = torch.empty([sz], dtype=torch.float32)

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)

        self._local_pool_latents = None
        self._local_pool_combined = None
        self._gathered_combined_main = None
        self._full_combined_main = None

        self.pool_latents = None

        self.nn_search_batch = H.nn_search_batch

        self.projections = []
        self.l2_projection = None
        self.total_excluded = 0
        self.total_excluded_percentage = 0.0

        if H.search_type in ('lpips', 'combined'):
            self.lpips_net = LPNet(pnet_type=H.lpips_net, path=H.lpips_path).to(self.device)
            self.lpips_net.eval()
            self.lpips_net.requires_grad_(False)

        fake = torch.zeros(1, H.image_channels, self.latent_spatial_size, self.latent_spatial_size, device=self.device)

        safe_barrier()

        if H.search_type == 'lpips':
            interpolated = F.interpolate(fake, scale_factor=H.l2_search_downsample, antialias=True, mode='bicubic')
            out, _ = self.lpips_net(interpolated)
            dims = [int(H.proj_dim * 1. / len(out)) for _ in range(len(out))]
            if H.proj_proportion:
                sm = sum([f.shape[1] for f in out])
                dims = [int(out[i].shape[1] * (H.proj_dim / sm)) for i in range(1, len(out))]
                dims.insert(0, H.proj_dim - sum(dims))
            for ind, feat in enumerate(out):
                self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind], device=self.device), p=2, dim=1))
            sum_dims = sum(dims)

        elif H.search_type == 'l2':
            interpolated = fake.reshape(fake.shape[0], -1)
            sum_dims = interpolated.shape[1]

        else:
            raise ValueError(f'Unsupported search_type: {H.search_type}')
        
        # print search type on rank 0
        if is_main_process():
            print(f"Using search type: {H.search_type} with projection dimension: {sum_dims}")
            
        self.dci_dim = sum_dims

        self.dataset_proj_torch = torch.empty([sz, sum_dims], dtype=torch.float32, device='cpu')
        self.dataset_proj = None
        self.pool_samples_proj = None
        self._dataset_proj_gpu = None

        self.compress_comm = bool(getattr(H, 'compress_comm', True))
        if self.compress_comm:
            self._comm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self._comm_dtype = torch.float32

        self.dataset_size = sz
        self.db_iter = 0
        self.generator_seed = torch.Generator(device=self.device)         
        self.generator_seed.manual_seed(H.seed + self.rank)

        self.faiss_res = faiss.StandardGpuResources()
        self.faiss_res.setTempMemory(64 * 1024 * 1024)  # 64 MB — avoid competing with PyTorch allocator
        index_flat = faiss.IndexFlatL2(self.dci_dim)
        dev_id = torch.cuda.current_device()
        self.faiss_index_flat = faiss.index_cpu_to_gpu(self.faiss_res, dev_id, index_flat)


    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def get_l2_feature(self, inp, permute=True):
        if permute:
            inp = inp.permute(0, 3, 1, 2)
        interpolated = inp.reshape(inp.shape[0], -1)
        return interpolated

    def get_projected(self, inp, permute=True):
        if permute:
            inp = inp.permute(0, 3, 1, 2)
        interpolated = F.interpolate(inp.float(), scale_factor=self.H.l2_search_downsample, antialias=True, mode='bicubic')
        out, _ = self.lpips_net(interpolated.to(self.device))
        gen_feat = [torch.mm(out[i], self.projections[i]) for i in range(len(out))]
        return torch.cat(gen_feat, dim=1)


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
                image_channels=self.H.image_channels,
                cache_dataset_id=getattr(self.H, 'cache_dataset_id', ''),
                search_type=self.H.search_type,
                proj_dim=getattr(self.H, 'proj_dim', 0),
                lpips_net=getattr(self.H, 'lpips_net', ''),
                proj_proportion=getattr(self.H, 'proj_proportion', 0),
                l2_search_downsample=getattr(self.H, 'l2_search_downsample', 1.0),
            )
            cached = load_latent_cache(cache_dir, key, expected_size=self.sz)

        if cached is not None:
            if is_main_process():
                print(f"[cache] Loaded projections from cache ({cached.shape[0]} samples).")
            self.dataset_proj_torch.copy_(cached)
        else:
            ae_batch = getattr(self.H, 'ae_batch', self.H.imle_batch)
            dataloader = DataLoader(dataset, batch_size=ae_batch)

            if is_main_process():
                print("Starting Initialization")

            with torch.inference_mode():
                for ind, x in tqdm(enumerate(dataloader), total=len(dataloader), desc="Initializing"):
                    batch_slice = slice(ind * ae_batch, ind * ae_batch + x[0].shape[0])
                    features = self.preprocess_fn(x)[-1]
                    if self.H.search_type == 'lpips':
                        self.dataset_proj_torch[batch_slice] = self.get_projected(features).cpu()
                    elif self.H.search_type == 'l2':
                        self.dataset_proj_torch[batch_slice] = self.get_l2_feature(features).cpu()
                    else:
                        raise ValueError(f'Unsupported search_type: {self.H.search_type}')

            if use_cache and is_main_process():
                save_latent_cache(cache_dir, key, self.dataset_proj_torch)

        # Pin host memory for faster async H2D transfers.
        if torch.cuda.is_available() and not self.dataset_proj_torch.is_pinned():
            try:
                self.dataset_proj_torch = self.dataset_proj_torch.pin_memory()
            except RuntimeError as e:
                if is_main_process():
                    print(f"Warning: could not pin dataset_proj_torch ({e})")

        self.dataset_proj = self.dataset_proj_torch.numpy()

        # GPU cache of query features.
        self._dataset_proj_gpu = self.dataset_proj_torch.to(self.device, non_blocking=True)

    def sample(self, latents, gen, snoise=None):
        with torch.inference_mode():
            with autocast(device_type='cuda', dtype=getattr(self.H, 'amp_dtype_torch', torch.float16)):
                latents = latents.to(self.device)
                px_z = gen(latents)

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

    def robust_fn(self, sq_diff):
        if self.H.loss_type == 'huber':
            return self.pseudo_huber(sq_diff)
        elif self.H.loss_type == 'pseudo_l1':
            return torch.sqrt(sq_diff + 1e-8) * self.H.huber_delta
        elif self.H.loss_type == 'mclure':
            return sq_diff / (self.H.loss_scale**2 + sq_diff)
        elif self.H.loss_type == 'welsch':
            return 1 - torch.exp(-sq_diff / self.H.loss_scale**2)
        elif self.H.loss_type == 'cauchy':
            return torch.log(1 + 0.5 * sq_diff / self.H.loss_scale**2)
        else:
            return sq_diff

    def get_lpips_loss(self, inp, tar, use_mean=True):
        if inp.shape[2] < 32:
            inp = F.interpolate(inp, size=(32, 32), mode='bicubic')
            tar = F.interpolate(tar, size=(32, 32), mode='bicubic')
        inp_feat, inp_shape = self.lpips_net(inp.float())
        tar_feat, _ = self.lpips_net(tar.float())
        res = 0
        for i, g_feat in enumerate(inp_feat):
            sq_diff = (g_feat - tar_feat[i]) ** 2
            res = res + torch.sum(self.robust_fn(sq_diff), dim=1) / (inp_shape[i] ** 2)
        return res.mean() if use_mean else res

    def calc_loss(self, inp, tar, use_mean=True, logging=False):
        sq_diff = (inp - tar) ** 2
        l2_loss = self.robust_fn(sq_diff).mean(dim=[1, 2, 3])
        per_sample = self.H.l2_coef * l2_loss
        if self.H.lpips_coef > 0:
            per_sample = per_sample + self.H.lpips_coef * self.get_lpips_loss(inp, tar, use_mean=False)
        return per_sample.mean() if use_mean else per_sample
    
    def resample_pool(self, gen):
        local_pool_size = self.pool_size

        # Reuse buffers to avoid repeated GPU allocations.
        if self._local_pool_latents is None or self._local_pool_latents.shape[0] != local_pool_size:
            self._local_pool_latents = torch.empty((local_pool_size, self.H.latent_dim), device=self.device)
            self._local_pool_combined = torch.empty(
                (local_pool_size, self.H.latent_dim + self.dci_dim),
                device=self.device, dtype=self._comm_dtype,
            )
            if self.rank == 0:
                self._gathered_combined_main = [
                    torch.empty_like(self._local_pool_combined) for _ in range(self.world_size)
                ]

        self._local_pool_latents.normal_(generator=self.generator_seed)

        with torch.inference_mode():
            for start in range(0, local_pool_size, self.H.imle_batch):
                end = min(start + self.H.imle_batch, local_pool_size)
                batch_slice = slice(start, end)
                cur_latents = self._local_pool_latents[batch_slice]
                self._local_pool_combined[batch_slice, :self.H.latent_dim].copy_(cur_latents.to(self._comm_dtype))
                with autocast(device_type='cuda', dtype=getattr(self.H, 'amp_dtype_torch', torch.float16)):
                    outputs = gen(cur_latents)
                    if self.H.search_type == 'lpips':
                        proj = self.get_projected(outputs, False)
                    elif self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    else:
                        raise ValueError(f'Unsupported search_type: {self.H.search_type}')
                    self._local_pool_combined[batch_slice, self.H.latent_dim:].copy_(proj.to(self._comm_dtype))

        # Gather to rank 0.
        if self.rank == 0:
            if self._gathered_combined_main is None or len(self._gathered_combined_main) != self.world_size:
                self._gathered_combined_main = [torch.empty_like(self._local_pool_combined) for _ in range(self.world_size)]
            torch.distributed.gather(self._local_pool_combined, gather_list=self._gathered_combined_main, dst=0)
        else:
            torch.distributed.gather(self._local_pool_combined, gather_list=None, dst=0)

        if self.rank == 0:
            full_pool_size = local_pool_size * self.world_size
            combined_dim = self.H.latent_dim + self.dci_dim
            if self._full_combined_main is None or self._full_combined_main.shape[0] != full_pool_size:
                self._full_combined_main = torch.empty((full_pool_size, combined_dim), dtype=torch.float32, device=self.device)
            torch.cat([c.to(torch.float32) for c in self._gathered_combined_main], dim=0, out=self._full_combined_main)
            self.pool_latents = self._full_combined_main[:, :self.H.latent_dim]
            self.pool_samples_proj = self._full_combined_main[:, self.H.latent_dim:]
    

    def nn_search_batched(self, queries, dataset):
        if isinstance(queries, np.ndarray):
            queries_t = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to(self.device)
        else:
            queries_t = queries.to(self.device)

        if isinstance(dataset, np.ndarray):
            dataset_t = torch.from_numpy(np.ascontiguousarray(dataset, dtype=np.float32)).to(self.device)
        else:
            dataset_t = dataset.to(self.device)

        self.faiss_index_flat.reset()
        self.faiss_index_flat.add(dataset_t.contiguous())
        D, I = self.faiss_index_flat.search(queries_t.contiguous(), 1)
        self.faiss_index_flat.reset()

        return D.squeeze(1).to(torch.float32), I.squeeze(1).to(torch.long)


    def imle_sample_force(self, gen, to_update=None):
        self._imle_sample_force_unconditional(gen)

    def _imle_sample_force_unconditional(self, gen):
        """Force resampling for unconditional training. Rank 0 does NN search, broadcasts."""
        if is_main_process():
            t1 = time.time()
            print("Starting pool resampling...")

        gen.eval()
        self.resample_pool(gen)
        gen.train()

        if is_main_process():
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")

        torch.cuda.empty_cache()
        self.selected_dists_tmp[:] = np.inf

        with torch.inference_mode():
            if is_main_process():
                if self._dataset_proj_gpu is None:
                    self._dataset_proj_gpu = self.dataset_proj_torch.to(self.device, non_blocking=True)

                local_ds_feats = self._dataset_proj_gpu
                pool_feats = self.pool_samples_proj

                _, local_indices = self.nn_search_batched(local_ds_feats, pool_feats)
                self.unique_indices = torch.unique(local_indices).numel() / self.sz

                local_indices = local_indices.to(device=self.pool_latents.device, non_blocking=True)
                new_latents = self.pool_latents.index_select(0, local_indices)

                full_updated_latents = new_latents
                perturbation = self.H.imle_perturb_coef * torch.randn(
                    (self.sz, self.H.latent_dim),
                    device=self.device, generator=self.generator_seed,
                )
                full_updated_latents = full_updated_latents + perturbation
                comm_latents = full_updated_latents.to(self._comm_dtype)
            else:
                comm_latents = torch.empty(self.sz, self.H.latent_dim, dtype=self._comm_dtype, device=self.device)

            torch.distributed.broadcast(comm_latents, src=0)
            full_updated_latents = comm_latents.to(torch.float32)

            self.last_selected_latents.copy_(self.selected_latents)
            self.selected_latents.copy_(full_updated_latents.cpu())

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        self.faiss_index_flat.reset()

