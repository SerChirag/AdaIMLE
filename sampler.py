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
from diffusers import AutoencoderTiny
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

        ############
        #  Can be removed
        self.selected_dists_lpips = torch.empty([sz], dtype=torch.float32)
        self.selected_dists_lpips[:] = np.inf

        self.selected_dists_l2 = torch.empty([sz], dtype=torch.float32)
        self.selected_dists_l2[:] = np.inf 
        #############

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float32)

        self.pool_latents = None

        self.projections = []
        self.lpips_net = LPNet(pnet_type=H.lpips_net, path=H.lpips_path).to(self.device)
        self.lpips_net.eval()
        self.lpips_net.requires_grad_(False)

        ## TODO: check this is required or not
        self.lpips_net = torch.compile(self.lpips_net)

        self.dino_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

        model = AutoModel.from_pretrained("./models--facebook--dinov2-base/snapshots/main").eval().to(self.device)
        self.dino_encoder = torch.compile(model)


        # self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(self.device)
        # # self.vae = AutoencoderTiny.from_pretrained("./tiny-auto/models--madebyollin--taesd/snapshots/main").to(self.device)
        # self.vae.eval()
        # self.vae.requires_grad_(False)

        self.l2_projection = None

        fake = torch.zeros(1, 3, H.image_size, H.image_size, device=self.device)

        torch.distributed.barrier()

        if(H.search_type == 'lpips'):
            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample, antialias=True, mode='bicubic')
            out, shapes = self.lpips_net(interpolated)
            sum_dims = 0
            dims = [int(H.proj_dim * 1. / len(out)) for _ in range(len(out))]
            if H.proj_proportion:
                sm = sum([dim.shape[1] for dim in out])
                dims = [int(out[feat_ind].shape[1] * (H.proj_dim / sm)) for feat_ind in range(1,len(out))]
                dims.insert(0,H.proj_dim - sum(dims))
            for ind, feat in enumerate(out):
                self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind], device=self.device), p=2, dim=1))
            sum_dims = sum(dims)

        elif(H.search_type == 'l2'):
            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample, antialias=True, mode='bicubic')
            interpolated = interpolated.reshape(interpolated.shape[0],-1)
            self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim, device=self.device), p=2, dim=1)
            sum_dims = H.proj_dim

        # elif(H.search_type == 'vae'):
        #     interpolated = self.vae.encode(fake).latents
        #     interpolated = interpolated.reshape(interpolated.shape[0],-1)
        #     self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim, device=self.device), p=2, dim=1)
        #     sum_dims = H.proj_dim
        
        elif(H.search_type == 'combined'):
            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample, antialias=True, mode='bicubic')
            out, shapes = self.lpips_net(interpolated)
            sum_dims = 0
            dims = [int(H.proj_dim * 1. / len(out)) for _ in range(len(out))]
            if H.proj_proportion:
                sm = sum([dim.shape[1] for dim in out])
                dims = [int(out[feat_ind].shape[1] * (H.proj_dim / sm)) for feat_ind in range(1,len(out))]
                dims.insert(0,H.proj_dim - sum(dims))
            for ind, feat in enumerate(out):
                self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind], device=self.device), p=2, dim=1))
            sum_dims = sum(dims)

            interpolated = self.preprocess_dino_tensor(fake)
            with torch.no_grad():
                out = self.dino_encoder(pixel_values=interpolated)
                out = out.last_hidden_state.mean(dim=1)            
            sum_dims += out.shape[-1]

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
        self.gpu_index_flat = faiss.index_cpu_to_gpu(self.faiss_res, self.rank, index_flat)

    def preprocess_dino_tensor(self, inp):
        # x: [B, C, H, W], range [0, 1]

        x = (inp + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)

        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        return (x - self.dino_mean) / self.dino_std

    # def get_vae_features(self, inp, permute=True):
    #     if(permute):
    #         inp = inp.permute(0, 3, 1, 2)
    #     interpolated = self.vae.encode(inp).latents
    #     interpolated = interpolated.reshape(interpolated.shape[0],-1)
    #     return interpolated

    def get_projected(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        
        interpolated = F.interpolate(inp,scale_factor = self.H.l2_search_downsample, antialias=True, mode='bicubic')
        out, _ = self.lpips_net(interpolated.to(self.device))
        gen_feat = []
        for i in range(len(out)):
            gen_feat.append(torch.mm(out[i], self.projections[i]))
            # TODO divide?
        lpips_feat = torch.cat(gen_feat, dim=1)
        # lpips_feat = F.normalize(lpips_feat, p=2, dim=1)
        return lpips_feat
    
    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        interpolated = F.interpolate(inp,scale_factor = self.H.l2_search_downsample, antialias=True, mode='bicubic')
        interpolated = interpolated.reshape(interpolated.shape[0],-1)
        interpolated = torch.mm(interpolated, self.l2_projection)
        # interpolated = F.normalize(interpolated, p=2, dim=1)
        return interpolated
    
    def get_dino_features(self, inp, permute=True, scale_factor=10):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        interpolated = self.preprocess_dino_tensor(inp)
        with torch.no_grad():
            out = self.dino_encoder(pixel_values=interpolated)
            out = out.last_hidden_state.mean(dim=1)   
            out = F.normalize(out, p=2, dim=1)
            out = out * scale_factor
        return out
    
    def get_combined_feature(self, inp, permute=True):
        lpisps_feat = self.get_projected(inp, permute)
        dino_feat = self.get_dino_features(inp, permute)
        # print(f'LPIPS is {torch.norm(lpisps_feat, p=2, dim=1).mean()} \n')
        # print(f'DINO is {torch.norm(dino_feat, p=2, dim=1).mean()} \n')
        combined_feat = torch.cat((lpisps_feat, dino_feat), dim=1)
        return combined_feat

    def init_projection(self, dataset):

        for ind, x in enumerate(DataLoader(TensorDataset(dataset), batch_size=self.H.n_batch)):
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + x[0].shape[0])
            if(self.H.search_type == 'lpips'):
                self.dataset_proj[batch_slice] = self.get_projected(self.preprocess_fn(x)[1]).cpu()
            elif(self.H.search_type == 'l2'):
                self.dataset_proj[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[1]).cpu()
            # elif(self.H.search_type == 'vae'):
            #     self.dataset_proj[batch_slice] = self.get_vae_features(self.preprocess_fn(x)[1]).cpu()
            elif(self.H.search_type == 'combined'):
                self.dataset_proj[batch_slice] = self.get_combined_feature(self.preprocess_fn(x)[1]).cpu()
            else:
                exit()

        self.dataset_proj = self.dataset_proj.cpu().numpy().astype(np.float32)

    def sample(self, latents, gen, snoise=None):
        with torch.no_grad():
            with autocast(device_type='cuda'):
                latents = latents.to(self.device)
                px_z = gen(latents, None).permute(0, 2, 3, 1)
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

    def calc_loss(self, inp, tar, use_mean=True, logging=False):

        if use_mean:       
            l2_loss = torch.mean(self.l2_loss(inp, tar))
            res = 0
            
            lpips_loss = self.get_lpips_loss(inp, tar)

            if(inp.shape[2] < 32):
                dino_loss = self.get_dino_loss(inp, tar)
            else:
                dino_loss = torch.tensor(0.0, device=self.device)

            loss = self.H.lpips_coef * lpips_loss + self.H.l2_coef * l2_loss + self.H.dino_coef * dino_loss
            
            if logging:
                return loss, res.mean(), l2_loss.mean()
            else:
                return loss

        else:
            inp_feat, inp_shape = self.lpips_net(inp)
            tar_feat, _ = self.lpips_net(tar)
            res = 0
            for i, g_feat in enumerate(inp_feat):
                res += torch.sum((g_feat - tar_feat[i]) ** 2, dim=1) / (inp_shape[i] ** 2)
            l2_loss = torch.mean(self.l2_loss(inp, tar), dim=[1, 2, 3])
            loss = self.H.lpips_coef * res + self.H.l2_coef * l2_loss
            if logging:
                return loss, res.mean(), l2_loss
            else:
                return loss
            
    ############### Can be removed ###########
    
    def calc_dists_existing(self, dataset_tensor, gen, dists=None, dists_lpips = None, dists_l2 = None, latents=None, to_update=None, snoise=None, logging=False):
        if dists is None:
            dists = self.selected_dists
        if dists_lpips is None:
            dists_lpips = self.selected_dists_lpips
        if dists_l2 is None:
            dists_l2 = self.selected_dists_l2
        if latents is None:
            latents = self.selected_latents

        if to_update is not None:
            latents = latents[to_update]
            dists = dists[to_update]
            dataset_tensor = dataset_tensor[to_update]

        for ind, x in enumerate(DataLoader(TensorDataset(dataset_tensor), batch_size=self.H.n_batch)):
            _, target = self.preprocess_fn(x)
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + target.shape[0])
            cur_latents = latents[batch_slice]
            with torch.no_grad():
                with autocast(device_type='cuda'):
                    out = gen(cur_latents, None)
                    if(logging):
                        dist, dist_lpips, dist_l2 = self.calc_loss(target.permute(0, 3, 1, 2), out, use_mean=False, logging=True)
                        dists[batch_slice] = torch.squeeze(dist)
                        dists_lpips[batch_slice] = torch.squeeze(dist_lpips)
                        dists_l2[batch_slice] = torch.squeeze(dist_l2)
                    else:
                        dist = self.calc_loss(target.permute(0, 3, 1, 2), out, use_mean=False)
                        dists[batch_slice] = torch.squeeze(dist)
        
        if(logging):
            return dists, dists_lpips, dists_l2
        else:
            return dists
    
    ############### Can be removed ###########


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
                    if self.H.search_type == 'lpips':
                        proj = self.get_projected(outputs, False)
                    elif self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    # elif self.H.search_type == 'vae':
                    #     proj = self.get_vae_features(outputs, False)
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

    def imle_sample_force(self, dataset, gen, to_update=None):
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

            # Perform NN search for the local chunk. Returns arrays of shape (local_size, 1).
            distances, indices = self.gpu_index_flat.search(local_ds_feats, 1)
            local_distances = torch.from_numpy(distances).squeeze(1)  # (local_size,)
            local_indices   = torch.from_numpy(indices).squeeze(1)    # (local_size,)

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

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        torch.distributed.barrier()  # Ensure synchronization before leaving the function
        self.gpu_index_flat.reset()
