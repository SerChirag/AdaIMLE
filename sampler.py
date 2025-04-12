from curses import update_lines_cols
from math import comb, ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from LPNet import LPNet
from torch.optim import AdamW
from helpers.utils import ZippedDataset
from models import parse_layer_string
from helpers.angle_sampler import Angle_Generator
from knn_cuda import KNN
from torch.cuda.amp import autocast
from diffusers import AutoencoderTiny
import faiss
from accelerate import Accelerator

class Sampler:
    def __init__(self, H, sz, preprocess_fn):
        self.accelerator = Accelerator(mixed_precision="fp16")

        self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduce=False)
        self.H = H
        self.latent_lr = H.latent_lr
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

        self.selected_dists_lpips = torch.empty([sz], dtype=torch.float32, device=self.accelerator.device)
        self.selected_dists_lpips[:] = np.inf

        self.selected_dists_l2 = torch.empty([sz], dtype=torch.float32, device=self.accelerator.device)
        self.selected_dists_l2[:] = np.inf 

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float32)

        self.pool_latents = torch.randn([self.pool_size, H.latent_dim], dtype=torch.float32)

        self.projections = []
        self.lpips_net = self.accelerator.prepare_model(LPNet(pnet_type=H.lpips_net, path=H.lpips_path))
        

        # self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").cuda()
        # self.vae.eval()
        # self.vae.requires_grad_(False)

        self.l2_projection = None

        self.knn = KNN(k=1, transpose_mode=True)

        fake = torch.zeros(1, 3, H.image_size, H.image_size, device='cuda')

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
                self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind], device='cuda'), p=2, dim=1))
            sum_dims = sum(dims)

        elif(H.search_type == 'l2'):
            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample, antialias=True, mode='bicubic')
            interpolated = interpolated.reshape(interpolated.shape[0],-1)
            self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim), p=2, dim=1).cuda()
            sum_dims = H.proj_dim

        elif(H.search_type == 'vae'):
            interpolated = self.vae.encode(fake).latents
            interpolated = interpolated.reshape(interpolated.shape[0],-1)
            self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim), p=2, dim=1).cuda()
            sum_dims = H.proj_dim

        else:
            projection_dim = H.proj_dim // 2
            dims = [int(projection_dim * 1. / len(out)) for _ in range(len(out))]
            if H.proj_proportion:
                sm = sum([dim.shape[1] for dim in out])
                dims = [int(out[feat_ind].shape[1] * (projection_dim / sm)) for feat_ind in range(len(out) - 1)]
                dims.append(projection_dim - sum(dims))
            for ind, feat in enumerate(out):
                self.projections.append(F.normalize(torch.randn(feat.shape[1], dims[ind]), p=2, dim=1).cuda())

            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample, antialias=True, mode='bicubic')
            interpolated = interpolated.reshape(interpolated.shape[0],-1)
            self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim // 2), p=2, dim=1).cuda()
            sum_dims = H.proj_dim

        self.dci_dim = sum_dims
        print('dci_dim', self.dci_dim)

        self.temp_samples_proj = torch.empty([self.H.imle_db_size, sum_dims], dtype=torch.float32, device='cuda')
        self.dataset_proj = torch.empty([sz, sum_dims], dtype=torch.float32, device='cuda')
        self.pool_samples_proj = torch.empty([self.pool_size, sum_dims], dtype=torch.float32, device='cuda')

        self.knn_ignore = H.knn_ignore
        self.ignore_radius = H.ignore_radius
        self.resample_angle = H.resample_angle

        self.angle_generator = Angle_Generator(self.H.latent_dim)
        self.max_sample_angle_rad = H.max_sample_angle_rad
        self.min_sample_angle_rad = H.min_sample_angle_rad

        self.total_excluded = 0
        self.total_excluded_percentage = 0
        self.dataset_size = sz
        self.db_iter = 0

    def get_vae_features(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        interpolated = self.vae.encode(inp).latents
        interpolated = interpolated.reshape(interpolated.shape[0],-1)
        return interpolated.cuda()

    def get_projected(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        
        interpolated = F.interpolate(inp,scale_factor = self.H.l2_search_downsample, antialias=True, mode='bicubic')
        out, _ = self.lpips_net(interpolated.cuda())
        gen_feat = []
        for i in range(len(out)):
            gen_feat.append(torch.mm(out[i], self.projections[i]))
            # TODO divide?
        lpips_feat = torch.cat(gen_feat, dim=1)
        # lpips_feat = F.normalize(lpips_feat, p=2, dim=1)
        return lpips_feat.cuda()
    
    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        interpolated = F.interpolate(inp,scale_factor = self.H.l2_search_downsample, antialias=True, mode='bicubic')
        interpolated = interpolated.reshape(interpolated.shape[0],-1)
        interpolated = torch.mm(interpolated, self.l2_projection)
        # interpolated = F.normalize(interpolated, p=2, dim=1)
        return interpolated.cuda()
    
    def get_combined_feature(self, inp, permute=True):
        lpips_feat = self.get_projected(inp, permute)
        l2_feat = self.get_l2_feature(inp, permute)
        return torch.cat([lpips_feat, l2_feat], dim=1)
        # return torch.cat([lpips_feat, l2_feat], dim=1)
        # if(permute):
        #     inp = inp.permute(0, 3, 1, 2)

        # out, _ = self.lpips_net(inp.cuda())
        # gen_feat = []
        # for i in range(len(out)):
        #     gen_feat.append(torch.mm(out[i], self.projections[i]))
        #     # TODO divide?
        # gen_feat = torch.cat(gen_feat, dim=1)
        # interpolated = F.interpolate(inp,scale_factor = self.H.l2_search_downsample)
        # interpolated = interpolated.reshape(interpolated.shape[0],-1)
        # interpolated = torch.mm(interpolated, self.l2_projection)
        # return gen_feat + interpolated.cuda()

    def init_projection(self, dataset):
        for proj_mat in self.projections:
            proj_mat[:] = F.normalize(torch.randn(proj_mat.shape), p=2, dim=1)

        for ind, x in enumerate(DataLoader(TensorDataset(dataset), batch_size=self.H.n_batch)):
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + x[0].shape[0])
            if(self.H.search_type == 'lpips'):
                self.dataset_proj[batch_slice] = self.get_projected(self.preprocess_fn(x)[1])
            elif(self.H.search_type == 'l2'):
                self.dataset_proj[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[1])
            elif(self.H.search_type == 'vae'):
                self.dataset_proj[batch_slice] = self.get_vae_features(self.preprocess_fn(x)[1])
            else:
                self.dataset_proj[batch_slice] = self.get_combined_feature(self.preprocess_fn(x)[1])

    def sample(self, latents, gen, snoise=None):
        with torch.no_grad():
            nm = latents.shape[0]
            latents = latents.to('cuda')
            px_z = gen(latents, None).permute(0, 2, 3, 1)
            xhat = (px_z + 1.0) * 127.5
            xhat = xhat.detach().cpu().numpy()
            xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            return xhat

    def sample_from_out(self, px_z):
        with torch.no_grad():
            px_z = px_z.permute(0, 2, 3, 1)
            xhat = (px_z + 1.0) * 127.5
            xhat = xhat.detach().cpu().numpy()
            xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
            return xhat
    
    def calc_loss_projected(self, inp, tar):
        inp_feat = self.get_projected(inp,False)
        tar_feat = self.get_projected(tar,False)
        res = torch.linalg.norm(inp_feat - tar_feat, dim=1)
        return res
    
    def calc_loss_l2(self, inp, tar):
        inp_feat = self.get_l2_feature(inp,False)
        tar_feat = self.get_l2_feature(tar,False)
        res = torch.linalg.norm(inp_feat - tar_feat, dim=1)
        return res

    def calc_loss(self, inp, tar, use_mean=True, logging=False, only_l2 = False):

        if use_mean:       
            l2_loss = torch.mean(self.l2_loss(inp, tar), dim=[1, 2, 3])
            res = 0

            if only_l2:
                return l2_loss.mean()

            inp_feat, inp_shape = self.lpips_net(inp)
            tar_feat, _ = self.lpips_net(tar)
        
            for i, g_feat in enumerate(inp_feat):
                lpips_feature_loss = (g_feat - tar_feat[i]) ** 2

                # if(self.H.use_eps_ignore and self.H.use_eps_ignore_advanced):
                #     lpips_feature_loss[bool_mask] = 0.0

                res += torch.sum(lpips_feature_loss, dim=1) / (inp_shape[i] ** 2)

            loss = self.H.lpips_coef * res.mean() + self.H.l2_coef * l2_loss.mean()
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
            cur_latents = latents[batch_slice].to('cuda')
            with torch.no_grad():
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
    
    def calc_dists_existing_nn(self, dataset_tensor, gen, dists=None, latents=None, to_update=None, snoise=None):
        if dists is None:
            dists = self.selected_dists
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
                out = gen(cur_latents, None)
                if(self.H.search_type == 'lpips'):
                    dist = self.calc_loss_projected(target.permute(0, 3, 1, 2), out)
                else:
                    dist = self.calc_loss_l2(target.permute(0, 3, 1, 2), out)
                dists[batch_slice] = torch.squeeze(dist)
        return dists
        


    def resample_pool(self, gen):
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        # Step 1: Init latent pool only on rank 0 (optional)
        self.pool_latents.normal_()
        total_pool = self.pool_latents.shape[0]

        # Step 2: Slice the pool for this rank
        local_size = (total_pool + world_size - 1) // world_size
        start = rank * local_size
        end = min((rank + 1) * local_size, total_pool)
        batch_slice = slice(start, end)

        cur_latents = self.pool_latents[batch_slice].to(self.accelerator.device)

        # Step 3: Create local DataLoader
        dataloader = DataLoader(TensorDataset(cur_latents), batch_size=self.H.imle_batch, shuffle=False)
        dataloader = self.accelerator.prepare_data_loader(dataloader)

        local_proj = []

        for batch in dataloader:
            with torch.no_grad():
                latents = batch[0].to(self.accelerator.device)
                if self.H.search_type == 'lpips':
                    feat = self.get_projected(gen(latents, None), False)
                elif self.H.search_type == 'l2':
                    feat = self.get_l2_feature(gen(latents, None), False)
                elif self.H.search_type == 'vae':
                    feat = self.get_vae_features(gen(latents, None), False)
                else:
                    feat = self.get_combined_feature(gen(latents, None), False)

                local_proj.append(feat.cpu())

        local_proj = torch.cat(local_proj, dim=0)

        # Step 4: Gather from all processes
        gathered_proj = [torch.zeros_like(local_proj) for _ in range(world_size)]
        dist.all_gather(gathered_proj, local_proj.contiguous().to(self.accelerator.device))

        # Step 5: Merge into global pool on all processes
        self.pool_samples_proj = torch.cat(gathered_proj, dim=0)[:total_pool].to(self.accelerator.device)

        if dist.is_initialized():
            dist.barrier()


        # for j in range(self.pool_size // self.H.imle_batch):
        #     batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)

        #     if(self.H.use_angular_resample):
        #         cur_latents = self.sample_angle(self.pool_latents[batch_slice])
            
        #     else:
        #         cur_latents = self.pool_latents[batch_slice]
        #         cur_latents = cur_latents.to('cuda')

        #     with torch.no_grad():
        #         with torch.amp.autocast('cuda'):
        #             if(self.H.search_type == 'lpips'):
        #                 self.pool_samples_proj[batch_slice] = self.get_projected(gen(cur_latents, None), False)
        #             elif(self.H.search_type == 'l2'):
        #                 self.pool_samples_proj[batch_slice] = self.get_l2_feature(gen(cur_latents, None), False)
        #             elif(self.H.search_type == 'vae'):
        #                 self.pool_samples_proj[batch_slice] = self.get_vae_features(gen(cur_latents, None), False)
        #             else:
        #                 self.pool_samples_proj[batch_slice] = self.get_combined_feature(gen(cur_latents, None), False)

    def imle_sample_force(self, dataset, gen, to_update=None):
        """
        Optimized force resampling routine using FAISS for batched nearest neighbor search.
        This implementation replaces the nested loop over DataLoader batches and pool partitions with a one-shot FAISS query.
        
        It assumes:
        - self.resample_pool() has updated the pool samples projections (self.pool_samples_proj) and pool latents (self.pool_latents).
        - self.dataset_proj contains precomputed projections of your dataset.
        - All features are in float32.
        """


        t1 = time.time()
        self.resample_pool(gen)
        print(f"Resampling pool took {time.time() - t1:.2f} seconds")

        # Reset temporary distances (we assume selected_dists_tmp is a torch tensor)
        self.selected_dists_tmp[:] = np.inf
        total_rejected = 0
        self.total_excluded = total_rejected
        self.total_excluded_percentage = (total_rejected * 1.0 / self.pool_size) * 100

        with torch.no_grad():
            # Prepare the dataset features corresponding to the indices to update.
            # Make sure dataset_proj is float32.
            ds_feats = self.dataset_proj.cpu().numpy().astype(np.float32)
            
            # Prepare the pool samples features.
            pool_feats = self.pool_samples_proj.cpu().numpy().astype(np.float32)
            feature_dim = pool_feats.shape[1]
            
            # Create a FAISS index for L2 distance search.
            index = faiss.IndexFlatL2(feature_dim)
            # If your GPU can handle it and you want to accelerate further, you can transfer the index to GPU:
            # res = faiss.StandardGpuResources()
            # index = faiss.index_cpu_to_gpu(res, 0, index)
            
            index.add(pool_feats)  # add the entire pool of features at once
            
            # Perform batched nearest neighbor search for all dataset features.
            # The returned arrays have shape (num_samples, 1).
            distances, indices = index.search(ds_feats, 1)
            
            # Convert the results to torch tensors.
            distances_tensor = torch.from_numpy(distances).squeeze(1)  
            indices_tensor   = torch.from_numpy(indices).squeeze(1)    
            
            # Get the current stored distances for these indices.
            current_dists = self.selected_dists_tmp.cpu()
            
            # Determine which samples should be updated.
            need_update = distances_tensor < current_dists
            # Identify absolute indices in the full dataset:
            update_indices = need_update
            
            if update_indices.numel() > 0:
                # Use indices from FAISS to fetch corresponding latents from the pool.
                # Ensure that pool_latents is on the same device (or move it accordingly).
                new_latents = self.pool_latents[indices_tensor[need_update].to(self.pool_latents.device)].clone()
                
                # Add random perturbation as in your original function.
                perturbation = self.H.imle_perturb_coef * torch.randn(
                    (need_update.sum().item(), self.H.latent_dim),
                    device=new_latents.device
                )
                new_latents.add_(perturbation)
                
                # Update temporary distances and latents.
                self.selected_dists_tmp[update_indices] = distances_tensor[need_update].to(self.selected_dists_tmp.device)
                self.selected_latents_tmp[update_indices] = new_latents
            else:
                print("No updates found in this iteration.")

        # After processing, update the selected and last-selected latents.
        self.last_selected_latents = self.selected_latents
        self.selected_latents = self.selected_latents_tmp

        print(f"Force resampling took {time.time() - t1:.2f} seconds")
