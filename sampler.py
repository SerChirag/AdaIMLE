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
from diffusers.models import AutoencoderKL


class Sampler:
    def __init__(self, H, sz, preprocess_fn):
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduce=False).cuda()
        self.H = H
        self.latent_lr = H.latent_lr
        self.entire_ds = torch.arange(sz)
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.last_selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float32)
        self.selected_latents_tmp = torch.empty([sz, H.latent_dim], dtype=torch.float32)

        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))
        self.neutral_snoise = [torch.zeros([self.H.imle_db_size, 1, s, s], dtype=torch.float32) for s in self.res]

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").cuda()
        for param in self.vae.parameters():
            param.requires_grad = False


        if(H.use_snoise == True):
            self.snoise_tmp = [torch.randn([self.H.imle_db_size, 1, s, s], dtype=torch.float32) for s in self.res]
            self.selected_snoise = [torch.randn([sz, 1, s, s,], dtype=torch.float32) for s in self.res]
            self.snoise_pool = [torch.randn([self.pool_size, 1, s, s], dtype=torch.float32) for s in self.res]
        else:
            self.snoise_tmp = [torch.zeros([self.H.imle_db_size, 1, s, s], dtype=torch.float32) for s in self.res]
            self.selected_snoise = [torch.zeros([sz, 1, s, s,], dtype=torch.float32) for s in self.res]
            self.snoise_pool = [torch.zeros([self.pool_size, 1, s, s], dtype=torch.float32) for s in self.res]
            
        self.selected_dists = torch.empty([sz], dtype=torch.float32)
        self.selected_dists[:] = np.inf
        self.selected_dists_tmp = torch.empty([sz], dtype=torch.float32)

        self.selected_dists_lpips = torch.empty([sz], dtype=torch.float32).cuda()
        self.selected_dists_lpips[:] = np.inf

        self.selected_dists_l2 = torch.empty([sz], dtype=torch.float32).cuda()
        self.selected_dists_l2[:] = np.inf 

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float32)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float32)

        self.pool_latents = torch.randn([self.pool_size, H.latent_dim], dtype=torch.float32)
        self.sample_pool_usage = torch.ones([sz], dtype=torch.bool)

        self.lpips_net = LPNet(pnet_type=H.lpips_net, path=H.lpips_path).cuda()
        self.l2_projection = None

        self.knn = KNN(k=1, transpose_mode=True)

        fake = torch.zeros(1, 3, H.image_size, H.image_size).cuda()

        if(H.search_type == 'lpips'):
            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample, antialias=True, mode='bicubic')
            out = self.encode(interpolated).reshape(interpolated.shape[0],-1)
            self.l2_projection = F.normalize(torch.randn(out.shape[1], H.proj_dim), p=2, dim=1).cuda()
            sum_dims = H.proj_dim

        elif(H.search_type == 'l2'):
            interpolated = F.interpolate(fake,scale_factor = H.l2_search_downsample, antialias=True, mode='bicubic')
            interpolated = interpolated.reshape(interpolated.shape[0],-1)
            self.l2_projection = F.normalize(torch.randn(interpolated.shape[1], H.proj_dim), p=2, dim=1).cuda()
            sum_dims = H.proj_dim

        self.dci_dim = sum_dims
        print('dci_dim', self.dci_dim)

        self.temp_samples_proj = torch.empty([self.H.imle_db_size, sum_dims], dtype=torch.float32).cuda()
        self.dataset_proj = torch.empty([sz, sum_dims], dtype=torch.float32).cuda()
        self.pool_samples_proj = torch.empty([self.pool_size, sum_dims], dtype=torch.float32).cuda()

        self.knn_ignore = H.knn_ignore
        self.ignore_radius = H.ignore_radius
        self.resample_angle = H.resample_angle

        self.total_excluded = 0
        self.total_excluded_percentage = 0
        self.dataset_size = sz
        self.db_iter = 0
    
    def encode(self, image):
        return self.vae.encode(image).latent_dist.sample().mul_(self.vae.config.scaling_factor)

    def forward_decoder(self, latents):
        return self.vae.decode(latents/self.vae.config.scaling_factor).sample

    def get_projected(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)
        
        interpolated = F.interpolate(inp,scale_factor = self.H.l2_search_downsample, antialias=True, mode='bicubic')
        out = self.encode(interpolated.cuda()).reshape(interpolated.shape[0],-1).detach()
        out = torch.mm(out, self.l2_projection)
        # interpolated = F.normalize(interpolated, p=2, dim=1)
        return out.cuda()

    def get_flattened(self, inp, permute=True):
        out = torch.mm(inp.reshape(inp.shape[0],-1), self.l2_projection)
        # interpolated = F.normalize(interpolated, p=2, dim=1)
        return out.cuda()
    
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

    def init_projection(self, dataset):

        for ind, x in enumerate(DataLoader(TensorDataset(dataset), batch_size=self.H.n_batch)):
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + x[0].shape[0])
            if(self.H.search_type == 'lpips'):
                self.dataset_proj[batch_slice] = self.get_projected(self.preprocess_fn(x)[1])
            elif(self.H.search_type == 'l2'):
                self.dataset_proj[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[1])
            else:
                self.dataset_proj[batch_slice] = self.get_combined_feature(self.preprocess_fn(x)[1])

    def sample(self, latents, gen, snoise=None):
        with torch.no_grad():
            nm = latents.shape[0]
            out = gen(latents, snoise)
            px_z = self.forward_decoder(out).permute(0, 2, 3, 1)
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
        if snoise is None:
            snoise = self.selected_snoise

        if to_update is not None:
            latents = latents[to_update]
            dists = dists[to_update]
            dataset_tensor = dataset_tensor[to_update]
            snoise = [s[to_update] for s in snoise]

        for ind, x in enumerate(DataLoader(TensorDataset(dataset_tensor), batch_size=self.H.n_batch)):
            _, target = self.preprocess_fn(x)
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + target.shape[0])
            cur_latents = latents[batch_slice]
            cur_snoise = [s[batch_slice] for s in snoise]
            with torch.no_grad():
                out = torch.from_numpy(self.sample(cur_latents, gen)).cuda()
                _, out = self.preprocess_fn([out])

                if(logging):
                    dist, dist_lpips, dist_l2 = self.calc_loss(target.permute(0, 3, 1, 2), out.permute(0, 3, 1, 2), use_mean=False, logging=True)
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
        if snoise is None:
            snoise = self.selected_snoise

        if to_update is not None:
            latents = latents[to_update]
            dists = dists[to_update]
            dataset_tensor = dataset_tensor[to_update]
            snoise = [s[to_update] for s in snoise]

        for ind, x in enumerate(DataLoader(TensorDataset(dataset_tensor), batch_size=self.H.n_batch)):
            _, target = self.preprocess_fn(x)
            batch_slice = slice(ind * self.H.n_batch, ind * self.H.n_batch + target.shape[0])
            cur_latents = latents[batch_slice]
            cur_snoise = [s[batch_slice] for s in snoise]
            with torch.no_grad():
                out = gen(cur_latents, cur_snoise)
                if(self.H.search_type == 'lpips'):
                    dist = self.calc_loss_projected(target.permute(0, 3, 1, 2), out)
                else:
                    dist = self.calc_loss_l2(target.permute(0, 3, 1, 2), out)
                dists[batch_slice] = torch.squeeze(dist)
        return dists

    def imle_sample(self, dataset, gen, factor=None):
        if factor is None:
            factor = self.H.imle_factor
        imle_pool_size = int(len(dataset) * factor)
        t1 = time.time()
        self.selected_dists_tmp[:] = self.selected_dists[:]
        for i in range((imle_pool_size // self.H.imle_db_size)+1):
            self.temp_latent_rnds.normal_()
            for j in range(len(self.res)):
                if(self.H.use_snoise == True):
                    self.snoise_tmp[j].normal_()
            for j in range(self.H.imle_db_size // self.H.imle_batch):
                batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)
                cur_latents = self.temp_latent_rnds[batch_slice]
                cur_snoise = [x[batch_slice] for x in self.snoise_tmp]
                with torch.no_grad():
                    self.temp_samples[batch_slice] = gen(cur_latents, cur_snoise)
                    if(self.H.search_type == 'lpips'):
                        self.temp_samples_proj[batch_slice] = self.get_projected(self.temp_samples[batch_slice], False)
                    elif(self.H.search_type == 'l2'):
                        self.temp_samples_proj[batch_slice] = self.get_l2_feature(self.temp_samples[batch_slice], False)
                    else:
                        self.temp_samples_proj[batch_slice] = self.get_combined_feature(self.temp_samples[batch_slice], False)

            if not gen.module.dci_db:
                device_count = torch.cuda.device_count()

                gen.module.dci_db = MDCI(self.temp_samples_proj.shape[1], num_comp_indices=self.H.num_comp_indices,
                                            num_simp_indices=self.H.num_simp_indices, devices=[i for i in range(device_count)], ts=device_count)

                # gen.module.dci_db = DCI(self.temp_samples_proj.shape[1], num_comp_indices=self.H.num_comp_indices,
                                            # num_simp_indices=self.H.num_simp_indices)
            gen.module.dci_db.add(self.temp_samples_proj)

            t0 = time.time()
            for ind, y in enumerate(DataLoader(dataset, batch_size=self.H.imle_batch)):
                # t2 = time.time()
                _, target = self.preprocess_fn(y)
                x = self.dataset_proj[ind * self.H.imle_batch:ind * self.H.imle_batch + target.shape[0]]
                cur_batch_data_flat = x.float()

                nearest_indices, _ = gen.module.dci_db.query(cur_batch_data_flat, num_neighbours=1)
                nearest_indices = nearest_indices.long()[:, 0]

                batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + x.size()[0])
                actual_selected_dists = self.calc_loss(target.permute(0, 3, 1, 2),
                                                       self.temp_samples[nearest_indices].cuda(), use_mean=False)
                # actual_selected_dists = torch.squeeze(actual_selected_dists)

                to_update = torch.nonzero(actual_selected_dists < self.selected_dists[batch_slice], as_tuple=False)
                to_update = torch.squeeze(to_update)
                self.selected_dists[ind * self.H.imle_batch + to_update] = actual_selected_dists[to_update].clone()
                self.selected_latents[ind * self.H.imle_batch + to_update] = self.temp_latent_rnds[nearest_indices[to_update]].clone()
                for k in range(len(self.res)):
                    self.selected_snoise[k][ind * self.H.imle_batch + to_update] = self.snoise_tmp[k][nearest_indices[to_update]].clone()

                del cur_batch_data_flat

        # adding perturbation
        changed = torch.sum(self.selected_dists_tmp != self.selected_dists).item()
        print("Samples and NN are calculated, time: {}, mean: {} # changed: {}, {}%".format(time.time() - t1,
                                                                                            self.selected_dists.mean(),
                                                                                            changed, (changed / len(
                dataset)) * 100))
      
    def resample_pool(self, gen, ds):
        # self.init_projection(ds)
        self.pool_latents.normal_()
        gen.eval()

        for j in range(self.pool_size // self.H.imle_batch):
            batch_slice = slice(j * self.H.imle_batch, (j + 1) * self.H.imle_batch)

            if(self.H.use_angular_resample):
                cur_latents = self.sample_angle(self.pool_latents[batch_slice])
            
            else:
                cur_latents = self.pool_latents[batch_slice]

            cur_snosie = [s[batch_slice] for s in self.snoise_pool]
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if(self.H.search_type == 'lpips'):
                        self.pool_samples_proj[batch_slice] = self.get_flattened(gen(cur_latents, cur_snosie), False)
                    elif(self.H.search_type == 'l2'):
                        self.pool_samples_proj[batch_slice] = self.get_l2_feature(gen(cur_latents, cur_snosie), False)
                    else:
                        self.pool_samples_proj[batch_slice] = self.get_combined_feature(gen(cur_latents, cur_snosie), False)

        gen.train()

    def imle_sample_force(self, dataset, gen, to_update=None):
        if to_update is None:
            to_update = self.entire_ds
        if to_update.shape[0] == 0:
            return
        
        to_update = to_update.cpu()

        t1 = time.time()
        if torch.any(self.sample_pool_usage[to_update]):
            self.resample_pool(gen, dataset)
            self.sample_pool_usage[:] = False
            print(f'resampling took {time.time() - t1}')

        self.selected_dists_tmp[:] = np.inf
        self.sample_pool_usage[to_update] = True

        ## removing samples too close

        total_rejected = 0

        self.total_excluded = total_rejected
        self.total_excluded_percentage = (total_rejected * 1.0 / self.pool_size) * 100

        with torch.no_grad():
            for i in range(self.pool_size // self.H.imle_db_size):
                pool_slice = slice(i * self.H.imle_db_size, (i + 1) * self.H.imle_db_size)
                pool_latents = self.pool_latents[pool_slice]

                t0 = time.time()
                for ind, y in enumerate(DataLoader(TensorDataset(dataset[to_update]), batch_size=self.H.imle_batch)):
                    _, target = self.preprocess_fn(y)
                    batch_slice = slice(ind * self.H.imle_batch, ind * self.H.imle_batch + target.shape[0])
                    indices = to_update[batch_slice]
                    x = self.dataset_proj[indices]

                    nearest_dist , nearest_indices = self.knn(torch.unsqueeze(self.pool_samples_proj[pool_slice],0), torch.unsqueeze(x,0))  # 32 x 50 x 10
                    nearest_indices = torch.squeeze(nearest_indices, dim=[0,2]).cpu()
                    nearest_dist = torch.squeeze(nearest_dist, dim=[0,2]).cpu()

                    need_update = nearest_dist < self.selected_dists_tmp[indices]
                    need_update = need_update.cpu()
                    global_need_update = indices[need_update]

                    self.selected_dists_tmp[global_need_update] = nearest_dist[need_update].clone()
                    self.selected_latents_tmp[global_need_update] = pool_latents[nearest_indices[need_update]].clone() + self.H.imle_perturb_coef * torch.randn((need_update.sum(), self.H.latent_dim))

                if i % 100 == 0:
                    print("NN calculated for {} out of {} - {}".format((i + 1) * self.H.imle_db_size, self.pool_size, time.time() - t0))
        

        self.last_selected_latents[to_update] = self.selected_latents[to_update]
        self.selected_latents[to_update] = self.selected_latents_tmp[to_update]

        print(f'Force resampling took {time.time() - t1}')
