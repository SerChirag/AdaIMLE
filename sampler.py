from curses import update_lines_cols
from math import comb, ceil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from transformers import AutoImageProcessor, AutoModel

from LPNet import LPNet
from helpers.utils import is_dist_avail_and_initialized, is_main_process, get_world_size, get_rank, safe_barrier
from models import parse_layer_string
from helpers.angle_sampler import Angle_Generator
from torch import autocast
from diffusers import AutoencoderTiny
import faiss
from tqdm import tqdm

class Sampler:
    def __init__(self, H, sz, preprocess_fn):
        
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.world_size = get_world_size()
        self.rank = get_rank()

        self.pool_size = ceil(int(H.pool_size_per_class) / H.imle_db_size) * H.imle_db_size
        self.preprocess_fn = preprocess_fn
        self.l2_loss = torch.nn.MSELoss(reduce=False).to(self.device)
        self.H = H
        self.latent_lr = H.latent_lr
        self.sz = sz
        self.selected_latents = torch.empty([sz, H.latent_dim], dtype=torch.float16)
        self.last_selected_latents = torch.empty([H.num_images_visualize, H.latent_dim], dtype=torch.float16)

        blocks = parse_layer_string(H.dec_blocks)
        self.block_res = [s[0] for s in blocks]
        self.res = sorted(set([s[0] for s in blocks if s[0] <= H.max_hierarchy]))

        self.temp_latent_rnds = torch.empty([self.H.imle_db_size, self.H.latent_dim], dtype=torch.float16)
        self.temp_samples = torch.empty([self.H.imle_db_size, H.image_channels, self.H.image_size, self.H.image_size],
                                        dtype=torch.float16)

        self.pool_latents = torch.empty([self.pool_size, H.latent_dim], dtype=torch.float16, device=self.device)
        self.projections = []
        self.lpips_net = LPNet(pnet_type=H.lpips_net, path=H.lpips_path).to(self.device)
        self.lpips_net.eval()
        self.lpips_net.requires_grad_(False)
        self.delta = H.huber_delta 

        ## TODO: check this is required or not
        if(self.H.compile):
            self.lpips_net = torch.compile(self.lpips_net)

        self.dino_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)

        self.dino_encoder = AutoModel.from_pretrained("./models--facebook--dinov2-base/snapshots/main").eval().to(self.device)
        
        if(self.H.compile):
            self.dino_encoder = torch.compile(self.dino_encoder)
        
        self.nn_search_batch = H.nn_search_batch


        # self.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(self.device)
        # # self.vae = AutoencoderTiny.from_pretrained("./tiny-auto/models--madebyollin--taesd/snapshots/main").to(self.device)
        # self.vae.eval()
        # self.vae.requires_grad_(False)

        self.l2_projection = None

        fake = torch.zeros(1, 3, H.image_size, H.image_size, device=self.device)

        safe_barrier()
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
            if(self.H.search_type == 'lpips'):
                self.dataset_proj[batch_slice] = self.get_projected(self.preprocess_fn(x[0])).cpu()
            elif(self.H.search_type == 'l2'):
                self.dataset_proj[batch_slice] = self.get_l2_feature(self.preprocess_fn(x[0])).cpu()
            # elif(self.H.search_type == 'vae'):
            #     self.dataset_proj[batch_slice] = self.get_vae_features(self.preprocess_fn(x[0])[1]).cpu()
            elif(self.H.search_type == 'combined'):
                self.dataset_proj[batch_slice] = self.get_combined_feature(self.preprocess_fn(x[0])).cpu()
            else:
                exit()

        # Convert to numpy
        self.dataset_proj = self.dataset_proj.numpy().astype(np.float16)
        # print(f"Rank {self.rank} class ranges: {self.class_ranges}")


    def sample(self, latents, labels, gen, snoise=None):
        with torch.no_grad():
            with autocast(device_type='cuda'):
                latents = latents.to(self.device)
                px_z = gen(latents, labels).permute(0, 2, 3, 1)
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
            return res.mean(dim=tuple(range(1, res.ndim)))
    
    def pseudo_huber(self, diff):
        return 2.0 * self.delta**2 * (torch.sqrt(1 + (diff / (self.delta)**2)) - 1)
    
    def get_dino_loss(self, inp, tar, use_mean=True):
        dino_feat = self.get_dino_features(inp, scale_factor=1, permute=False)
        tar_feat = self.get_dino_features(tar, scale_factor=1, permute=False)
        dino_loss = self.l2_loss(dino_feat, tar_feat)
        if use_mean:
            return dino_loss.mean()
        else:
            return dino_loss.mean(dim=tuple(range(1, dino_loss.ndim)))

    def calc_loss(self, inp, tar):

        l2_loss = self.l2_loss(inp, tar).mean(dim=tuple(range(1, inp.ndim)))
        res = 0
        
        lpips_loss = self.get_lpips_loss(inp, tar, use_mean=False)

        if(inp.shape[2] < 32):
            dino_loss = self.get_dino_loss(inp, tar, use_mean=False)
        else:
            dino_loss = torch.tensor(0.0, device=self.device)

        residuals = self.H.lpips_coef * lpips_loss + self.H.l2_coef * l2_loss + self.H.dino_coef * dino_loss

        if(self.H.loss_type == 'huber'):
            loss = self.pseudo_huber(residuals).mean()
        else:
            loss = residuals.mean() 
        return loss
        
            
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
                    self.pool_samples_proj[batch_slice] = proj.to('cpu').detach().cpu().numpy().astype(np.float16)

        gen.train()


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
                self.gpu_index_flat.reset()


                # Obtain the full dataset features (on CPU) and then slice locally.
                local_ds_feats = self.dataset_proj[self.class_ranges[i][0]:self.class_ranges[i][1]]

                self.gpu_index_flat.add(self.pool_samples_proj)  # add entire pool

                # Perform NN search for the local chunk. Returns arrays of shape (local_size, 1).
                _, indices = self.gpu_index_flat.search(local_ds_feats, 1)
                local_indices   = torch.from_numpy(indices).squeeze(1)    # (local_size,)


                new_latents = self.pool_latents[local_indices].clone()
                all_pool_latents.append(new_latents)
            
        all_pool_latents = torch.cat(all_pool_latents, dim=0).detach().cpu()

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
