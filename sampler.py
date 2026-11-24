from math import ceil
import copy
import os
import time

import numpy as np
import scipy.optimize
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel

from LPNet import LPNet
from helpers.utils import is_main_process, get_world_size, get_rank, safe_barrier
from models import parse_layer_string
from torch import autocast
import faiss
import faiss.contrib.torch_utils
from tqdm import tqdm
from helpers.autoencoder import (load_autoencoder, encode_images_to_latents, decode_latents_to_images,
                                 decode_latents_to_images_differentiable)
from helpers.cache_utils import latent_cache_key, load_latent_cache, save_latent_cache
from helpers.lpips_vgg import load_lpips_vgg
from helpers.dino_spatial import load_dino_spatial

class Sampler:
    def __init__(self, H, sz, preprocess_fn, autoencoder=None):
        
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.world_size = get_world_size()
        self.rank = get_rank()

        self.num_classes = getattr(H, 'num_classes', 0)
        self.class_ranges = None  # populated in init_projection

        if self.num_classes > 0:
            per_class_n = sz // self.num_classes
            if getattr(H, 'pool_size_per_class', 0) > 0:
                self.pool_size_per_class = int(H.pool_size_per_class)
            else:
                self.pool_size_per_class = ceil(H.force_factor * per_class_n)
            self.local_classes = list(range(self.rank, self.num_classes, self.world_size))
            # Global pool size not used in conditional path, but set for buffer-sizing compat
            self.pool_size = self.pool_size_per_class
        else:
            self.pool_size = ceil(int(H.force_factor * sz) / H.imle_db_size) * H.imle_db_size
            self.local_classes = []

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

        self.pool_latents = None

        self.decode_for_metrics = bool(getattr(H, 'autoencoder_decode_for_metrics', True))
        self.autoencoder = autoencoder if autoencoder is not None else load_autoencoder(H, self.device)
        if is_main_process():
            ae_name = type(self.autoencoder).__name__
            ae_source = getattr(getattr(self.autoencoder, 'config', None), '_name_or_path', 'unknown')
            print(f'\n[autoencoder] Loaded {ae_name} from {ae_source}\n')
        # Do not compile the autoencoder — it runs in frozen inference-only mode rarely
        # (FID, visualization). CUDA graph capture for the VAE decoder is expensive and
        # its graph memory stays resident permanently, causing VRAM spikes after FID runs.

        self.autoencoder_native_latent_size = None
        fake_rgb = torch.zeros(1, 3, H.image_size, H.image_size, device=self.device)
        native_latents = encode_images_to_latents(self.autoencoder, fake_rgb, target_spatial=None)
        self.autoencoder_native_latent_size = (native_latents.shape[-2], native_latents.shape[-1])

        # Perceptual loss in decoder (pixel) space. Only built when actually used, since
        # it pulls a VGG16 into VRAM and forces a differentiable decode every step.
        self.lpips_coef = float(getattr(H, 'lpips_coef', 0.0))
        self.lpips_net = None
        if self.lpips_coef > 0.0:
            lpips_path = os.path.join(getattr(H, 'lpips_path', './lpips'), 'weights/v0.1/vgg.pth')
            self.lpips_net = load_lpips_vgg(self.device, lin_path=lpips_path)
            if is_main_process():
                print(f'\n[lpips] Decoder-space LPIPS-VGG enabled (coef={self.lpips_coef}, '
                      f'latent_loss_coef={float(getattr(H, "latent_loss_coef", 1.0))}), '
                      f'weights from {lpips_path}\n')

        # Alternative perceptual loss: spatial DINO patch tokens (--dino_coef). Either this or
        # LPIPS, unless --lpips_with_dino (then both share one decode: calc_dino_lpips_loss).
        self.dino_coef = float(getattr(H, 'dino_coef', 0.0))
        self.dino_net = None
        if self.dino_coef > 0.0:
            dino_path = getattr(H, 'dino_path', 'facebook/dinov2-base')
            self.dino_net = load_dino_spatial(self.device, path=dino_path, size=int(getattr(H, 'dino_size', 224)))
            if is_main_process():
                print(f'\n[dino] Decoder-space spatial DINO loss enabled (coef={self.dino_coef}, '
                      f'latent_loss_coef={float(getattr(H, "latent_loss_coef", 1.0))}, '
                      f'weights from {dino_path}, {self.dino_net.size}px, skipping {self.dino_net.num_skip} '
                      f'CLS/register tokens)\n')

        # --perc_bf16: bf16 copy of the VAE decoder (+ post_quant_conv) for the perceptual-loss decodes, and
        # DINO under bf16 autocast. Frozen weights; gradients flow through the bf16 activations to the latents.
        self.perc_bf16 = bool(getattr(H, 'perc_bf16', False))
        self._perc_dec16 = self._perc_pq16 = None
        if self.perc_bf16 and (self.lpips_coef > 0.0 or self.dino_coef > 0.0):
            if not (hasattr(self.autoencoder, 'decoder') and hasattr(self.autoencoder, 'post_quant_conv')):
                raise ValueError('--perc_bf16 needs an AutoencoderKL-style autoencoder (decoder + post_quant_conv).')
            self._perc_dec16 = copy.deepcopy(self.autoencoder.decoder).to(torch.bfloat16).to(
                memory_format=torch.channels_last).eval().requires_grad_(False)
            use_pq = getattr(getattr(self.autoencoder, 'config', None), 'use_post_quant_conv', True)
            if use_pq and self.autoencoder.post_quant_conv is not None:
                self._perc_pq16 = copy.deepcopy(self.autoencoder.post_quant_conv).to(torch.bfloat16).eval().requires_grad_(False)
            if is_main_process():
                print('[perc] perceptual-loss decode in bf16 (decoder copy), DINO under bf16 autocast')

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
        self.latent_channels = H.image_channels
        self.dataset_proj_torch = torch.empty([sz, sum_dims], dtype=torch.float32, device='cpu')
        self.dataset_proj = None
        self.pool_samples_proj = None
        self._dataset_proj_gpu = None
        self._local_pool_latents = None
        self._local_pool_proj = None
        self._local_pool_combined = None
        self._gathered_combined_main = None
        self._full_combined_main = None

        # RS-IMLE (see --use_rs_imle): fixed rejection radius epsilon.
        self.use_rs_imle = bool(getattr(H, 'use_rs_imle', False))
        self.rs_radius = float(getattr(H, 'rs_radius', 0.0))
        if self.use_rs_imle and self.rs_radius <= 0.0:
            raise ValueError('--use_rs_imle needs --rs_radius > 0.')
        self.rs_reject_pct = 0.0

        # NN search / rejection space (see --match_space). In 'dino' mode the training targets stay the
        # latents in dataset_proj_torch; only the matching uses DINO features. Training-image features are
        # built lazily, per rank for its local classes only (~2.5 GB bf16 per rank for 125 classes).
        self.match_space = getattr(H, 'match_space', 'latent')
        self.match_dino = None
        self._dino_ds_feats = None
        if self.match_space == 'dino':
            if self.num_classes <= 0:
                raise ValueError('--match_space dino is only implemented for the conditional search.')
            self.match_dino = self.dino_net if self.dino_net is not None else load_dino_spatial(
                self.device, path=getattr(H, 'dino_path', 'facebook/dinov2-base'), size=int(getattr(H, 'dino_size', 224)))
            # bf16 copy of the VAE decoder for decoding the pool at every resample: ~2x faster than the fp32
            # decode, ~0.004 median shift in the spatial DINO distance (a few % of images move by > eps/2).
            # The cached training-image features are still decoded in fp32 (once), so only one side is noisy.
            self._match_dec16 = None
            if hasattr(self.autoencoder, 'decoder') and hasattr(self.autoencoder, 'post_quant_conv'):
                self._match_dec16 = copy.deepcopy(self.autoencoder.decoder).to(torch.bfloat16).to(
                    memory_format=torch.channels_last).eval()
                use_pq = getattr(getattr(self.autoencoder, 'config', None), 'use_post_quant_conv', True)
                self._match_pq16 = (copy.deepcopy(self.autoencoder.post_quant_conv).to(torch.bfloat16).eval()
                                    if use_pq and self.autoencoder.post_quant_conv is not None else None)
            if is_main_process():
                print(f'[match] NN search / RS rejection in DINO spatial space '
                      f'({self.match_dino.num_patches} patches), rs_radius={self.rs_radius}, '
                      f'pool decode {"bf16" if self._match_dec16 is not None else "fp32"}')

        # Balanced matching (see --match_balanced): each training image owns `balanced_quota` pool latents per
        # resample, kept in balanced_latents [sz, quota, latent_dim] (nearest first); set_balanced_slot() copies
        # one slot per epoch into selected_latents, so the training loop / visualization are unchanged.
        self.match_balanced = bool(getattr(H, 'match_balanced', False))
        self.balanced_priority = bool(getattr(H, 'balanced_priority', False))
        self.balanced_latents = None
        self.balanced_moved = None          # [sz, quota] bool: slot's latent currently copies a different image
        self.balanced_moved_pct = 0.0
        if self.balanced_priority and not self.match_balanced:
            raise ValueError('--balanced_priority needs --match_balanced True.')
        self.balanced_moved_weight = float(getattr(H, 'balanced_moved_weight', 1.0))
        self.balanced_extra_moved = bool(getattr(H, 'balanced_extra_moved', False))
        if (self.balanced_moved_weight != 1.0 or self.balanced_extra_moved) and not self.match_balanced:
            raise ValueError('--balanced_moved_weight / --balanced_extra_moved need --match_balanced True.')
        if self.balanced_extra_moved and self.balanced_priority:
            raise ValueError('--balanced_extra_moved adds the priority pick on top of the default rotation; '
                             'do not combine it with --balanced_priority.')
        self.balanced_rs_radius = float(getattr(H, 'balanced_rs_radius', 0.0) or 0.0)
        if self.balanced_rs_radius > 0 and not self.match_balanced:
            raise ValueError('--balanced_rs_radius needs --match_balanced True.')
        if self.balanced_rs_radius > 0 and (self.balanced_priority or self.balanced_extra_moved):
            raise ValueError('--balanced_rs_radius picks the slot itself; do not combine it with '
                             '--balanced_priority / --balanced_extra_moved.')
        self.balanced_dist = None           # [sz, q] distance of each slot's latent to its assigned image
        self.balanced_active = None         # [sz] bool (--balanced_rs_radius): image has a slot outside the radius
        self.balanced_zone_stats = {}
        self.balanced_slot_moved = None     # [sz] bool: the slot trained this epoch is moved
        self.balanced_extra_idx = None      # [n_extra] long: images that get an extra moved-latent example
        self.balanced_extra_latents = None  # [n_extra, latent_dim]
        if self.match_balanced:
            if self.num_classes <= 0:
                raise ValueError('--match_balanced is only implemented for the conditional search.')
            if self.use_rs_imle:
                raise ValueError('--match_balanced replaces RS rejection: pass --use_rs_imle False.')
            if float(H.force_factor) != int(H.force_factor) or int(H.force_factor) < 1:
                raise ValueError('--match_balanced needs an integer --force_factor (the per-image quota).')
            self.balanced_quota = int(H.force_factor)
            if self.balanced_quota * (sz // self.num_classes) > self.pool_size_per_class:
                raise ValueError('--match_balanced: pool_size_per_class is smaller than quota * images per class.')
            if is_main_process():
                print(f'[match] balanced matching: {self.balanced_quota} latents per training image '
                      f'(pool {self.pool_size_per_class}/class, {self.match_space} space)')

        self.dataset_size = sz
        self.db_iter = 0
        self.generator_seed = torch.Generator(device=self.device)         
        self.generator_seed.manual_seed(H.seed + self.rank)

        self.compress_comm = bool(getattr(H, 'compress_comm', True))
        if self.compress_comm:
            self._comm_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self._comm_dtype = torch.float32

        self.faiss_res = None

        self.faiss_res = faiss.StandardGpuResources()  # one per process
        # FlatL2 needs no temp memory; cap it to avoid competing with PyTorch's allocator.
        self.faiss_res.setTempMemory(64 * 1024 * 1024)  # 64 MB
        index_flat = faiss.IndexFlatL2(self.dci_dim)
        dev_id = torch.cuda.current_device()
        self.faiss_index_flat = faiss.index_cpu_to_gpu(self.faiss_res, dev_id, index_flat)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    
    def get_l2_feature(self, inp, permute=True):
        if(permute):
            inp = inp.permute(0, 3, 1, 2)

        interpolated = inp.reshape(inp.shape[0],-1)
        # interpolated = F.normalize(interpolated, p=2, dim=1)
        return interpolated


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
                autoencoder_type=getattr(self.H, 'autoencoder_type', 'kl'),
                autoencoder_name_or_path=getattr(self.H, 'autoencoder_name_or_path', ''),
                image_channels=self.latent_channels,
                num_classes=self.num_classes,
                sorted_by_class=(self.num_classes > 0),
                cache_dataset_id=getattr(self.H, 'cache_dataset_id', ''),
            )
            cached = load_latent_cache(cache_dir, key, expected_size=self.sz)

        if cached is not None:
            if is_main_process():
                print(f"[cache] Loaded latent projections from cache "
                      f"({cached.shape[0]} samples, dim={cached.shape[1]}).")
            self.dataset_proj_torch.copy_(cached)
        else:
            ae_batch = getattr(self.H, 'ae_batch', self.H.imle_batch)
            dataloader = DataLoader(
                dataset,
                batch_size=ae_batch,
            )

            if is_main_process():
                print("Starting Initialization")

            with torch.inference_mode():
                for ind, x in tqdm(enumerate(dataloader), total=len(dataloader), desc="Initializing"):
                    batch_slice = slice(ind * ae_batch, ind * ae_batch + x[0].shape[0])
                    if self.H.search_type == 'l2':
                        self.dataset_proj_torch[batch_slice] = self.get_l2_feature(self.preprocess_fn(x)[-1]).cpu()
                    else:
                        exit()

            if use_cache and is_main_process():
                save_latent_cache(cache_dir, key, self.dataset_proj_torch)

        # Pin host memory so per-batch H2D copies can use faster async transfer.
        if torch.cuda.is_available() and not self.dataset_proj_torch.is_pinned():
            try:
                self.dataset_proj_torch = self.dataset_proj_torch.pin_memory()
            except RuntimeError as e:
                if is_main_process():
                    print(f"Warning: could not pin dataset_proj_torch ({e}); continuing without pinned cache.")

        # Keep a torch tensor for fast indexed target lookup in training,
        # and a NumPy view for FAISS nearest-neighbor search.
        self.dataset_proj = self.dataset_proj_torch.numpy()

        # Build class_ranges for conditional NN search.
        if self.num_classes > 0:
            labels_cpu = self.H.labels  # [sz] int64, class-sorted
            unique, counts = torch.unique_consecutive(labels_cpu, return_counts=True)
            starts = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)[:-1]])
            self.class_ranges = {
                int(c): (int(s), int(s + n))
                for c, s, n in zip(unique, starts, counts)
            }
            # A dataset holding only some of the classes (e.g. the first N class folders, labels unchanged):
            # skip the absent ones in the resampling loop. No-op when every class has images.
            self.local_classes = [c for c in self.local_classes if c in self.class_ranges]

        # Build a persistent GPU cache of query features.
        # For conditional: all ranks need it (each handles its own local_classes slice).
        # For unconditional: only rank 0 needs it.
        if self.num_classes > 0 or is_main_process():
            self._dataset_proj_gpu = self.dataset_proj_torch.to(self.device, non_blocking=True)

    def sample(self, latents, gen, snoise=None, condition=None):
        with torch.inference_mode():
            with autocast(device_type='cuda', dtype=self.H.amp_dtype_torch):
                latents = latents.to(self.device)
                cond = condition.to(self.device) if condition is not None else None
                px_z = gen(latents, cond)
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

    def calc_loss(self, inp, tar, use_mean=True, logging=False, weights=None):
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

        if weights is not None:
            # --old_weight: per-image weights [B]; mean over the batch (not over the weights), so
            # down-weighting old images shrinks their pull without scaling up the new ones'.
            per_img = per_elem.flatten(1).mean(dim=1) if per_elem.dim() > 1 else per_elem
            return (per_img * weights.to(per_img.dtype)).mean()
        return per_elem.mean()

    def _robust_lpips(self, d):
        """Optional saturating transform on the per-sample LPIPS distance ``d`` (>= 0).

        Mirrors calc_loss's robust residual transforms, but scaled for LPIPS' own numeric
        range (H.lpips_loss_scale) rather than H.loss_scale, since LPIPS-VGG distances are
        typically O(0.01-0.3) -- a different range than raw latent residuals.

        Purpose: without this, a latent contested between two very different real targets
        (its nearest-neighbor assignment keeps flipping between them across rounds) gets an
        unbounded perceptual pull toward each in turn, which a squared-error-like loss
        resolves by settling on their average -- a blurry "double exposure". A saturating
        transform caps how hard a single far-off target can pull, so the network is freer to
        commit to one target instead of compromising between them.

        H.lpips_robust_type == 'none' (the default) returns ``d`` unchanged, so default
        behavior -- and every existing config that doesn't set this flag -- is unaffected.
        """
        robust_type = getattr(self.H, 'lpips_robust_type', 'none') or 'none'
        if robust_type == 'none':
            return d
        scale = self.H.lpips_loss_scale
        if robust_type == 'mclure':
            return (d ** 2) / (scale ** 2 + d ** 2)
        elif robust_type == 'welsch':
            return 1 - torch.exp(-(d / scale) ** 2)
        elif robust_type == 'cauchy':
            return torch.log(1 + 0.5 * (d / scale) ** 2)
        elif robust_type == 'huber':
            return self.pseudo_huber(d ** 2)
        else:
            raise ValueError(f'Unknown lpips_robust_type: {robust_type}')

    def _decode_pair(self, inp, tar):
        """Decode the prediction and target latents (BCHW at ``latent_spatial_size``) to images.

        Only ``inp``'s decode is differentiable: ``tar`` is a fixed property of the dataset image,
        so decoding it under ``no_grad`` keeps the backward graph to a single decoder pass.
        """
        if self._perc_dec16 is not None and (inp.shape[-2], inp.shape[-1]) == tuple(self.autoencoder_native_latent_size):
            with torch.no_grad():
                img_tar = self._perc_decode16(tar.detach()).clamp(-1.0, 1.0)
            return self._perc_decode16(inp), img_tar

        with torch.no_grad():
            img_tar = decode_latents_to_images_differentiable(
                self.autoencoder, tar.detach(), self.autoencoder_native_latent_size)
            img_tar = img_tar.clamp(-1.0, 1.0)

        img_inp = decode_latents_to_images_differentiable(
            self.autoencoder, inp, self.autoencoder_native_latent_size)
        return img_inp, img_tar

    def _perc_decode16(self, z):
        """Differentiable decode with the bf16 decoder copy (unclamped, like the fp32 path); returns fp32."""
        sf = getattr(self.autoencoder, '_cached_scaling_factor', 1.0)
        z16 = (z / sf).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
        if self._perc_pq16 is not None:
            z16 = self._perc_pq16(z16)
        return self._perc_dec16(z16).float()

    def calc_lpips_loss(self, inp, tar):
        """LPIPS between the decoded prediction and the decoded target latent.

        VGG expects fp32, and the decoder is forced out of autocast anyway, so the whole
        term runs at full precision regardless of the ambient autocast context.
        """
        if self.lpips_net is None:
            return torch.zeros((), device=inp.device, dtype=torch.float32)

        img_inp, img_tar = self._decode_pair(inp, tar)
        with autocast(device_type='cuda', enabled=False):
            per_sample = self.lpips_net(img_inp.float(), img_tar.float())  # (B,), >= 0
            return self._robust_lpips(per_sample).mean()

    def calc_dino_loss(self, inp, tar):
        """Spatial DINO distance (helpers/dino_spatial.py) between the decoded prediction and the
        decoded target latent. Same decode path as calc_lpips_loss; fp32, outside autocast."""
        if self.dino_net is None:
            return torch.zeros((), device=inp.device, dtype=torch.float32)

        img_inp, img_tar = self._decode_pair(inp, tar)
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.perc_bf16):
            return self.dino_net(img_inp.float(), img_tar.float()).float().mean()

    def calc_dino_lpips_loss(self, inp, tar, weights=None):
        """--lpips_with_dino: (spatial DINO, LPIPS-VGG) between the decoded prediction and target, from ONE
        decode of the pair; each term exactly as calc_dino_loss / calc_lpips_loss compute it.
        weights: optional per-sample [B] (--balanced_moved_weight), mean over the batch like calc_loss."""
        img_inp, img_tar = self._decode_pair(inp, tar)
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.perc_bf16):
            dino = self.dino_net(img_inp.float(), img_tar.float()).float()
        with autocast(device_type='cuda', enabled=False):
            lp = self._robust_lpips(self.lpips_net(img_inp.float(), img_tar.float()))
        self._last_per_sample = (dino.detach(), lp.detach())   # train.py's own/moved logging split
        if weights is not None:
            if dino.dim() != 1 or lp.dim() != 1:
                raise ValueError(f'per-sample weights need per-sample losses, got {tuple(dino.shape)}, {tuple(lp.shape)}')
            w = weights.float()
            return (dino * w).mean(), (lp * w).mean()
        return dino.mean(), lp.mean()

    def resample_pool(self, gen, class_condition=None):

        # Determine local pool size
        if self.num_classes > 0:
            local_pool_size = self.pool_size_per_class
        else:
            local_pool_size = ceil(self.pool_size / self.world_size)

        # Reuse local buffers across resamples to avoid repeated allocations.
        if self._local_pool_latents is None or self._local_pool_latents.shape[0] != local_pool_size:
            self._local_pool_latents = torch.empty((local_pool_size, self.H.latent_dim), device=self.device)
            self._local_pool_combined = torch.empty(
                (local_pool_size, self.H.latent_dim + self.dci_dim),
                device=self.device,
                dtype=self._comm_dtype,
            )
            if self.rank == 0 and self.num_classes == 0:
                self._gathered_combined_main = [torch.empty_like(self._local_pool_combined) for _ in range(self.world_size)]

        # Regenerate the entire local pool each resample.
        self._local_pool_latents.normal_(mean=0.0, std=1.0, generator=self.generator_seed)

        # Process local chunk in batches, including the tail batch.
        with torch.inference_mode():
            for start in range(0, local_pool_size, self.H.imle_batch):
                end = min(start + self.H.imle_batch, local_pool_size)
                batch_slice = slice(start, end)
                cur_latents = self._local_pool_latents[batch_slice]
                self._local_pool_combined[batch_slice, :self.H.latent_dim].copy_(cur_latents.to(self._comm_dtype))
                with autocast(device_type='cuda', dtype=self.H.amp_dtype_torch):
                    if class_condition is not None:
                        class_tensor = torch.full(
                            (cur_latents.shape[0],), class_condition,
                            dtype=torch.long, device=self.device
                        )
                        outputs = gen(cur_latents, class_tensor)
                    else:
                        outputs = gen(cur_latents)
                    if self.H.search_type == 'l2':
                        proj = self.get_l2_feature(outputs, False)
                    else:
                        exit()
                    self._local_pool_combined[batch_slice, self.H.latent_dim:].copy_(proj.to(self._comm_dtype))

        if self.num_classes > 0:
            # Conditional: each rank owns its class's pool — no gather needed.
            self.pool_latents = self._local_pool_combined[:, :self.H.latent_dim].float()
            self.pool_samples_proj = self._local_pool_combined[:, self.H.latent_dim:].float()
            return

        # Unconditional: gather to rank 0.
        # One collective for both latents and projections to reduce comm overhead.
        if self.rank == 0:
            if self._gathered_combined_main is None or len(self._gathered_combined_main) != self.world_size:
                self._gathered_combined_main = [torch.empty_like(self._local_pool_combined) for _ in range(self.world_size)]
            torch.distributed.gather(self._local_pool_combined, gather_list=self._gathered_combined_main, dst=0)
        else:
            torch.distributed.gather(self._local_pool_combined, gather_list=None, dst=0)

        gen.train()  # unconditional path only — conditional caller manages eval/train

        # Aggregate the full pool latents and projected features.
        # cat into a pre-allocated combined buffer (one GPU kernel), then take contiguous
        # slices — no extra allocation beyond the buffer itself.
        if self.rank == 0:
            full_pool_size = local_pool_size * self.world_size
            combined_dim = self.H.latent_dim + self.dci_dim
            if (self._full_combined_main is None or self._full_combined_main.shape[0] != full_pool_size):
                self._full_combined_main = torch.empty((full_pool_size, combined_dim), dtype=torch.float32, device=self.device)
            torch.cat([c.to(torch.float32) for c in self._gathered_combined_main], dim=0, out=self._full_combined_main)
            self.pool_latents = self._full_combined_main[:, :self.H.latent_dim]
            self.pool_samples_proj = self._full_combined_main[:, self.H.latent_dim:]
    

    def nn_search_batched(self, queries, dataset):
        """Exact L2 nearest-neighbour search via FAISS with optional hard-first greedy Top-K."""
        if isinstance(queries, np.ndarray):
            queries_t = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float32)).to(self.device)
        else:
            queries_t = queries.to(self.device)

        if isinstance(dataset, np.ndarray):
            dataset_t = torch.from_numpy(np.ascontiguousarray(dataset, dtype=np.float32)).to(self.device)
        else:
            dataset_t = dataset.to(self.device)

        queries_t = queries_t.contiguous()
        dataset_t = dataset_t.contiguous()

        if getattr(self.H, 'nn_search_normalize', False):
            queries_t = F.normalize(queries_t, dim=-1)
            dataset_t = F.normalize(dataset_t, dim=-1)

        # RS-IMLE: drop pool samples within epsilon of any data point, search the rest,
        # then map indices back into the full pool.
        rs_index_map = None
        if self.use_rs_imle:
            dataset_t, rs_index_map = self._rs_filter_pool_faiss(queries_t, dataset_t)

        D, I = self._nn_search_faiss(queries_t, dataset_t)
        if rs_index_map is not None:
            I = rs_index_map.index_select(0, I.to(rs_index_map.device)).to(I.device)
        return D, I

    def _rs_filter_pool_faiss(self, queries_t, dataset_t):
        """
        Unconditional RS-IMLE rejection (exact). Search 1: index the data points and find each pool
        sample's nearest data point; reject every sample within epsilon of any data point.
        The actual NN search (search 2) then runs on the surviving pool.
        Returns (kept pool, map kept -> original index) or (pool, None) if nothing was rejected.
        """
        Nd = dataset_t.shape[0]
        self.faiss_index_flat.reset()
        self.faiss_index_flat.add(queries_t)
        D1, _ = self.faiss_index_flat.search(dataset_t, 1)  # squared L2, [Nd, 1]
        self.faiss_index_flat.reset()

        reject = (D1.squeeze(1) < self.rs_radius ** 2).to(dataset_t.device)
        n_rejected = int(reject.sum().item())
        if n_rejected == Nd:
            # Everything rejected: epsilon is too big, fall back to plain IMLE this round.
            n_rejected = 0
            reject.zero_()
        self.rs_reject_pct = 100.0 * n_rejected / Nd

        if n_rejected == 0:
            return dataset_t, None
        keep_idx = torch.nonzero(~reject, as_tuple=False).squeeze(1)
        return dataset_t.index_select(0, keep_idx).contiguous(), keep_idx

    def _nn_search_faiss(self, queries_t, dataset_t):
        topk = getattr(self.H, 'imle_db_topk', 1)

        self.faiss_index_flat.reset()
        self.faiss_index_flat.add(dataset_t)
        D, I = self.faiss_index_flat.search(queries_t, min(topk, dataset_t.shape[0]))
        self.faiss_index_flat.reset()

        if topk == 1:
            return D.squeeze(1).to(torch.float32), I.squeeze(1).to(torch.long)

        # Hard-first greedy assignment via numpy
        D_np = D.cpu().numpy()
        I_np = I.cpu().numpy()
        Nq, K = D_np.shape
        Nd = dataset_t.shape[0]

        # Sort all (query, candidate) pairs by distance ascending
        q_ids = np.repeat(np.arange(Nq), K)
        c_ids = I_np.flatten()
        d_vals = D_np.flatten()
        order = np.argsort(d_vals, kind='stable')

        assigned_q = np.full(Nq, -1, dtype=np.int64)
        assigned_d = np.full(Nq, np.inf, dtype=np.float32)
        used_c = np.zeros(Nd, dtype=bool)

        for pos in order:
            q, c = q_ids[pos], c_ids[pos]
            if assigned_q[q] == -1 and not used_c[c]:
                assigned_q[q] = c
                assigned_d[q] = d_vals[pos]
                used_c[c] = True

        # Fallback: unassigned queries get their k=1 match
        unassigned = np.where(assigned_q == -1)[0]
        if len(unassigned) > 0:
            assigned_q[unassigned] = I_np[unassigned, 0]
            assigned_d[unassigned] = D_np[unassigned, 0]

        return torch.from_numpy(assigned_d), torch.from_numpy(assigned_q)


    def imle_sample_force(self, gen, to_update=None):
        if self.num_classes > 0:
            self._imle_sample_force_conditional(gen)
        else:
            self._imle_sample_force_unconditional(gen)

    def _imle_sample_force_unconditional(self, gen):
        """
        Optimized force resampling routine using FAISS for batched nearest-neighbor search.
        In a DDP setting, each process contributes to the pool; rank 0 performs NN search,
        adds perturbation, and broadcasts the result to all processes.
        """
        if is_main_process():
            t1 = time.time()
            print("Starting pool resampling...")

        # Resample pool first (each process contributes its part);
        # this updates self.pool_samples_proj and self.pool_latents.
        gen.eval()
        self.resample_pool(gen)
        gen.train()

        if(is_main_process()):
            print(f"Resampling pool took {time.time() - t1:.2f} seconds")

        self.selected_dists_tmp[:] = np.inf

        with torch.inference_mode():

            if(is_main_process()):
                if (
                    self._dataset_proj_gpu is None
                    or self._dataset_proj_gpu.shape != self.dataset_proj_torch.shape
                    or self._dataset_proj_gpu.device != self.device
                ):
                    self._dataset_proj_gpu = self.dataset_proj_torch.to(self.device, non_blocking=True)

                local_ds_feats = self._dataset_proj_gpu

                # Pool features (as computed from resample_pool).
                pool_feats = self.pool_samples_proj

                # Perform NN search for the local chunk. Returns arrays of shape (local_size, 1).
                _, local_indices = self.nn_search_batched(local_ds_feats, pool_feats)

                # get count of unique indices for logging
                self.unique_indices = torch.unique(local_indices).numel() / self.sz

                local_indices = local_indices.to(device=self.pool_latents.device, non_blocking=True)
                new_latents = self.pool_latents.index_select(0, local_indices)

            if is_main_process():
                full_updated_latents = new_latents
                perturbation = self.H.imle_perturb_coef * torch.randn(
                    (self.sz, self.H.latent_dim),
                    device=self.device,
                    generator=self.generator_seed)
                full_updated_latents += perturbation
                comm_latents = full_updated_latents.to(self._comm_dtype)
            else:
                comm_latents = torch.empty(self.sz, self.H.latent_dim, dtype=self._comm_dtype, device=self.device)

            torch.distributed.broadcast(comm_latents, src=0)
            full_updated_latents = comm_latents.to(torch.float32)

            # Update last and current selected latents on all processes.
            self.last_selected_latents.copy_(self.selected_latents)
            self.selected_latents.copy_(full_updated_latents.cpu())

            if is_main_process():
                print(f"Force resampling took {time.time() - t1:.2f} seconds")

        self.faiss_index_flat.reset()

    def set_balanced_slot(self, epoch):
        """--match_balanced: pick which of each image's assigned latents to train this epoch.
        Default: slot (epoch % quota). With --balanced_priority: a moved slot (rotating among the image's moved
        slots by epoch), or slot 0 (its nearest, usually its own copy) when the image has no moved slot."""
        if self.balanced_latents is None:
            return
        rows = torch.arange(self.balanced_latents.shape[0])
        m = self.balanced_moved                                         # [sz, q] bool (None before the first match)
        if m is not None:
            n_moved = m.sum(dim=1)                                      # [sz]
            k = torch.remainder(torch.full_like(n_moved, epoch), n_moved.clamp(min=1))   # which moved slot this epoch
            rank = m.long().cumsum(dim=1) - 1                           # rank of each moved slot within its row
            pick = (m & (rank == k.unsqueeze(1))).float().argmax(dim=1)    # the k-th moved slot
        if self.balanced_rs_radius > 0 and self.balanced_dist is not None:
            # RS-IMLE's rule inside balanced matching: a slot whose latent is already within the radius of its
            # assigned image is not trained. Each image trains its NEAREST slot outside the radius (slots are
            # sorted nearest first, so the first True); an image with every slot inside skips this epoch.
            out = self.balanced_dist >= self.balanced_rs_radius        # [sz, q]
            self.balanced_active = out.any(dim=1)
            slot = torch.where(self.balanced_active, out.float().argmax(dim=1), torch.zeros(rows.numel(), dtype=torch.long))
        elif not self.balanced_priority or m is None:
            slot = torch.full((rows.numel(),), epoch % self.balanced_quota, dtype=torch.long)
        else:
            slot = torch.where(n_moved > 0, pick, torch.zeros_like(pick))
        self.selected_latents.copy_(self.balanced_latents[rows, slot])
        self.balanced_slot_moved = m[rows, slot] if m is not None else None
        if self.balanced_extra_moved and m is not None:
            # default rotation above, plus one extra example per image on its rotating moved slot
            self.balanced_extra_idx = torch.nonzero(n_moved > 0).flatten()
            self.balanced_extra_latents = self.balanced_latents[self.balanced_extra_idx, pick[self.balanced_extra_idx]]
            if is_main_process():
                print(f"[match] balanced_extra_moved: {self.balanced_extra_idx.numel()} extra moved-latent examples "
                      f"({100.0 * self.balanced_extra_idx.numel() / rows.numel():.1f}% of images)")

    @torch.inference_mode()
    def _dino_match_feats(self, latents_flat, fast=False):
        """Flattened latents (n, dci_dim; generator/dataset_proj layout) -> unit-norm DINO patch tokens,
        flattened to (n, num_patches * C) in bf16, for the spatial distance 1 - (a . b) / num_patches.
        fast=True decodes with the bf16 decoder copy (pool samples); otherwise the fp32 decode."""
        feats = []
        for start in range(0, latents_flat.shape[0], self.H.match_batch):
            z = latents_flat[start:start + self.H.match_batch].to(self.device).float()
            z = z.reshape(-1, self.latent_channels, self.latent_spatial_size, self.latent_spatial_size)
            if fast and self._match_dec16 is not None and \
                    (z.shape[-2], z.shape[-1]) == tuple(self.autoencoder_native_latent_size):
                sf = getattr(self.autoencoder, '_cached_scaling_factor', 1.0)
                z16 = (z / sf).to(torch.bfloat16).contiguous(memory_format=torch.channels_last)
                if self._match_pq16 is not None:
                    z16 = self._match_pq16(z16)
                imgs = self._match_dec16(z16).float().clamp_(-1.0, 1.0)
            else:
                imgs = decode_latents_to_images(self.autoencoder, z, self.autoencoder_native_latent_size)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                tok = self.match_dino._tokens(imgs.float())                     # (b, num_patches, C), unit-norm
            feats.append(tok.reshape(tok.shape[0], -1).to(torch.bfloat16))
        return torch.cat(feats)

    def _dino_match_dists(self, ds_feats, pool_feats):
        """Spatial DINO distance (n_real, n_pool) = 1 - mean patch cosine."""
        sim = (ds_feats @ pool_feats.T).float() / self.match_dino.num_patches
        return 1.0 - sim

    def _imle_sample_force_conditional(self, gen):
        """
        Conditional force resampling: each rank handles its local_classes.
        All local classes are processed in one fused forward pass loop — latents for all
        classes are concatenated with their class IDs and run through the generator in
        contiguous batches of imle_batch, then split back per class for NN search.
        all_reduce(SUM) merges results across ranks (each rank only writes its owned slices).
        """
        t1 = time.time()
        if is_main_process():
            print("Starting conditional pool resampling...")

        comm_latents = torch.zeros(
            self.sz, self.H.latent_dim,
            dtype=torch.float32, device=self.device
        )
        if self.match_balanced:
            # allocated outside inference_mode (like comm_latents): they are all_reduced / updated in place after it
            comm_balanced = torch.zeros(self.sz, self.balanced_quota, self.H.latent_dim,
                                        dtype=torch.float32, device=self.device)
            bal_stats = torch.zeros(2, dtype=torch.float64, device=self.device)  # [moved, assigned]
            comm_moved = torch.zeros(self.sz, self.balanced_quota, dtype=torch.float32, device=self.device)
            comm_bdist = torch.zeros(self.sz, self.balanced_quota, dtype=torch.float32, device=self.device)

        n_local = len(self.local_classes)
        if n_local == 0:
            gen.train()
            return

        total_pool = n_local * self.pool_size_per_class

        # Allocate / reuse fused buffers for all local classes at once.
        # Latents kept in float32 to avoid bfloat16 quantization degrading assignments.
        # Projections stored in _comm_dtype (bfloat16) since they are only used for NN search.
        if (self._local_pool_latents is None or self._local_pool_latents.shape[0] != total_pool):
            self._local_pool_latents = torch.empty((total_pool, self.H.latent_dim), device=self.device, dtype=torch.float32)
            self._local_pool_proj = torch.empty((total_pool, self.dci_dim), device=self.device, dtype=self._comm_dtype)

        # Fill latents and build class-ID tensor for all local classes
        self._local_pool_latents.normal_(mean=0.0, std=1.0, generator=self.generator_seed)
        class_ids = torch.repeat_interleave(
            torch.tensor(self.local_classes, device=self.device, dtype=torch.long),
            self.pool_size_per_class,
        )  # [total_pool]

        # Single fused forward pass loop over all local classes
        gen.eval()
        with torch.inference_mode():
            for start in range(0, total_pool, self.H.imle_batch):
                end = min(start + self.H.imle_batch, total_pool)
                cur_latents = self._local_pool_latents[start:end]
                cur_classes = class_ids[start:end]
                with autocast(device_type='cuda', dtype=self.H.amp_dtype_torch):
                    outputs = gen(cur_latents, cur_classes)
                    proj = self.get_l2_feature(outputs, False)
                self._local_pool_proj[start:end].copy_(proj.to(self._comm_dtype))

            if self.match_space == 'dino' and self._dino_ds_feats is None:
                t_ds = time.time()
                self._dino_ds_feats = {}
                for class_id in self.local_classes:
                    ds_start, ds_end = self.class_ranges[class_id]
                    if ds_end > ds_start:
                        # kept in pinned host memory (~2.5 GB/rank): on the GPU it pushed the training step OOM
                        self._dino_ds_feats[class_id] = self._dino_match_feats(
                            self.dataset_proj_torch[ds_start:ds_end]).cpu().pin_memory()
                if is_main_process():
                    print(f"[match] cached DINO features of the local training images in {time.time() - t_ds:.1f}s")

            # Per-class NN search using the fused buffers
            all_local_indices = []
            rs_stats = torch.zeros(2, dtype=torch.float64, device=self.device)  # [rejected, pool total]
            for i, class_id in enumerate(self.local_classes):
                start_pool = i * self.pool_size_per_class
                end_pool   = start_pool + self.pool_size_per_class
                pool_latents = self._local_pool_latents[start_pool:end_pool]          # float32

                ds_start, ds_end = self.class_ranges[class_id]
                if ds_end <= ds_start:
                    continue
                n_real = ds_end - ds_start
                if self.match_space == 'dino':
                    pool_feats = self._dino_match_feats(self._local_pool_proj[start_pool:end_pool], fast=True)
                    ds_feats = self._dino_ds_feats[class_id].to(self.device, non_blocking=True)
                    dists = self._dino_match_dists(ds_feats, pool_feats)  # [n_real, pool_size]
                else:
                    pool_feats = self._local_pool_proj[start_pool:end_pool].float()  # bfloat16 → float32 for cdist
                    class_ds_feats = self._dataset_proj_gpu[ds_start:ds_end]  # [n_real, dci_dim]
                    dists = torch.cdist(class_ds_feats, pool_feats)  # [n_real, pool_size]

                if self.use_rs_imle:
                    # Exact RS-IMLE rejection: drop every sample within epsilon of any image in the class.
                    reject = dists.min(dim=0).values < self.rs_radius  # [pool_size]
                    n_rej = int(reject.sum().item())
                    if n_rej == reject.numel():
                        n_rej = 0  # everything rejected: fall back to plain IMLE for this class
                    else:
                        # Add (not replace) a large finite offset: survivors always win, but rejected samples
                        # keep their order, so a real forced onto one (fewer survivors than reals under the
                        # 1-to-1 assignment) falls back to its nearest sample -- its own copy -- instead of
                        # an arbitrary tie-break. Finite (not inf) so linear_sum_assignment accepts the matrix.
                        dists = dists + (float(dists.max()) + 1e6) * reject.unsqueeze(0)
                    rs_stats[0] += n_rej
                    rs_stats[1] += reject.numel()

                if self.match_balanced:
                    # Capacity-constrained assignment: each training image gets `quota` slots and every slot a
                    # distinct pool sample, so surplus copies of an over-produced image go to the images that
                    # are short instead of staying untouched.
                    q = self.balanced_quota
                    if n_real * q > dists.shape[1]:
                        raise ValueError(f'--match_balanced: class {class_id} has {n_real} images, needs {n_real * q} '
                                         f'pool samples but the pool is {dists.shape[1]}; raise --pool_size_per_class.')
                    cost = dists.repeat_interleave(q, dim=0)                       # [n_real * q, pool_size]
                    _, col_ind = scipy.optimize.linear_sum_assignment(cost.cpu().numpy())
                    col = torch.from_numpy(col_ind).to(self.device, dtype=torch.long).view(n_real, q)
                    col = col.gather(1, dists.gather(1, col).argsort(dim=1))         # each image's slots, nearest first
                    owner = dists.argmin(dim=0)                                     # image each pool sample is closest to
                    moved = owner[col] != torch.arange(n_real, device=self.device).unsqueeze(1)    # [n_real, q]
                    comm_moved[ds_start:ds_end] = moved.float()
                    comm_bdist[ds_start:ds_end] = dists.gather(1, col).float()     # ascending per row
                    bal_stats[0] += moved.sum()
                    bal_stats[1] += col.numel()
                    comm_balanced[ds_start:ds_end] = pool_latents[col]             # [n_real, q, latent_dim]
                    local_indices = col[:, 0]
                elif self.H.imle_db_topk is not None and self.H.imle_db_topk > 1:
                    # Optimal 1-to-1 assignment: no pool latent shared across images.
                    _, col_ind = scipy.optimize.linear_sum_assignment(dists.cpu().numpy())
                    local_indices = torch.from_numpy(col_ind).to(self.device, dtype=torch.long)
                else:
                    local_indices = dists.argmin(dim=1)

                all_local_indices.append((local_indices, n_real))
                new_latents = pool_latents.index_select(0, local_indices)  # float32, no precision loss
                comm_latents[ds_start:ds_end] = new_latents

        gen.train()
        if self.match_space == 'dino':
            # hand the decode/DINO activations back before the training step (it OOM'd without this)
            pool_feats = ds_feats = dists = None
            torch.cuda.empty_cache()

        if all_local_indices:
            # Measure per-class uniqueness: fraction of dataset images that got a unique pool latent.
            # Denominator is n_real (images per class), so 1.0 = every image has a distinct latent.
            self.unique_indices = sum(
                torch.unique(idx).numel() / n_real
                for idx, n_real in all_local_indices
            ) / len(all_local_indices)

        # all_reduce(SUM): each rank only wrote its local_classes slices (zeros elsewhere)
        torch.distributed.all_reduce(comm_latents, op=torch.distributed.ReduceOp.SUM)
        if self.use_rs_imle:
            torch.distributed.all_reduce(rs_stats, op=torch.distributed.ReduceOp.SUM)
            self.rs_reject_pct = 100.0 * rs_stats[0].item() / max(rs_stats[1].item(), 1.0)
            if is_main_process():
                print(f"[rs-imle] epsilon={self.rs_radius} rejected {self.rs_reject_pct:.2f}% of pool")
        full_updated_latents = comm_latents.float()

        perturbation = self.H.imle_perturb_coef * torch.randn(
            (self.sz, self.H.latent_dim), device=self.device,
            generator=self.generator_seed
        )
        full_updated_latents.add_(perturbation)

        self.last_selected_latents.copy_(self.selected_latents)
        self.selected_latents.copy_(full_updated_latents.cpu())

        if self.match_balanced:
            torch.distributed.all_reduce(comm_balanced, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(bal_stats, op=torch.distributed.ReduceOp.SUM)
            comm_balanced.add_(self.H.imle_perturb_coef * torch.randn(
                comm_balanced.shape, device=self.device, generator=self.generator_seed))
            self.balanced_latents = comm_balanced.cpu()
            torch.distributed.all_reduce(comm_moved, op=torch.distributed.ReduceOp.SUM)
            self.balanced_moved = (comm_moved > 0.5).cpu()
            self.balanced_moved_pct = 100.0 * bal_stats[0].item() / max(bal_stats[1].item(), 1.0)
            torch.distributed.all_reduce(comm_bdist, op=torch.distributed.ReduceOp.SUM)
            self.balanced_dist = comm_bdist.cpu()
            d, mv = self.balanced_dist, self.balanced_moved
            qs = torch.tensor([0.1, 0.5, 0.9])
            st = {'own_d10_50_90': [round(x, 2) for x in torch.quantile(d[~mv], qs).tolist()] if (~mv).any() else [],
                  'moved_d10_50_90': [round(x, 2) for x in torch.quantile(d[mv], qs).tolist()] if mv.any() else [],
                  'slot0_d50': round(d[:, 0].median().item(), 2)}
            if self.balanced_rs_radius > 0:
                out = d >= self.balanced_rs_radius
                st['in_zone_pct'] = round(100.0 * (~out).float().mean().item(), 2)
                st['own_in_zone_pct'] = round(100.0 * (~out[~mv]).float().mean().item(), 2) if (~mv).any() else 0.0
                st['moved_in_zone_pct'] = round(100.0 * (~out[mv]).float().mean().item(), 2) if mv.any() else 0.0
                st['active_img_pct'] = round(100.0 * out.any(dim=1).float().mean().item(), 2)
            self.balanced_zone_stats = st
            if is_main_process():
                print(f"[match] balanced: {self.balanced_moved_pct:.2f}% of assigned latents are pulled "
                      f"from a different image than the one they currently copy")
                print(f"[match] slot distances: {self.balanced_zone_stats}")

        if is_main_process():
            print(f"Conditional force resampling took {time.time() - t1:.2f}s")

        self.faiss_index_flat.reset()
