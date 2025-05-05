import torch
import numpy as np
import imageio

def slerp(low, high, val):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def sample_from_out(px_z):
    with torch.no_grad():
        px_z = px_z.permute(0, 2, 3, 1)
        xhat = (px_z + 1.0) * 127.5
        xhat = xhat.detach().cpu().numpy()
        xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
        return xhat


def random_interp(H, sampler, shape, imle, fname, logprint):
    num_lin = 1
    mb = 15
    K = 10000

    device = torch.device("cuda", torch.cuda.current_device())
    batches = []
    # step = (-f_latent + s_latent) / num_lin
    for t in range(num_lin):
        f_latent = torch.randn([1, H.latent_dim], dtype=torch.float32, 
                               device=device, 
                               generator=sampler.generator_seed)
        
        cands = torch.randn([K, H.latent_dim], device=device,
                        generator=sampler.generator_seed)
        # 3) compute Euclid distance to f_latent
        #    expand f_latent to (K, latent_dim) to match
        diffs = cands - f_latent.expand_as(cands)
        dists = diffs.norm(dim=1)               # shape (K,)
        # choose the index of the minimal distance
        best_idx = torch.argmin(dists).item()
        s_latent = cands[best_idx:best_idx+1]
  
        sample_w = torch.cat([slerp(f_latent, s_latent, v) for v in torch.linspace(0, 1, mb, device=device)], dim=0)

        out = imle(sample_w)
       
        batches.append(sample_from_out(out))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


# def random_interp(H, sampler, shape, imle, fname, logprint):
#     num_lin = 1
#     mb = 15

#     device = torch.device("cuda", torch.cuda.current_device())
#     batches = []
#     # step = (-f_latent + s_latent) / num_lin
#     for t in range(num_lin):
#         f_latent = torch.randn([1, H.latent_dim], dtype=torch.float32, 
#                                device=device, 
#                                generator=sampler.generator_seed)
        
#         walk = []

#         new_latent = f_latent.clone()
#         for i in range(mb):
#             z = torch.randn([1, H.latent_dim], dtype=torch.float32,
#                                  device=device, 
#                                  generator=sampler.generator_seed) * 0.1
#             new_latent = new_latent + z
#             new_latent = new_latent / torch.norm(new_latent, dim=1, keepdim=True)
#             new_latent = new_latent * torch.norm(f_latent, dim=1, keepdim=True)
#             walk.append(new_latent)
        
#         walk = torch.cat(walk, dim=0)

#         out = imle(walk)
       
#         batches.append(sample_from_out(out))

#     n_rows = len(batches)
#     im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
#         [n_rows * shape[1], mb * shape[2], 3])

#     logprint(f'printing samples to {fname}')
#     imageio.imwrite(fname, im)



# def random_interp(H, sampler, shape, imle, fname, logprint):
#     num_lin = 1
#     mb = 15

#     device = torch.device("cuda", torch.cuda.current_device())
#     batches = []
#     # step = (-f_latent + s_latent) / num_lin
#     for t in range(num_lin):
#         f_latent = torch.randn([1, H.latent_dim], dtype=torch.float32, 
#                                device=device, 
#                                generator=sampler.generator_seed)
        
#         s_latent = torch.randn([1, H.latent_dim], dtype=torch.float32, 
#                                device=device,
#                                generator=sampler.generator_seed)
  
#         sample_w = torch.cat([slerp(f_latent, s_latent, v) for v in torch.linspace(0, 1, mb, device=device)], dim=0)

#         out = imle(sample_w)
       
#         batches.append(sample_from_out(out))

#     n_rows = len(batches)
#     im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
#         [n_rows * shape[1], mb * shape[2], 3])

#     logprint(f'printing samples to {fname}')
#     imageio.imwrite(fname, im)