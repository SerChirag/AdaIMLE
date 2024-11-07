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

def generate_video(H, sampler, shape, imle, fname, logprint, lat1=None, lat2=None, sn1=None, sn2=None):
    num_lin = 10
    mb = 180

    f_latent = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda()
    s_latent = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda()
    first_latent = f_latent.clone()

    video_out = imageio.get_writer(fname, mode='I', fps=30, codec='libx264')

    # step = (-f_latent + s_latent) / num_lin
    for t in range(num_lin+1):

        batches = []
        if lat1 is not None:
            print('loading from input')
            f_latent = lat1
            s_latent = lat2
  
        sample_w = torch.cat([slerp(f_latent, s_latent, v) for v in torch.linspace(0, 1, mb).cuda()], dim=0)

        for j in range(0, mb, 8):
            batch_slice = slice(j, j + 8)
            out = imle(sample_w[batch_slice], spatial_noise=None)
            batches.append(sampler.sample_from_out(out))

        # out = imle(sample_w, spatial_noise=snoise, input_is_w=True)
       
        batches = np.concatenate(batches, axis=0)
        # video_out.append_data(batches)
        for image in batches:
            # Append the image to the list of all images
            video_out.append_data(image)
        
        if(t != num_lin - 1):
            f_latent = s_latent.clone()
            s_latent = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda()
        else:
            f_latent = s_latent.clone()
            s_latent = first_latent.cuda()
    
    video_out.close()
