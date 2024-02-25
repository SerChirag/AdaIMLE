import torch
import numpy as np
import imageio

def random_interp(H, sampler, shape, imle, fname, logprint, lat1=None, lat2=None, sn1=None, sn2=None):
    num_lin = 1
    mb = 8

    batches = []
    # step = (-f_latent + s_latent) / num_lin
    for t in range(num_lin):
        f_latent = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda()
        s_latent = torch.randn([1, H.latent_dim], dtype=torch.float32).cuda()

        if lat1 is not None:
            print('loading from input')
            f_latent = lat1
            s_latent = lat2

        f_latent = imle.module.decoder.mapping_network(f_latent)[0]
        s_latent = imle.module.decoder.mapping_network(s_latent)[0]
        sample_w = torch.cat([torch.lerp(f_latent, s_latent, v) for v in torch.linspace(0, 1, mb).cuda()], dim=0)


        out = imle(sample_w, None, input_is_w=True)
        batches.append(sampler.sample_from_out(out))

    n_rows = len(batches)
    im = np.concatenate(batches, axis=0).reshape((n_rows, mb, *shape[1:])).transpose([0, 2, 1, 3, 4]).reshape(
        [n_rows * shape[1], mb * shape[2], 3])

    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)