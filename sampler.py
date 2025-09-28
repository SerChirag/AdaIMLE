import numpy as np
import torch
from torch import autocast

class Sampler:
    def __init__(self):
        
        self.device = torch.device("cuda", torch.cuda.current_device())
        
    def sample(self, latents, gen, snoise=None):
        with torch.no_grad():
            with autocast(device_type='cuda'):
                latents = latents.to(self.device)
                px_z = gen(latents, None)
                px_z = px_z[-1]
                px_z = px_z.permute(0, 2, 3, 1)
                xhat = (px_z + 1.0) * 127.5
                xhat = xhat.detach().cpu().numpy()
                xhat = np.minimum(np.maximum(0.0, xhat), 255.0).astype(np.uint8)
                return xhat
