import torch
from PIL import Image
import dnnlib
import legacy
from diffusers import UNet2DModel, DDPMPipeline, DDPMScheduler, DDIMScheduler

class SyntheticIrisGenerator(object):
    def __init__(
        self, 
        gan_net_path='./models/network-snapshot-025000.pkl',
        diffusion_net_path='./models/checkpoint-epoch-50', 
        device=None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gan_net_path = gan_net_path
        
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        with dnnlib.util.open_url(self.gan_net_path) as f:
            # Load and set explicitly to evaluation mode
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device).eval()

        unet = UNet2DModel.from_pretrained(
            diffusion_net_path, 
            torch_dtype=self.dtype
        ).to(self.device)
        
        scheduler = DDPMScheduler.from_pretrained(diffusion_net_path)
        self.pipeline = DDPMPipeline(unet=unet, scheduler=scheduler).to(self.device)
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)

    @torch.inference_mode()
    def generate_image(self, model_type='diffusion'):
        if model_type == 'gan':
            z = torch.randn(1, self.G.z_dim, device=self.device, dtype=torch.float32)
            img = self.G(z, None, truncation_psi=0.5, noise_mode='random')
                
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return Image.fromarray(img[0][0].cpu().numpy(), 'L')
            
        elif model_type == 'diffusion':
            img = self.pipeline(
                batch_size=1, 
                num_inference_steps=150
            ).images[0]
            return img
            
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Choose 'gan' or 'diffusion'.")