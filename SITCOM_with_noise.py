from functools import partial
import os
import argparse
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion_correct import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from common_utils import *
from ddim_sampler import *
import shutil
import lpips
import time
parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--diffusion_config', type=str)
parser.add_argument('--task_config', type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--file_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--num_steps', type=int, default=30)
parser.add_argument('--n_step', type=int, default=20)
parser.add_argument('--threshold', type=int, default=30)
parser.add_argument('--random_seed', type=int, default=123)

args = parser.parse_args()

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming the image is normalized to [0, 1]
    psnr = 20 * np.log10(max_pixel / (mse**0.5))
    return psnr.item()

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# Load configurations
model_config = load_yaml(args.model_config)
diffusion_config = load_yaml(args.diffusion_config)
task_config = load_yaml(args.task_config)

#Load model
model = create_model(**model_config)
model = model.to(args.device)
model.eval()

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=args.device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])


# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning


# Load diffusion sampler
sampler = create_sampler(**diffusion_config) 
sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)


# Prepare dataloader
data_config = task_config['data']
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
    mask_gen = mask_generator(
       **measure_config['mask_opt']
    )
# Or define mask
mask = torch.ones(1,3,256,256)
mask[:,:,70:200,70:190]=0  
mask = mask.to(args.device)
scheduler = DDIMScheduler()


def mask_A(image,mask):
    return (image*mask).to(device)

# our sampler method
def optimize_input(input,  sqrt_one_minus_alpha_cumprod, sqrt_alpha_cumprod, t, num_steps, learning_rate):
    input_tensor = torch.randn(1, model.in_channels, 256, 256, requires_grad=True)
    input_tensor.data = input.clone().to(args.device)
    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)
    tt = (torch.ones(1) * t).to(args.device)
    for step in range(num_steps):
        optimizer.zero_grad()
       
        noise_pred = model(input_tensor.to(args.device), tt)
        noise_pred = noise_pred[:, :3]
        pred_x0 = (input_tensor.to(args.device) -sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
        pred_x0= torch.clamp(pred_x0, -1, 1)
        out =operator.forward(pred_x0)
        loss = torch.norm(out-y_n)**2
        if loss < args.threshold*torch.sqrt(torch.tensor(len(y), dtype=torch.float32)):
            break   
        loss.backward(retain_graph=True)    
        optimizer.step()
    with torch.no_grad():
        output_numpy = pred_x0.detach().cpu().squeeze().numpy()
        output_numpy = (output_numpy/2+0.5)#.clamp(0, 1)
        output_numpy = np.transpose(output_numpy, (1, 2, 0))
        # calculate psnr
        ref_numpy = (ref_img/2+0.5)
        ref_numpy = np.array(ref_numpy.cpu().detach().numpy()[0].transpose(1,2,0))
        tmp_psnr = compute_psnr(ref_numpy, output_numpy)
        psnrs.append(tmp_psnr)

    if len(psnrs) == 1 or (len(psnrs) > 1 and tmp_psnr > np.max(psnrs[:-1])):
        best_img[0] = output_numpy
    return input_tensor.detach(), pred_x0.detach()


# define the sampler step
out = []
n_step = args.n_step
scheduler.set_timesteps(num_inference_steps=n_step)
step_size = 1000//n_step

dtype = torch.float32


psnrs = []
times =[]
for i, ref_img in enumerate(loader):
    best_img = []
    best_img.append(None)
    ref_img = ref_img.to(dtype).to(args.device)
    if measure_config['operator'] ['name'] == 'inpainting':
        mask = mask
        measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
        sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img, mask=mask)
        y_n = noiser(y)

    else: 
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)
    y_n.requires_grad = False


    # start reverse sampling with pretrain diffusion model
    input =torch.randn((1, 3, 256, 256), device=args.device, dtype=dtype)
    noise = torch.randn(input.shape)*((1-scheduler.alphas_cumprod[-1])**0.5)
    input = torch.tensor(input)*((scheduler.alphas_cumprod[-1])**0.5) + noise.to(args.device)
    start_time = time.time() 
    for i, t in enumerate(scheduler.timesteps):
            prev_timestep = t - step_size
            #print(prev_timestep.dtype)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.alphas_cumprod[0]

            beta_prod_t = 1 - alpha_prod_t
            sqrt_one_minus_alpha_cumprod = beta_prod_t**0.5

            for k in range(1):
                input, pred_original_sample = optimize_input(input.clone(), sqrt_one_minus_alpha_cumprod, alpha_prod_t**0.5, t, num_steps=args.num_steps, learning_rate=args.learning_rate)
                input= pred_original_sample * alpha_prod_t**0.5+(1-alpha_prod_t)**0.5*torch.randn(input.size()).to(args.device)
            input = pred_original_sample * alpha_prod_t_prev**0.5+(1-alpha_prod_t_prev)**0.5*torch.randn(input.size()).to(args.device)
    
            
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    times.append(elapsed_time)  # Store the time
    print(f"Processing time for image {i + 1}: {elapsed_time:.2f} seconds")

    psnr_value =np.max(psnrs)
    print(f"After diffusion PSNR: {psnr_value} dB")
    out = (pred_original_sample + 1) / 2
    out_image = out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


