from functools import partial
import os
import argparse
import yaml
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
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
parser.add_argument('--noiselevel', type=float, default=0.05)
parser.add_argument('--random_seed', type=int, default=123)

args = parser.parse_args()
if args.device == 'cuda':
    args.device = f'cuda:{args.gpu}'

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming the image is normalized to [0, 1]
    psnr = 20 * np.log10(max_pixel / (mse**0.5))
    return psnr.item()

def denorm_to_01(x):
    return torch.clamp((x + 1) / 2, 0, 1)

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

# Infer working resolution from task config (falls back to 256x256).
op_in_shape = measure_config['operator'].get('in_shape', None)
if op_in_shape is not None and len(op_in_shape) >= 4:
    img_h, img_w = int(op_in_shape[-2]), int(op_in_shape[-1])
else:
    img_h, img_w = 256, 256


# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning


# Load diffusion sampler
sampler = create_sampler(**diffusion_config) 
sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)


# Prepare dataloader
data_config = task_config['data']
if args.file_path:                        # ← 加这行
    data_config['root'] = args.file_path  # ← 加这行
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
if measure_config['operator']['name'] == 'super_resolution':
    # Keep original input size for SR unless you explicitly resize in dataset/files.
    transform = transforms.Compose([transforms.ToTensor()])
else:
    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor()
    ])
dataset = get_dataset(**data_config, transforms=transform)
loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
dataset_fpaths = getattr(dataset, 'fpaths', None)

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
    mask_gen = mask_generator(
       **measure_config['mask_opt']
    )
# Or define mask
mask = torch.ones(1, 3, img_h, img_w)
mask[:, :, int(70 * img_h / 256):int(200 * img_h / 256), int(70 * img_w / 256):int(190 * img_w / 256)] = 0
mask = mask.to(args.device)
scheduler = DDIMScheduler()


def mask_A(image,mask):
    return (image*mask).to(args.device)

# our sampler method
def optimize_input(input,  sqrt_one_minus_alpha_cumprod, sqrt_alpha_cumprod, t, num_steps, learning_rate, mask=None):
    input_tensor = torch.randn(1, model.in_channels, img_h, img_w, requires_grad=True)
    input_tensor.data = input.clone().to(args.device)
    optimizer = torch.optim.Adam([input_tensor], lr=learning_rate)
    tt = (torch.ones(1) * t).to(args.device)
    for step in range(num_steps):
        optimizer.zero_grad()
       
        noise_pred = model(input_tensor.to(args.device), tt)
        noise_pred = noise_pred[:, :3]
        pred_x0 = (input_tensor.to(args.device) -sqrt_one_minus_alpha_cumprod * noise_pred) / sqrt_alpha_cumprod
        pred_x0= torch.clamp(pred_x0, -1, 1)
        if mask is not None:
            out = operator.forward(pred_x0, mask=mask)
        else:
            out = operator.forward(pred_x0)
        loss = torch.norm(out-y_n)**2
        height, width = y.shape[-2], y.shape[-1]
        product = height * width *3
        if loss < (args.noiselevel + 0.001)**2 * torch.tensor(product, dtype=torch.float32):
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



times =[]
for image_idx, ref_img in enumerate(loader):
    psnrs = []
    best_img = []
    best_img.append(None)
    ref_img = ref_img.to(dtype).to(args.device)
    ref_img = ref_img * 2 - 1

    # For SR, always follow each image's original resolution.
    if measure_config['operator']['name'] == 'super_resolution':
        cur_h, cur_w = int(ref_img.shape[-2]), int(ref_img.shape[-1])
        if (cur_h, cur_w) != (img_h, img_w):
            img_h, img_w = cur_h, cur_w
            measure_config['operator']['in_shape'] = (1, 3, img_h, img_w)
            operator = get_operator(device=args.device, **measure_config['operator'])
            cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
        if image_idx == 0:
            print(f"Super-resolution running at original size: {img_h}x{img_w}")

    # if measure_config['operator'] ['name'] == 'inpainting':
    #     mask = mask
    #     measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
    #     sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

    #     # Forward measurement model (Ax + n)
    #     y = operator.forward(ref_img, mask=mask)
    #     y_n = noiser(y)
    if measure_config['operator']['name'] == 'inpainting':
        mask = mask_gen(ref_img).to(args.device)   # ✅ 每张图动态生成 mask，不再用硬编码 box
        measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
        # ✅ 删掉 sample_fn 那行，不需要
        y = operator.forward(ref_img, mask=mask)
        y_n = noiser(y)

    else: 
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)


    # start reverse sampling with pretrain diffusion model
    input = torch.randn((1, 3, img_h, img_w), device=args.device, dtype=dtype)
    noise = torch.randn(input.shape)*((1-scheduler.alphas_cumprod[-1])**0.5)
    input = input.clone().detach()*((scheduler.alphas_cumprod[-1])**0.5) + noise.to(args.device)
    start_time = time.time() 
    for step_idx, t in enumerate(scheduler.timesteps):
            prev_timestep = t - step_size
            #print(prev_timestep.dtype)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.alphas_cumprod[0]

            beta_prod_t = 1 - alpha_prod_t
            sqrt_one_minus_alpha_cumprod = beta_prod_t**0.5

            for k in range(1):
                input, pred_original_sample = optimize_input(
                    input.clone(),
                    sqrt_one_minus_alpha_cumprod,
                    alpha_prod_t**0.5,
                    t,
                    num_steps=args.num_steps,
                    learning_rate=args.learning_rate,
                    mask=mask if measure_config['operator']['name'] == 'inpainting' else None,
                )
                # ✅ 加入 PS 引导，让 mask 区域也有梯度信号
                if measure_config['operator']['name'] == 'inpainting':
                    with torch.enable_grad():
                        ps_x0 = pred_original_sample.detach().requires_grad_(True)
                        y_hat = operator.forward(ps_x0, mask=mask)
                        ps_loss = torch.norm(y_hat - y_n) ** 2
                        ps_grad = torch.autograd.grad(ps_loss, ps_x0)[0]
                    scale = cond_config['params']['scale']
                    pred_original_sample = pred_original_sample - scale * ps_grad.detach()
                    pred_original_sample = torch.clamp(pred_original_sample, -1, 1)            
                input= pred_original_sample * alpha_prod_t**0.5+(1-alpha_prod_t)**0.5*torch.randn(input.size()).to(args.device)
            input = pred_original_sample * alpha_prod_t_prev**0.5+(1-alpha_prod_t_prev)**0.5*torch.randn(input.size()).to(args.device)
    
            
    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time  # Calculate elapsed time
    times.append(elapsed_time)  # Store the time
    print(f"Processing time for image {image_idx + 1}: {elapsed_time:.2f} seconds")

    psnr_value =np.max(psnrs)
    print(f"After diffusion PSNR: {psnr_value} dB")
    if best_img[0] is not None:
        out = torch.from_numpy(best_img[0]).permute(2, 0, 1).unsqueeze(0).float().to(args.device)
    else:
        out = denorm_to_01(pred_original_sample)

    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        save_file = os.path.join(args.save_path, f"result_image_{image_idx}.png")
        save_image(out, save_file)

        # Save input/measurement for side-by-side inspection.
        input_file = os.path.join(args.save_path, f"input_image_{image_idx}.png")
        save_image(denorm_to_01(ref_img), input_file)

        if measure_config['operator']['name'] == 'super_resolution':
            meas_vis = operator.transpose(y_n).clamp(-1, 1)
            meas_lr_file = os.path.join(args.save_path, f"measurement_lr_{image_idx}.png")
            save_image(torch.clamp((y_n + 1) / 2, 0, 1), meas_lr_file)
        else:
            meas_vis = y_n.clamp(-1, 1)
        meas_file = os.path.join(args.save_path, f"measurement_image_{image_idx}.png")
        save_image(denorm_to_01(meas_vis), meas_file)

        panel = torch.cat([denorm_to_01(ref_img), denorm_to_01(meas_vis), out], dim=3)
        panel_file = os.path.join(args.save_path, f"compare_panel_{image_idx}.png")
        save_image(panel, panel_file)

        # Save original source resolution when dataset path is available.
        if dataset_fpaths is not None and image_idx < len(dataset_fpaths):
            src_file = os.path.join(args.save_path, f"input_original_{image_idx}.png")
            Image.open(dataset_fpaths[image_idx]).convert('RGB').save(src_file)


