import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm

# pytorch imports
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# custom imports
from model import NeRF, Embedder

from nerf_helpers import get_rays, nerf_forward, crop_center, render_video
from load_blender import load_blender_data


def prepare_data(datadir, target_size):
    images, poses, render_poses, hwf, i_split = load_blender_data(datadir, target_size=target_size)
    height, width, focal = hwf
    idxs = np.concatenate(i_split)
    return images, poses, height, width, focal, render_poses, idxs



def main(scene_name):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training for {scene_name} on {device}')
    para = yaml.load(Path(f'config/{scene_name}.yaml').read_text(), Loader=yaml.FullLoader)
    images, poses, height, width, focal, render_poses, idxs_train = prepare_data(para['datadir'], para['target_size'])
    render_poses = render_poses.to(device)
    near, far = 2., 6.
    kwargs_sample_stratified = {
        'n_samples': para['n_samples'],
        'perturb': para['perturb'],
    }

    kwargs_sample_hierarchical = {
        'perturb': para['perturb'],
    }


    #### Define embedders
    origin_embedder = Embedder(3, para['n_freqs'], log_space=para['log_space'])
    embed_origin = lambda x: origin_embedder(x)

    if para['use_viewdirs']:
        views_embedder = Embedder(3, para['n_freqs_views'], log_space=para['log_space'])
        embed_view = lambda x: views_embedder(x)
        d_viewdirs = views_embedder.d_output
    else :
        views_embedder = None
        embed_view = None
        d_viewdirs = None

    #### Define NeRF model
    model = NeRF(origin_embedder.d_output, hidden_dim=para['hidden_dim'], n_layers=para['n_layers'], d_viewdirs=d_viewdirs)
    model.to(device)
    model_params = list(model.parameters())

    fine_model = NeRF(origin_embedder.d_output, hidden_dim=para['hidden_dim'], n_layers=para['n_layers_fine'], d_viewdirs=d_viewdirs)
    fine_model.to(device)
    model_params = model_params + list(fine_model.parameters())
    total_params = sum(p.numel() for p in model_params if p.requires_grad)

    #### Optimizer
    optimizer = torch.optim.Adam(model_params, lr=float(para['lr']))
    print(f'Trainable parameters: {total_params/1_000_000:.2f}M')
    
    #### Training loop
    logger = SummaryWriter(os.path.join('runs', para['expname']))
    videos_path = os.path.join('rendered_videos', para['expname'])
    models_path = os.path.join('models', para['expname'])
    os.makedirs(videos_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    pbar = tqdm(range(para['n_iters']))
    best_psnr = -1
    for i in pbar:
        model.train()

        # get a random image
        img_i = np.random.choice(idxs_train)
        target_img = images[img_i]
        target_img = torch.from_numpy(target_img).to(device)
        if para['center_crop'] and i < para['center_crop_iters']:
            target_img = crop_center(target_img)
        height, width = target_img.shape[:2]
        target_pose = torch.from_numpy(poses[img_i]).to(device)
        rays_o, rays_d = get_rays(height, width, focal, target_pose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])

        target_img = target_img.reshape([-1, 3]).float()

        outputs = nerf_forward(rays_o, rays_d,
            near, far, embed_origin, model,
            kwargs_sample_stratified=kwargs_sample_stratified,
            kwargs_sample_hierarchical=kwargs_sample_hierarchical,

            n_samples_hierarchical=para['n_samples_hierarchical'],
            fine_model=fine_model,
            viewdirs_encoding_fn=embed_view,
            chunksize=int(eval(para['chunksize']))
        )

        rgb_predicted = outputs['rgb_map']
        # TASK 5: training loop
        loss = F.mse_loss(rgb_predicted, target_img)  # Calculate MSE loss between predicted and target RGB values
        optimizer.zero_grad()  # Reset gradients to zero to prevent accumulation
        loss.backward()  # Perform backpropagation to calculate gradients
        optimizer.step()  # Update model parameters based on gradients

        psnr = -10. * torch.log10(loss)
        pbar.set_postfix(MSE=loss.item(), PSNR=psnr.item())
        logger.add_scalar("MSE", loss.item(), global_step=i)
        logger.add_scalar("PSNR", psnr.item(), global_step=i)

        if psnr.item() >= best_psnr:
            best_psnr = psnr.item()
            torch.save(model.state_dict(), os.path.join(models_path, 'model.pth'))
            torch.save(fine_model.state_dict(), os.path.join(models_path, 'fine_model.pth'))

            
        if i % para['display_rate'] == 0 or i == para['n_iters'] - 1:
            output_path = os.path.join(videos_path, f'{i}.mp4')
            render_video(render_poses, height, width, focal,
                near, far, embed_origin, model, kwargs_sample_stratified, kwargs_sample_hierarchical, para, fine_model, embed_view, output_path=output_path
            )

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-name', type=str, default='chair')
    args = parser.parse_args()
    main(args.scene_name)
    
    

    
    