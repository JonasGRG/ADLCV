import torch
import torch.nn as nn

import cv2
import numpy as np
from tqdm import tqdm


def im_normalize(im):
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def normalize(im):
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def tens2image(im):
    tmp = np.squeeze(im.numpy())
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0))
    

def calculate_mids(distances):
    mids = torch.zeros((distances.shape[0]-1), device=distances.device)
    for i in range(len(distances) - 1):
        mids[i] = (distances[i] + distances[i + 1]) / 2

    return mids


def sample_stratified(
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    perturb=True,
    ):
    """
    The stratified sampling approach splits the ray into evenly-spaced bins and randomly samples within each bin
    Arguments
    ---------
    rays_o      :   Represents the origin point(s) of the ray(s) in 3D space.
    rays_d      :   Represents the direction vector(s) of the ray(s) in 3D space.
    near        :   near depth of the ray (how close to camera)
    far         :   far depth of the ray (how far from the camera).
    n_samples   :   The number of samples to generate along the ray.
    """


    # Sample linearly between `near` and `far`
    distances = torch.linspace(near, far, n_samples, device=rays_o.device)    
    
    #Draw uniform samples from bins along ray
    if perturb:
        mids = calculate_mids(distances)
        far = torch.concat([mids, distances[-1].unsqueeze(0)], dim=-1)
        near = torch.concat([distances[1].unsqueeze(0), mids], dim=-1)
        
        t_rand = torch.rand([n_samples], device=distances.device) # random numbers between 0 and 1
        distances = near + (far - near) * t_rand
        
    distances = distances.expand(list(rays_o.shape[:-1]) + [n_samples])

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * distances[..., :, None]
    return pts, distances



def sample_pdf(bins, weights, n_samples, perturb=False):
    """
    Apply inverse transform sampling to a weighted set of points.
    """

    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous() # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]


def crop_center(img,  frac=0.5):
    """
    Crop center square from image.
    """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]



def sample_hierarchical(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    z_vals: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    perturb=False,
    ):
    """
    Apply hierarchical sampling to the rays.
    """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples, perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, new_z_samples


def get_rays(
    height: int,
    width: int,
    focal_length: float,
    c2w: torch.Tensor # pose or transformation matrix
    ):
    """
    Find origin and direction of rays through every pixel and camera origin.
    """
    device = c2w.device
    # Apply pinhole camera model to gather directions at each pixel
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing='ij')
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack([
        (i - width * .5) / focal_length,
        -(j - height * .5) / focal_length,
        -torch.ones_like(i, device=device)
        ], 
    dim=-1)

    # Apply camera pose to directions
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

    # Origin is same for all directions (the optical center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def cumprod_exclusive(tensor):
    """
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
    is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
    tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod



def raw2outputs(raw, z_vals, rays_d, raw_noise_std = 0.0, white_bkgd = True):
    """
    Convert the raw NeRF output into RGB and other maps.
    """

    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the near subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                                depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


def get_chunks(inputs, chunksize=2**15):
    """
    Divide an input into chunks.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def prepare_chunks(points,  encoding_function, chunksize: int = 2**15):
    """
    Encode and chunkify points to prepare for NeRF model.
    """
    points = points.reshape((-1, 3))
    points = encoding_function(points)
    points = get_chunks(points, chunksize=chunksize)
    return points


def prepare_viewdirs_chunks(points, rays_d, encoding_function, chunksize: int = 2**15, constant_viewdir=None):
    """
    Encode and chunkify viewdirs to prepare for NeRF model.
    """
    # Prepare the viewdirs
    if constant_viewdir is not None:
        viewdirs = torch.ones_like(rays_d) * constant_viewdir
    else :
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
    viewdirs = encoding_function(viewdirs)
    
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs


def nerf_forward(
    rays_o,
    rays_d,
    near,
    far,
    encoding_fn,
    coarse_model,
    kwargs_sample_stratified=None,
    n_samples_hierarchical=0,
    kwargs_sample_hierarchical=None,
    fine_model=None,
    viewdirs_encoding_fn=None,
    chunksize = 2**15,
    constant_viewdir=None
    ):
    """
    Compute forward pass through model(s).
    """

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, near, far, **kwargs_sample_stratified)
    
    # Prepare batches.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
        batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                viewdirs_encoding_fn,
                                                chunksize=chunksize, constant_viewdir=constant_viewdir)
    else:
        batches_viewdirs = [None] * len(batches)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        if batch_viewdirs is not None:
            batch_viewdirs = batch_viewdirs.float()
        predictions.append(coarse_model(batch.float(), viewdirs=batch_viewdirs))

    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {
        'z_vals_stratified': z_vals
    }

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(
            rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
            **kwargs_sample_hierarchical)

        # Prepare inputs as before.
        batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
        if viewdirs_encoding_fn is not None:
            batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                        viewdirs_encoding_fn,
                                                        chunksize=chunksize, constant_viewdir=constant_viewdir)
    else:
        batches_viewdirs = [None] * len(batches)

    # Forward pass new samples through fine model.
    fine_model = fine_model if fine_model is not None else coarse_model
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
        if batch_viewdirs is not None:
            batch_viewdirs = batch_viewdirs.float()

        predictions.append(fine_model(batch.float(), viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)

    # Store outputs.
    outputs['z_vals_hierarchical'] = z_hierarch
    outputs['rgb_map_0'] = rgb_map_0
    outputs['depth_map_0'] = depth_map_0
    outputs['acc_map_0'] = acc_map_0

    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    return outputs


@torch.no_grad()
def render_video(render_poses, height, width, focal,
        near, far, embed_origin, model, kwargs_sample_stratified, kwargs_sample_hierarchical, para, fine_model, embed_view,
        output_path='rendered_video.mp4', pbar=False, return_frames=False, constant_viewdir=None):
    # Load the NeRF model and other necessary parameters here if needed.
    model.eval()
    fine_model.eval()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 5, (height, width))

    if pbar :
        r = tqdm(range(render_poses.shape[0]), desc='Rendering')
    else : 
        r = range(render_poses.shape[0])
        
    # Render the scene at the current pose
    if return_frames:
        pred_frames = []
    for frame in r:
        pose = render_poses[frame]
        rays_o, rays_d = get_rays(height, width, focal, pose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])

        outputs = nerf_forward(rays_o, rays_d,
            near, far, embed_origin, model,
            kwargs_sample_stratified=kwargs_sample_stratified,
            kwargs_sample_hierarchical=kwargs_sample_hierarchical,

            n_samples_hierarchical=int(para['n_samples_hierarchical']),
            fine_model=fine_model,
            viewdirs_encoding_fn=embed_view,
            chunksize=int(eval(para['chunksize'])),
            constant_viewdir=constant_viewdir
        )

        rgb_predicted = outputs['rgb_map']
        rendered_frame = rgb_predicted.view(height, width, 3).cpu().detach().numpy()
        if return_frames:
            pred_frames.append(rendered_frame)

        img_uint8 = (rendered_frame * 255).astype(np.uint8)
        out.write(cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    out.release()


    model.train()
    fine_model.train()

    if return_frames:
        return pred_frames
