import torch
import cv2
import numpy as np
import os

from evaluation import *

###
def apply_temporal_filtering(rgb, filter_type: str, **kwargs):
    """
    Apply temporal filtering to the input RGB image.
    Args:
        rgb: input RGB image as a PyTorch tensor (H, W, 3)
        filter_type: type of filtering to apply ('gaussian', 'bilateral', 'guided', 'median')
        **kwargs: additional arguments for the specific filter
    Returns:
        torch.Tensor: filtered RGB image
    Raises:
        ValueError: if the filter_type is not recognized
    """
    filtering_functions = {
        'gaussian': gaussian_filtering,
        'bilateral': bilateral_filtering,
        'guided': guided_filtering,
        'median': median_filtering
    }
    return filtering_functions[filter_type](rgb, **kwargs)

###
def gaussian_filtering(rgb, **kwargs):
    """
    Apply Gaussian filtering to the image
    Args:
        rgb: input image to be filtered
        **kwargs: additional arguments; for Gaussian filter, it includes: 
            'kernel_size': dimensions of the window used for filtering (default 5)
            'sigma': sigma value for the Gaussian function; controls how much the image is blurred (default 1.0)
    Returns:
        torch.tensor: rgb image with applied filtering
    Raises:
        ValueError: if the input image is not an RGB image with 3 channels
    """   
    if rgb.shape[-1] != 3:
        raise ValueError('Input image must be an RGB image with 3 channels')
        
    kernel_size = kwargs.get('kernel_size', 5)
    sigma = kwargs.get('sigma_color', 1.0)

    if not isinstance(rgb, np.ndarray):
        rgb = rgb.detach().cpu().numpy()
    filtered = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), sigma)
    filtered_rgb_map = torch.tensor(filtered, dtype=torch.float32).to(rgb.device)
    return filtered_rgb_map

###
def bilateral_filtering(rgb, **kwargs):
    """
    Apply bilateral filtering to the image.
    Args:
        rgb: input image to be filtered
        **kwargs: additional arguments; for bilateral filter, it includes:
        'diameter': determines the filter window size (default 7)
        'sigma_color': controls how sensitive the filter is to color differences (default 20)
        'sigma_space': determines how much the spatial distance between pixels should be considered (default 20)
    Returns:
        torch.tensor: rgb image with applied filtering
    Raises:
        ValueError: if the input is not valid
    """   
    if rgb.shape[-1] != 3:
        raise ValueError('Input image must be an RGB image with 3 channels')
    
    diameter = kwargs.get('diameter', 7)
    sigma_color = kwargs.get('sigma_color', 20)
    sigma_space = kwargs.get('sigma_space', 20)

    if not isinstance(rgb, np.ndarray):    
        rgb = rgb.detach().cpu().numpy()    
    filtered_rgb = cv2.bilateralFilter(rgb, diameter, sigma_color, sigma_space)
    filtered_rgb_map = torch.tensor(filtered_rgb, dtype=torch.float32).to(rgb.device)
    return filtered_rgb_map
    
###
def guided_filtering(rgb, **kwargs):
    """
    Apply guided filtering to the input RGB image.
    Args:
        rgb: input image to be filtered
        **kwargs: additional arguments; for guided filter, it includes:
            'radius': local window radius (default 10)
            'eps': regularization strength (default 1e-2)
    Returns:
        torch.Tensor: filtered RGB image, same shape
    Raises:
        ValueError: if the input is not valid
    """
    if rgb.shape[-1] != 3:
        raise ValueError('Input image must be an RGB image with 3 channels')
    
    radius = kwargs.get('radius', 10)
    eps = kwargs.get('eps', 1e-2)

    if not isinstance(rgb, np.ndarray):
        rgb = rgb.detach().cpu().numpy()

    filtered_np = np.zeros_like(rgb)
    for c in range(3):
        filtered_np[..., c] = cv2.ximgproc.guidedFilter(
            guide=rgb[..., c],
            src=rgb[..., c],
            radius=radius,
            eps=eps
        )
    filtered_rgb_map = torch.tensor(filtered_np, dtype=torch.float32).to(rgb.device)
    return filtered_rgb_map
 
### 
def median_filtering(rgb: torch.Tensor, **kwargs):
    """
    Apply median filtering to the input RGB tensor.
    Args:
        rgb: input image to be filtered
        **kwargs: additional arguments; for median filter, it includes:
        'kernel_size' for the median filter (default 3)
    Returns:
        torch.Tensor: filtered RGB image
    Raises:
        ValueError: if the input is not valid
    """
    if rgb.shape[-1] != 3:
        raise ValueError('Input image must be an RGB image with 3 channels')

    kernel_size = kwargs.get('kernel_size', 3)
    if kernel_size % 2 == 0:
        raise ValueError('Kernel size must be an odd number to apply median filtering')
    
    if not isinstance(rgb, np.ndarray):
        rgb = rgb.detach().cpu().numpy()
    pad_size = kernel_size // 2
    padded_rgb = np.pad(rgb, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    filtered_rgb_map = np.zeros_like(rgb)

    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            for c in range(3):
                window = padded_rgb[i:i+kernel_size, j:j+kernel_size, c].flatten()
                filtered_rgb_map[i, j, c] = np.median(window)
    
    filtered_rgb_map = torch.tensor(filtered_rgb_map, dtype=torch.float32).to(rgb.device)
    return filtered_rgb_map

def save_frames(rgbs, disps, save_path, n=3):
    """
    Save RGB and disparity frames to the specified path.
    Args:
        rgbs: list of RGB frames
        disps: list of disparity frames 
        save_path: path to save the frames
        n: number of frames to save (default 3)
    """
    os.makedirs(save_path,exist_ok=True)
    for i in range(min(len(disps), n)):
        plt.imshow(disps[i], cmap='magma')
        plt.colorbar()
        plt.savefig(f'{save_path}/disp_frame_{i}.png')
        plt.close()

        rgb_frame = (rgbs[i] * 225).astype(np.uint8)
        cv2.imwrite(f'{save_path}/rgb_frame_{i}.png', cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

    print(f'Saved {n} frames to {save_path}')