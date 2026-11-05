import cv2
import numpy as np
import os
from CLRNet.utils.utils import lidar_project_depth_spherical
import torch

def enlarge_points(depth_image, enlarge_factor):
    """
    Enlarges non-black points on the depth image by drawing circles.

    Parameters:
        depth_image (np.ndarray): The depth image (grayscale).
        enlarge_factor (int): Radius of the circles to be drawn around non-black points.

    Returns:
        np.ndarray: The modified depth image with enlarged points.
    """
    # Ensure the image is in grayscale format
    if len(depth_image.shape) != 2:
        raise ValueError("Input depth_image must be a grayscale image.")

    # Create a copy of the image to draw the circles
    enlarged_image = depth_image.copy()

    # Get coordinates of all non-black pixels
    non_black_pixels = np.column_stack(np.where(depth_image > 0))

    # Draw a circle at each non-black pixel with the original pixel color
    for (y, x) in non_black_pixels:
        pixel_color = int(depth_image[y, x])  # Get the original pixel intensity
        cv2.circle(enlarged_image, (x, y), enlarge_factor, (pixel_color,), -1)  # Fill the circle with the pixel color

    return enlarged_image

def display_depth_image(depth_img_tensor):
    # Assume depth_img_tensor has shape (C, H, W)
    # Extract the depth channel (first channel)
    depth_channel = depth_img_tensor[0, :, :].cpu().numpy()

    # Normalize the depth values to the range 0-255 for visualization
    min_depth = depth_channel.min()
    max_depth = depth_channel.max()
    depth_normalized = ((depth_channel - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)

    # Enlarge the non-black points for better visualization
    enlarged_depth_img = enlarge_points(depth_normalized, enlarge_factor=3)

    # Display the image using OpenCV
    cv2.imshow('Depth Image', enlarged_depth_img)
    cv2.waitKey(0)

def parse_point_cloud(scan, dim=4):
    pc = scan.reshape((-1, dim))
    #correct wrong x,y,z calculations from VoD for Radar
    if dim == 7:
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        alpha = np.arctan2(y, x)
        cos_alpha = np.cos(alpha)

        # Handle potential division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            safe_cos_alpha = np.where(cos_alpha == 0, np.nan, cos_alpha)
            z = z / safe_cos_alpha
            z = np.nan_to_num(z, nan=np.inf)

        pc[:, 2] = z
    #filter out points closer than 2m
    valid_indices = pc[:, 0] < -2.
    valid_indices = valid_indices | (pc[:, 0] > 2.)
    valid_indices = valid_indices | (pc[:, 1] < -2.)
    valid_indices = valid_indices | (pc[:, 1] > 2.)
    pc = pc[valid_indices].copy()
    pc_org = torch.from_numpy(pc.astype(np.float32))

    pc_org = pc_org.t()

    return pc_org

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device", i, ":", torch.cuda.get_device_name(i))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset_dir = '/home/appuser/data/KITTI_Second_TIER_05_29/uvod'
lidar_folder = os.path.join(dataset_dir, 'lidar', 'training', 'velodyne')

lidar_files = sorted(os.listdir(lidar_folder))
for lidar_file in lidar_files:
    lidar_path = os.path.join(lidar_folder, lidar_file)
    cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    points = parse_point_cloud(cloud, 4)
    depth_img, _ = lidar_project_depth_spherical(points, [1024,2048])

    display_depth_image(depth_img)
