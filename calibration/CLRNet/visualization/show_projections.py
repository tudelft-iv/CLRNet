import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from CLRNet.datasets.DatasetCLR import load_calib
from CLRNet.utils.utils import rotation_vector_from_matrix, invert_pose

# Function to load image-lidar pairs from folders
def load_image_lidar_pairs(image_folder, lidar_folder):
    image_files = sorted(os.listdir(image_folder))
    lidar_files = sorted(os.listdir(lidar_folder))
    pairs = []
    for img_file, lidar_file in zip(image_files, lidar_files):
        if img_file.endswith('.jpg') and lidar_file.endswith('.bin'):
            img_path = os.path.join(image_folder, img_file)
            lidar_path = os.path.join(lidar_folder, lidar_file)
            pairs.append((img_path, lidar_path))
    return pairs

# Function to project lidar points onto image
def project_lidar_to_image(lidar_path, img_path, extrinsic_matrix, intrinsic_matrix):
    # Load lidar points
    lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    # Project lidar points to image
    lidar_points_homogeneous = np.hstack([lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))])
    projected_points = np.dot(extrinsic_matrix, lidar_points_homogeneous.T)
    projected_points = np.dot(intrinsic_matrix, projected_points[:3, :])
    projected_points = projected_points[:2, :] / projected_points[2, :]

    # Load image
    img = cv2.imread(img_path)

    # Draw projected points on image
    for i in range(projected_points.shape[1]):
        x, y = int(projected_points[0, i]), int(projected_points[1, i])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            # Calculate lidar point range
            point_range = np.linalg.norm(lidar_points[i, :3])
            # Map range to colormap
            color_value = min(point_range / 40.0, 1.0)  # Normalize range to 0-1, capped at 40 meters
            # Interpolate color using 'copper' colormap
            color = plt.cm.copper(color_value)[:3]  # Extract RGB values
            color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))  # Convert to BGR
            cv2.circle(img, (x, y), 2, color, -1)

    return img


def project_png_pcd(image, points, K, D, R, T):
    # Project points to image coordinates
    points_2d, _ = cv2.projectPoints(points, R, T, K, D)

    # output_img = np.zeros_like(image)
    output_img = image

    image_height, image_width, _ = image.shape
    point_area_size = 2
    for point in points_2d:
        if int(point[0][0]) <= len(output_img[0]) - 1 and int(point[0][1]) <= len(output_img) - 1 and int(point[0][0]) >= 0 and int(point[0][1]) >= 0:
            x_start = max(0, int(point[0][0]) - point_area_size)
            x_end = min(image_width - 1, int(point[0][0]) + point_area_size)
            y_start = max(0, int(point[0][1]) - point_area_size)
            y_end = min(image_height - 1, int(point[0][1]) + point_area_size)
            output_img[y_start:y_end, x_start:x_end] = (0, 0, 255)

    return output_img

dataset_dir = '/home/appuser/data/view_of_delft_PUBLIC'

# Load intrinsic and extrinsic matrices
calib = load_calib(os.path.join(dataset_dir, 'lidar', 'training', 'calib', '00000.txt')) #Reading 00000.txt since all of them are the same
extrinsic_matrix = calib.T_cam2_velo

K = calib.K_cam2
D = np.zeros(5)
print("Extrinsic Matrix: ", extrinsic_matrix)
print("Intrinsic Matrix: ", K)

rotation_matrix = extrinsic_matrix[:3, :3]
translation_vector = extrinsic_matrix[:3, 3]

R = rotation_vector_from_matrix(rotation_matrix)
T = translation_vector
print("Rotation Vector: ", R)
print("Translation Vector: ", T)
R_inv, T_inv = invert_pose(R, T)

# Folders containing image-lidar pairs
image_folder = os.path.join(dataset_dir, 'lidar', 'training', 'image_2')
lidar_folder = os.path.join(dataset_dir, 'lidar', 'training', 'velodyne')

# Load image-lidar pairs
pairs = load_image_lidar_pairs(image_folder, lidar_folder)
pair_index = 0

while True:
    img_path, lidar_path = pairs[pair_index]
    image = cv2.imread(img_path)
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    # Project lidar points onto image
    projected_img = project_png_pcd(image, points, K, D, R, T)

    # Display the image with projected lidar points
    cv2.imshow("Projected Lidar", projected_img)

    # Wait for key press
    key = cv2.waitKey(0)

    # Check if right or left arrow keys are pressed
    if key == ord('n'):  # 'n' key code
        pair_index = (pair_index + 30) % len(pairs)
    elif key == ord('b'):  # 'b' key code
        pair_index = (pair_index - 30) % len(pairs)
    elif key == 27:  # Escape key code
        break

cv2.destroyAllWindows()
