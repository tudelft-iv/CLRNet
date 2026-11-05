import os
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import re

# Set the folder path and output parameters
image_folder = '/home/appuser/data/output/CLR/projections/iter3/lidar'
output_video = '/home/appuser/data/output/CLR/projections/iter3/output_video_lidar.mp4'
fps = 10  # Desired frames per second

def get_image_number(filename):
    match = re.search(r'(\d+)', filename)  # Extract number from filename
    return int(match.group(0)) if match else -1  # Return -1 if no number found

# Function to sort files based on their category and numerical order
def load_and_sort_images(image_folder, category):
    files = [f for f in os.listdir(image_folder) if category in f]
    return sorted(files, key=get_image_number)

# Load and sort images by category
gt_images = load_and_sort_images(image_folder, 'gt')
initial_images = load_and_sort_images(image_folder, 'initial')
predicted_images = load_and_sort_images(image_folder, 'predicted')

assert len(gt_images) == len(initial_images) == len(predicted_images), "Mismatch in the number of images"

# Define a function to concatenate images horizontally
def concatenate_images(image_paths):
    images = [Image.open(os.path.join(image_folder, img_path)) for img_path in image_paths]
    # Find the minimum height among all images to align them
    min_height = min(img.height for img in images)
    resized_images = [img.resize((int(img.width * min_height / img.height), min_height), Image.Resampling.LANCZOS) for img in images]
    # Concatenate images horizontally
    total_width = sum(img.width for img in resized_images)
    concatenated_image = Image.new('RGB', (total_width, min_height))
    x_offset = 0
    for img in resized_images:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width
    return concatenated_image

# Get size of the concatenated images
sample_image = concatenate_images([gt_images[0], initial_images[0], predicted_images[0]])
width, height = sample_image.size

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Loop through all the images and create the video frames with progress bar
for gt, initial, predicted in tqdm(zip(gt_images, initial_images, predicted_images), total=len(gt_images), desc="Creating video"):
    concatenated_img = concatenate_images([gt, initial, predicted])
    # Convert PIL image to OpenCV format
    frame = np.array(concatenated_img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

# Release the video writer
video_writer.release()

print(f"Video saved as {output_video}")