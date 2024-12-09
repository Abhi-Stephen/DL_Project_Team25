import os
import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, std=0.01):
    noise = np.random.normal(mean, std, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 1)
    return noisy

def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):

    noisy = image.copy()
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 1
   
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    
    return noisy

image_dir = "D:\DL_Dataset_Fall_2024\Town 01\dvs\aligned_timestamps_data_dl_town01_day\images"   
event_dir = "D:\DL_Dataset_Fall_2024\Town 01\dvs\aligned_timestamps_data_dl_town01_day\events"  

output_image_dir = "D:\DL_Dataset_Fall_2024\Town 01\dvs\aligned_timestamps_data_dl_town01_day\Noiseimages"
output_event_dir = "D:\DL_Dataset_Fall_2024\Town 01\dvs\aligned_timestamps_data_dl_town01_day\NoiseEvents"

os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_event_dir, exist_ok=True)

noise_function = add_gaussian_noise 

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = image.astype(np.float32) / 255.0
        noisy_image = noise_function(image)
        noisy_image_uint8 = (noisy_image * 255).astype(np.uint8)
        output_path = os.path.join(output_image_dir, "noisy_" + filename)
        cv2.imwrite(output_path, noisy_image_uint8)

for filename in os.listdir(event_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        event_path = os.path.join(event_dir, filename)
        event_img = cv2.imread(event_path, cv2.IMREAD_COLOR)

        event_img = event_img.astype(np.float32) / 255.0
        noisy_event_img = noise_function(event_img)
        noisy_event_img_uint8 = (noisy_event_img * 255).astype(np.uint8)
        output_path = os.path.join(output_event_dir, "noisy_" + filename)
        cv2.imwrite(output_path, noisy_event_img_uint8)

print("Noise addition completed. Check the output directories for noisy images.")