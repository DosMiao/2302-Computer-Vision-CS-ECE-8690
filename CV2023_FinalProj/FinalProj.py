import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants
IMAGE_DIR = "./CV2023_FinalProj/testData_1/"


def load_images(directory):
    # Load images from the directory
    images = []
    for file in os.listdir(directory):
        if file.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(directory, file)
            img = cv2.imread(img_path)
            images.append(img)
    return images


def unify_contrast_brightness(image):
    # Compute the minimum and maximum pixel values in the image
    min_val, max_val, _, _ = cv2.minMaxLoc(image)

    # Normalize the pixel values to span the full range of 0 to 255
    if max_val > min_val:
        normalized = ((image - min_val) / (max_val - min_val)) * 255
    else:
        normalized = image.copy()

    # Convert the pixel values to the range of 0 to 255
    normalized = np.uint8(normalized)

    # Apply a histogram equalization to improve contrast
    equalized = cv2.equalizeHist(normalized)   # type: ignore

    return equalized


def unify_image(img, clip_limit=2.0, tile_size=8, brightness=128, contrast=1.0, detail_level=1.0):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)

    # Apply brightness and contrast adjustments to the L channel
    l = cv2.addWeighted(l, contrast, np.zeros_like(l), 0, brightness - 128)

    # Apply detail level adjustment to the L channel
    l = cv2.GaussianBlur(l, (0, 0), sigmaX=(1 - detail_level) * 20 + 1)

    # Merge channels and convert back to BGR color space
    lab = cv2.merge((l, a, b))
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return out


def auto_correlation_matrix(image, window_size=3):

    # Compute the derivatives of the image using Sobel kernels
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # Compute the elements of the structure tensor
    auto_rr_matrix_xx = cv2.GaussianBlur(Ix**2, (window_size, window_size), 0)
    auto_rr_matrix_yy = cv2.GaussianBlur(Iy**2, (window_size, window_size), 0)
    auto_rr_matrix_xy = cv2.GaussianBlur(Ix*Iy, (window_size, window_size), 0)

    auto_rr_matrix = [auto_rr_matrix_xx, auto_rr_matrix_yy, auto_rr_matrix_xy]

    return auto_rr_matrix


def preprocess_image(image):
    # Convert the image to Lab color space
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split the Lab image into L, a, and b channels
    L_channel, _, _ = cv2.split(lab_img)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(L_channel, (3, 3), 0)

    # Apply contrast and brightness enhancement
    enhanced_image = unify_contrast_brightness(blurred_img)

    # Apply CLAHE, brightness, and contrast adjustments
    unified_image = unify_image(enhanced_image)

    # Compute the auto-correlation matrix
    auto_corr_matrix = auto_correlation_matrix(unified_image)

    return unified_image, auto_corr_matrix


def DetectAndCompute(image, max_points):

    sift = cv2.SIFT_create()
    keypoints = sift.detect(image)
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    keypoints = keypoints[:max_points]

    keypoints, descriptors = sift.compute(image, keypoints)
    descriptors = descriptors[:max_points]

    return keypoints, descriptors


def extract_features(image):
    max_points = 1000
    keypoints, descriptors = DetectAndCompute(image, max_points)

    # Calculate the average area of keypoints
    areas = [kp.size for kp in keypoints]
    mean_area = np.mean(areas)

    # Calculate the average color values of keypoints
    colors = [image[int(kp.pt[1]), int(kp.pt[0])] for kp in keypoints]
    mean_color = np.mean(colors, axis=0)

    features = {
        'mean_area': mean_area,
        'mean_color': mean_color,
        'descriptors_size': descriptors.shape[0]
    }

    return features


def analyze_features(features):
    # Analyze the features to determine particle size, distribution, and color value

    # Calculate particle size statistics
    particle_sizes = features["areas"]
    mean_size = np.mean(particle_sizes)
    std_dev_size = np.std(particle_sizes)

    # Calculate color value statistics
    colors = features["colors"]
    mean_color = features["avg_color"]

    # Create a dictionary to store the results
    results = {
        "mean_size": mean_size,
        "std_dev_size": std_dev_size,
        "mean_color": mean_color,
    }

    return results


def display_results(results):
    # Display the results in a user interface

    # Print results to the console (simple example)
    print(f"Mean Particle Size: {results['mean_size']:.2f}")
    print(
        f"Standard Deviation of Particle Size: {results['std_dev_size']:.2f}")
    print(f"Mean Color Value (B, G, R): {tuple(results['mean_color'])}")

    # Alternatively, you can create a more advanced user interface using a library like Tkinter or PyQt.


def main():
    # Step 1: Image Acquisition
    images = load_images(IMAGE_DIR)

    # Step 2: Image Preprocessing
    preprocessed_images = [preprocess_image(img) for img in images]

    # Step 3: Feature Extraction
    feature_list = [extract_features(img) for img in preprocessed_images]

    # Step 4: Data Analysis
    analysis_results = [analyze_features(features)
                        for features in feature_list]

    # Step 5: Display the results
    for idx, results in enumerate(analysis_results):
        print(f"Results for Image {idx + 1}:")
        display_results(results)
        print()

    # Step 6: Testing and Validation
    # Compare the system's results to manual measurements
    pass


if __name__ == "__main__":
    main()
