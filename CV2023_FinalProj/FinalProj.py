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


def auto_correlation_matrix(image, window_size=3):

    image = image.astype(np.uint16)

    # Compute the derivatives of the image using Sobel kernels
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # Compute the elements of the structure tensor
    auto_rr_matrix_xx = cv2.GaussianBlur(Ix**2, (window_size, window_size), 0)
    auto_rr_matrix_yy = cv2.GaussianBlur(Iy**2, (window_size, window_size), 0)
    auto_rr_matrix_xy = cv2.GaussianBlur(Ix*Iy, (window_size, window_size), 0)
    auto_rr_matrix_xy = np.abs(auto_rr_matrix_xy - np.mean(auto_rr_matrix_xy))

    auto_rr_matrix = [auto_rr_matrix_xx, auto_rr_matrix_yy, auto_rr_matrix_xy]

    return auto_rr_matrix


def preprocess_image(image):
    # Convert the image to Lab color space
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Split the Lab image into L, a, and b channels
    # L_channel
    L_channel = cv2.split(lab_img)

    blurnum = 12
    blurlist = [1, 3, 5, 9, 13, 19, 25, 31, 37, 43, 49, 55]

    _img = np.zeros((blurnum, image.shape[0], image.shape[1]))
    for i in range(blurnum):
        _img[i] = cv2.GaussianBlur(L_channel[0], (blurlist[i], blurlist[i]), 0)

    # Compute the auto-correlation matrix
    auto_corr_matrix = np.zeros((blurnum, 3, image.shape[0], image.shape[1]))
    for i in range(blurnum):
        auto_corr_matrix[i] = auto_correlation_matrix(_img[i])

    fig, axs = plt.subplots(3, 4, figsize=(26, 13))
    for i in range(blurnum):
        axs[i//4, i % 4].imshow(auto_corr_matrix[i, 2], cmap='gray')
        axs[i//4, i % 4].set_title('Gaussian Blur %d' % blurlist[i])

    plt.show()

    return _img, auto_corr_matrix


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
    mean_size = features["mean_area"]  # Use 'mean_area' instead of 'areas'
    std_dev_size = np.std(features["mean_area"])

    # Calculate color value statistics
    mean_color = features["mean_color"]

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


def main():
    # Step 1: Image Acquisition
    images = load_images(IMAGE_DIR)

    # Step 2: Image Preprocessing
    preprocessed_images = [preprocess_image(img)[0] for img in images]

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
