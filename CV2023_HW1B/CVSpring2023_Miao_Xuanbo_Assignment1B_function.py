import cv2
import numpy as np
kernel_scale=11

def gradient_calc(img_data):
    sobelx = cv2.Sobel(img_data, cv2.CV_64F, 1, 0, ksize=kernel_scale)
    sobely = cv2.Sobel(img_data, cv2.CV_64F, 0, 1, ksize=kernel_scale)
    img_grad = np.sqrt(sobelx**2 + sobely**2)
    sobelx= np.uint8(255*abs(sobelx)/np.max(np.abs(sobelx)))
    sobely= np.uint8(255*abs(sobely)/np.max(np.abs(sobely)))
    img_grad= np.uint8(255*img_grad/np.max(img_grad))
    
    return img_grad, sobelx, sobely



def compute_color_structure_tensor(img):
    rows, cols, channels = img.shape

    # Convert the image to a float type
    img = img.astype(np.float32)

    # Compute the gradient in x and y directions
    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=kernel_scale)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=kernel_scale)

    # Initialize the 2D Color Structure Tensor
    color_structure_tensor_colored = np.zeros((rows, cols, 3, 2, 2))

    # Loop over the color channels (R, G, B)
    for c in range(channels):
        # Compute the elements of the 2D Color Structure Tensor for this color channel
        color_structure_tensor_colored[..., c, 0, 0] = sobel_x[..., c] ** 2
        color_structure_tensor_colored[..., c, 0, 1] = sobel_x[..., c] * sobel_y[..., c]
        color_structure_tensor_colored[..., c, 1, 0] = color_structure_tensor_colored[..., c, 0, 1]
        color_structure_tensor_colored[..., c, 1, 1] = sobel_y[..., c] ** 2
    color_structure_tensor=np.sum(color_structure_tensor_colored,axis=2)

    return color_structure_tensor

def compute_and_display_color_structure_tensor(img):
    color_structure_tensor = compute_color_structure_tensor(img)

    # Compute the eigenvalues and eigenvectors of the 2D Color Structure Tensor
    eigenvalues, eigenvectors = np.linalg.eig(color_structure_tensor[..., 0, :, :])

    # Normalize the eigenvectors
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=1, keepdims=True)

    # Plot the eigenvectors on top of the input image
    img_with_eigenvectors = np.uint8(img.copy())
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            x2 = int(x + eigenvectors[y, x, 0] * 100)
            y2 = int(y + eigenvectors[y, x, 1] * 100)
            cv2.line(img_with_eigenvectors, (x, y), (x2, y2), (0, 0, 255), 1)

    cv2.imshow("2D Color Structure Tensor", img_with_eigenvectors)
    cv2.waitKey(0)