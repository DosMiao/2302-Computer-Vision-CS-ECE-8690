import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import cv2
from CVSpring2023_Miao_Xuanbo_Assignment2_function import *
current_time = datetime.now().strftime("%H:%M:%S")

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
print("Current Time =", current_time)


img_orig = cv2.imread('./Claude_Monet,_Impression,_soleil_levant,_1872.jpg')

img_raw_b, img_raw_g, img_raw_r  = cv2.split(img_orig)

img_grad_b, img_grad_b_x, img_grad_b_y=gradient_calc(img_raw_b)
img_grad_g, img_grad_g_x, img_grad_g_y=gradient_calc(img_raw_g)
img_grad_r, img_grad_r_x, img_grad_r_y=gradient_calc(img_raw_r)

color_structure_tensor  =compute_color_structure_tensor(img_orig).astype(np.float32)
color_structure_tensor_trace=np.zeros(color_structure_tensor.shape[0:2])
color_structure_tensor_trace=np.trace(color_structure_tensor,axis1=2,axis2=3)

color_structure_tensor_maxvalue=np.max(np.abs(color_structure_tensor))

color_structure_tensor_00=np.uint8(np.abs(color_structure_tensor[:,:,0,0]/color_structure_tensor_maxvalue*255))
color_structure_tensor_01=np.uint8(np.abs(color_structure_tensor[:,:,0,1]/color_structure_tensor_maxvalue*255))
color_structure_tensor_10=np.uint8(np.abs(color_structure_tensor[:,:,1,0]/color_structure_tensor_maxvalue*255))
color_structure_tensor_11=np.uint8(np.abs(color_structure_tensor[:,:,1,1]/color_structure_tensor_maxvalue*255))
color_structure_tensor_trace=np.uint8(255*color_structure_tensor_trace/np.max(color_structure_tensor_trace))

color_structure_tensor_trace_cannied=cv2.Canny(color_structure_tensor_trace,20,80)

img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
img_gray_grad,img_gray_grad_x,img_gray_grad_y=gradient_calc(img_gray)

image_datalist=[ img_orig , img_grad_b , img_grad_b_x , img_grad_b_y , img_grad_g , img_grad_g_x , img_grad_g_y , img_grad_r , img_grad_r_x , img_grad_r_y , color_structure_tensor_00 , color_structure_tensor_01 , color_structure_tensor_10 , color_structure_tensor_11 , color_structure_tensor_trace , color_structure_tensor_trace_cannied , img_gray , img_gray_grad , img_gray_grad_x , img_gray_grad_y ]
image_namelist=['img_orig','img_grad_b','img_grad_b_x','img_grad_b_y','img_grad_g','img_grad_g_x','img_grad_g_y','img_grad_r','img_grad_r_x','img_grad_r_y','color_structure_tensor_00','color_structure_tensor_01','color_structure_tensor_10','color_structure_tensor_11','color_structure_tensor_trace', 'color_structure_tensor_trace_cannied','img_gray','img_gray_grad','img_gray_grad_x','img_gray_grad_y']

if 1:
    for image_id in range(len(image_datalist)):
        cv2.imshow(image_namelist[image_id], image_datalist[image_id]);  
        #cv2.resizeWindow(image_namelist[image_id], 808, 1051); 
        cv2.imwrite('./img/'+image_namelist[image_id]+'.jpg',image_datalist[image_id])

cv2.waitKey(0)

