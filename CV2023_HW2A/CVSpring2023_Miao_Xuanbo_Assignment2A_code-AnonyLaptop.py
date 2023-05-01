import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.signal import argrelmax
import pandas as pd

marker_color = [0, 0, 255]


def get_current_time():
    current_time = datetime.now().strftime("%H:%M:%S")
    return current_time

folder = "./wall/"
#folder = "./bikes/"
image_id1 = "img2.ppm"
image_id2 = "img4.ppm"
image_id3 = "img4.ppm"

print(get_current_time())


def read_ppm_file(file_path):
    with open(file_path, 'rb') as f:
        # Read the header information of the PPM file
        header = f.readline().decode('utf-8').strip()
        width, height = map(int, f.readline().decode('utf-8').split())
        max_val = int(f.readline().decode('utf-8'))

        # Read the image data from the PPM file
        img_data = f.read()

        # Convert the image data to a NumPy array
        img_array = bytearray(img_data)
        img_np = np.array(img_array)

        # Reshape the NumPy array into a 3D array with dimensions (height, width, 3)
        img_np = img_np.reshape((height, width, 3))

        # Convert the NumPy array to a cv2 object in BGR format
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        return img_cv2
    
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
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
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


def gaussian_smooth(image, sigma=2):
    # Create a 1D Gaussian kernel
    kernel_size = 2 * round(3 * sigma) + 1
    kernel = cv2.getGaussianKernel(kernel_size, sigma)

    # Compute the 2D Gaussian kernel
    kernel = np.outer(kernel, kernel.transpose())

    # Normalize the kernel to sum to 1
    kernel /= np.sum(kernel)

    # Apply the Gaussian filter to the input image
    smoothed_image = cv2.filter2D(image, -1, kernel)

    return smoothed_image


def computeFeatureMeasure(I, k=0.04, window_size=3):

    auto_rr_matrix = auto_correlation_matrix(I, window_size)
    [xx, yy, xy] = auto_rr_matrix
    xx = gaussian_smooth(xx)
    yy = gaussian_smooth(yy)
    xy = gaussian_smooth(xy)
    # Compute the Harris response
    det = xx * yy - xy ** 2
    trace = xx + yy
    R = (det+1)/ (trace+1)
    R /= np.mean(R)
    #print(np.max(abs(R)))
    return R

def FeatureMeasure2Points(R, npoints,threshold = 0.01):

    # Find the locations of local maxima above the threshold
    h, w = R.shape
    mask = np.zeros((h, w), np.uint8)
    mask[R > threshold] = 1

    # Use morphological dilation to find the connected regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_erosion1 = cv2.erode(mask,          kernel,iterations = 2)
    mask_dilate1 = cv2.dilate(mask_erosion1, kernel,iterations = 2)
    mask_erosion2 = cv2.erode(mask_dilate1,  kernel,iterations = 5)
    mask_dilate2 = cv2.dilate(mask_erosion2, kernel,iterations = 2)
    mask_erosion3 = cv2.erode(mask_dilate2,  kernel,iterations = 5)
    mask_dilate3 = cv2.dilate(mask_erosion3, kernel,iterations = 2)
    mask_erosion4 = cv2.erode(mask_dilate3,  kernel,iterations = 5)
    mask_dilate4 = cv2.dilate(mask_erosion4, kernel,iterations = 2)

    #cv2.imshow('imgh',mask_dilate1*250)
    # Compute the centroid of each connected region as the feature point location
    num_labels, labels, stats1, centroids1 = cv2.connectedComponentsWithStats(mask_dilate1)
    num_labels, labels, stats2, centroids2 = cv2.connectedComponentsWithStats(mask_dilate2)
    num_labels, labels, stats3, centroids3 = cv2.connectedComponentsWithStats(mask_dilate3)
    num_labels, labels, stats4, centroids4 = cv2.connectedComponentsWithStats(mask_dilate4)

    # Remove the background label
    centroids_=np.concatenate((centroids1[1:],centroids2[1:],centroids3[1:],centroids4[1:]), axis=0)
    stats_=np.concatenate((stats1[1:],stats2[1:],stats3[1:],stats4[1:]), axis=0)
    stats_=stats_[:,4]
    # Sample a fixed number of feature points from the list of all feature points
    centroids=[]
    for idx in range(len(stats_)):
        if 100 < stats_[idx] < 500:
            centroids=np.concatenate((centroids, centroids_[idx]), axis=0)
    centroids=np.reshape(centroids, (int(len(centroids)/2),2))
    
    if len(centroids) > npoints:
        centroids = centroids[:npoints]

    # Return the x, y coordinates of the feature points and the mask of the detected regions
    x = centroids[:, 0]
    y = centroids[:, 1]
    mask = np.zeros_like(R)
    for point in centroids:
        mask[int(point[1]), int(point[0])] = 1

    return x, y, mask

def extract_bordered_patch(img, x, y, w, h, border):

    bordered_img = cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_CONSTANT)

    # Adjust the patch coordinates to account for the border
    x += border
    y += border

    # Extract the patch from the bordered image
    patch = bordered_img[int(y-h//2):int(y+h//2)+1,int( x-w//2):int(x+w//2)+1]

    # Handle the case where the patch exceeds the boundary
    patch_height, patch_width = patch.shape[:2]
    if patch_height != h or patch_width != w:
        # The patch exceeds the boundary
        # Crop the patch to the available size
        patch = patch[:h+1, :w+1]

    return patch


def generateFeatureDescriptors(I, x, y, patch_size=21):
    # Define a square patch around each feature point
    half_patch_size = patch_size // 2
    num_points = len(x)
    Dlist = np.zeros((num_points, patch_size ** 2), dtype=np.float32)
    for i in range(num_points):
        patch=extract_bordered_patch(I, x[i], y[i], patch_size,patch_size , border=patch_size)
        norm_patch = (patch - np.mean(patch)) / np.std(patch)
        norm_kernel = np.ones(norm_patch.shape) / norm_patch.size
        inc_patch = cv2.filter2D(norm_patch, -1, norm_kernel)
        inc_descriptor=inc_patch.flatten()
        inc_descriptor /= np.linalg.norm(inc_descriptor)
        Dlist[i, :] = inc_descriptor

    return Dlist


def show_descriptor_image(image, x, y, Dlist, patch_size=21):
    # Draw a square patch around each feature point and display the image
    half_patch_size = patch_size // 2
    num_points = len(x)
    for i in range(num_points):
        cv2.rectangle(image, (int(x[i]-half_patch_size), int(y[i]-half_patch_size)),
                      (int(x[i]+half_patch_size), int(y[i]+half_patch_size)), marker_color, 1)
    #cv2.imshow('Patch', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return image


def transform_point(H1to2, x, y):
    point1 = np.array([x, y, 1])
    point2 = np.dot(H1to2, point1)
    point2_cartesian = point2 / point2[2]
    u, v = point2_cartesian[:2]

    return u, v


def eval_TFPN_s(kp1, kp2, selected_matches, all_matches, trans_matrix, judge_threshhold=16):
    all_num = max(len(kp1), len(kp2))
    selected_num = len(selected_matches)
    P, N, TP, FN, FP, TN = [0, 0, 0, 0, 0, 0]

    for single_match in all_matches:
        try:
            single_match = single_match[0]
        except:
            single_match = single_match
        pos_kp1 = kp1[single_match.queryIdx].pt
        pos_kp1_ = transform_point(trans_matrix, pos_kp1[0], pos_kp1[1])
        pos_kp2 = kp2[single_match.trainIdx].pt
        distance = np.linalg.norm(np.array(pos_kp2)-np.array(pos_kp1_))  # type: ignore
        if distance < judge_threshhold:
            P += 1
    N=all_num-P

    TP_matches=[]
    for single_match in selected_matches:
        try:
            single_match = single_match[0]
        except:
            single_match = single_match
        pos_kp1 = kp1[single_match.queryIdx].pt
        pos_kp1_ = transform_point(trans_matrix, pos_kp1[0], pos_kp1[1])
        pos_kp2 = kp2[single_match.trainIdx].pt
        distance = np.linalg.norm(np.array(pos_kp2)-np.array(pos_kp1_))  # type: ignore
        single_match=[pos_kp1,pos_kp2]
        if distance < judge_threshhold:
             TP+= 1
             TP_matches.append(single_match)

    FP=selected_num-TP
    FN = P-TP
    TN = N-FP
    TPR = TP/(max((TP+FN), 1))
    FPR = FP/(max((FP+TN), 1))
    PPV = TP/(max((TP+FP), 1))
    ACC = (TP+TN)/(P+N)
    
    result_array_d=[P, N, TP, FN, FP, TN, TPR, FPR, PPV, ACC]
    return result_array_d, TP_matches

def eval_TFPN(kp1_x, kp1_y, kp2_x, kp2_y,Dist,Matchlist, trans_matrix, judge_threshhold=16):
    all_num = kp1_x.size*kp2_x.size
    selected_num = len(Matchlist)
    P, N, TP, FN, FP, TN = [0, 0, 0, 0, 0, 0]
    
    for i in range(Dist.shape[0]):
        for j in range(Dist.shape[1]):
            pos_kp1=[kp1_x[i],kp1_y[i]]
            pos_kp1_ = transform_point(trans_matrix, pos_kp1[0], pos_kp1[1])
            pos_kp2=[kp2_x[j],kp2_y[j]]
            distance = np.linalg.norm(np.array(pos_kp2)-np.array(pos_kp1_)) # type: ignore
            if distance < judge_threshhold:
                P += 1
    P=int(P/2) 
    N=all_num-P

    TP_matches=[]
    for queryIdx,trainIdx in Matchlist:
        pos_kp1=[kp1_x[queryIdx],kp1_y[queryIdx]]
        pos_kp1_ = transform_point(trans_matrix, pos_kp1[0], pos_kp1[1])
        pos_kp2=[kp2_x[trainIdx],kp2_y[trainIdx]]
        distance = np.linalg.norm(np.array(pos_kp2)-np.array(pos_kp1_))  # type: ignore
        if distance < judge_threshhold:
            TP += 1
            single_match=[[kp1_x[queryIdx],kp1_y[queryIdx]],[kp2_x[trainIdx],kp2_y[trainIdx]]]
            TP_matches.append(single_match)

    TP=int(TP/2) 
    FP=selected_num-TP
    FN = P-TP
    TN = N-FP
    TPR = TP/(max((TP+FN), 1))
    FPR = FP/(max((FP+TN), 1))
    PPV = TP/(max((TP+FP), 1))
    ACC = (TP+TN)/(P+N)

    result_array_d=[P, N, TP, FN, FP, TN, TPR, FPR, PPV, ACC]
    return result_array_d, TP_matches


def print_Rarray(arrayname, number_array):
    str_array = arrayname+': '
    variable_names = ['P', 'N', 'TP', 'FN','FP', 'TN', 'TPR', 'FPR', 'PPV', 'ACC']
    for i in range(0, 6):
        str_array += variable_names[i]+': '+str(number_array[i])+', '
    for i in range(6, 10):
        str_array += variable_names[i]+': ' + \
            '{:.3f}'.format(number_array[i])+', '
    str_array += "\n"
    return str_array


def DetectAndCompute(image, max_points):

    sift = cv2.SIFT_create()
    keypoints = sift.detect(image)
    keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
    keypoints = keypoints[:max_points]

    keypoints, descriptors = sift.compute(image, keypoints)
    descriptors = descriptors[:max_points]

    return keypoints, descriptors

def evaluate_INC(descriptor1, descriptor2):
    
    eps = 1e-6 # A small number to prevent division by zero
    mag1 = np.linalg.norm(descriptor1) + eps 
    mag2 = np.linalg.norm(descriptor2) + eps 
    dot_product = np.dot(descriptor1, descriptor2) 
    inc = 1.0 - dot_product / (mag1 * mag2) 
    if dot_product>1:
        dot_product=dot_product
    return inc

def computeDescriptorDistances(Dlist1, Dlist2):
    Dist = np.zeros((Dlist1.shape[0], Dlist2.shape[0]))
    for i in range(Dlist1.shape[0]):
        for j in range(Dlist2.shape[0]):
            #dist[i, j] = np.sqrt(np.sum((Dlist1[i] - Dlist2[j])**2))
            Dist[i, j] = evaluate_INC(Dlist1[i], Dlist2[j])

    return Dist


def computeDescriptorMatchs_s(Dlist1, Dlist2):
    sift = cv2.SIFT_create()
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    Matchs = matcher.knnMatch(Dlist1, Dlist2, k=2)
    return Matchs

def draw_correspondences(image1, image2, points,color1,color2):

    # Convert points to integers
    points = np.round(points).astype(int)
 
    # Draw points on images
    radius = 4
    thickness = -1
    for p in points:
        image1 = cv2.circle(image1, tuple(p[0]), radius, color1, thickness)
        image2 = cv2.circle(image2, tuple(p[1]), radius, color1, thickness)
    # Draw line connecting points

    thickness = 1
    image = np.concatenate((image1, image2), axis=1)
    for p in points:
        image = cv2.line(image, tuple(p[0]), tuple(p[1] + [image1.shape[1], 0]), color2, thickness)

    return image

def post_Distance2Matches_s(MatchList, All_MatchList, kp1, kp2, img_o, img_t, trans_matrix):

    result_array_d, TP_matches = eval_TFPN_s(kp1, kp2, MatchList, All_MatchList, trans_matrix)

    img_matches = cv2.drawMatches(img_o, kp1, img_t, kp2, TP_matches, None,  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                                        #matchColor=marker_color,
    return result_array_d, img_matches


def Distance2Matches_DistThresh(Dist, Th1=100):
    # Distance threshhold
    # Apply distance thresholding and select good matches
    MatchList1 = []
    for i in range(Dist.shape[0]):
        for j in range(Dist.shape[1]):
            if Dist[i, j] < Th1:
                MatchList1.append([i, j])

    return MatchList1


def Distance2Matches_NearestMatch(Dist, Th2=0.6):
    # Nearest match
    # Sort matches by distance and select top x
    Dlist = []
    for i in range(Dist.shape[0]):
        for j in range(Dist.shape[1]):
            Dlist.append([Dist[i, j], i, j])
    Dlist = sorted(Dlist, key=lambda x: x[0])
    MatchList2_ = Dlist[:int(len(Dlist)*Th2/len(Dist))]

    MatchList2 = []
    for m in MatchList2_:
        MatchList2.append(m[1:3])

    return MatchList2


def Distance2Matches_NearestRatio(Dist, Th3=0.7):
    # NNDR

    DD = []
    for i in range(Dist.shape[0]):
        Dlist = []
        for j in range(Dist.shape[1]):
            Dlist.append([Dist[i, j], i, j])
        Dlist = sorted(Dlist, key=lambda x: x[0])
        DD.append(Dlist[:2])

    MatchList3 = []
    for DDD in DD:
        if DDD[0][0] < Th3 * DDD[1][0]:
            MatchList3.append(DDD[0][1:3])

    return MatchList3


def Distance2Matches_DistThresh_s(AllMatchList, threshold_match_d=100):
    # Distance threshhold
    # Apply distance thresholding and select good matches
    MatchList1 = []
    for m in AllMatchList:
        if m[0].distance < threshold_match_d:
            MatchList1.append(m[0])

    return MatchList1


def Distance2Matches_NearestMatch_s(AllMatchList, threshold_match_n=0.6):
    # Nearest match
    # Sort matches by distance and select top x
    AllMatchList = sorted(AllMatchList, key=lambda x: x[0].distance)
    MatchList2_ = AllMatchList[:int(len(AllMatchList)*threshold_match_n)]
    MatchList2 = []
    for m in MatchList2_:
        MatchList2.append(m[0])

    return MatchList2


def Distance2Matches_NearestRatio_s(AllMatchList, threshold_match_nn=0.7):
    # NNDR
    MatchList3 = []
    for m, n in AllMatchList:
        if m.distance < threshold_match_nn * n.distance:
            MatchList3.append(m)

    return MatchList3


# Load images
img_o = read_ppm_file(folder+image_id1)
img_o2 = read_ppm_file(folder+image_id2)
img_o3 = read_ppm_file(folder+image_id3)

gray_o_ = cv2.cvtColor(img_o, cv2.COLOR_BGR2GRAY)
gray_o3_ = cv2.cvtColor(img_o3, cv2.COLOR_BGR2GRAY)
# Load homography matrices
H1to2p = np.loadtxt(folder+'H1to2p'); H1to3p = np.loadtxt(folder+'H1to3p'); H1to4p = np.loadtxt(folder+'H1to4p'); H1to5p = np.loadtxt(folder+'H1to5p'); H1to6p = np.loadtxt(folder+'H1to6p')

HH_loop = [(H1to2p, 'H2p_'), (H1to3p, 'H3p_'), (H1to4p, 'H4p_'), (H1to5p, 'H5p_'), (H1to6p, 'H6p_')]
for TMatrix, Hid in HH_loop:
    # Apply transformation matrix to image
    sub_folder = folder+Hid
    print(sub_folder)
    img_t = cv2.warpPerspective(img_o2, TMatrix, (img_o2.shape[1], img_o2.shape[0]))
    gray_t_ = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
    gray_o = unify_contrast_brightness(gray_o_)
    gray_o3 = unify_contrast_brightness(gray_o3_)
    gray_t = unify_contrast_brightness(gray_t_)

    # Display original and transformed images
    cv2.imwrite(sub_folder+'0101_Original_image'+'.jpg', gray_o3_)
    cv2.imwrite(sub_folder+'0102_Transformed'+'.jpg', img_t)

    cv2.imwrite(sub_folder+'0201_Original_image gray_o'+'.jpg', gray_o3)
    cv2.imwrite(sub_folder+'0202_Transformed gray_t'+'.jpg', gray_t)
    
    # personal

    R_img_o = computeFeatureMeasure(gray_o)
    R_img_t = computeFeatureMeasure(gray_t)

    num_points = 800
    #x_img_o, y_img_o, mask_img_o = FeatureMeasure2Points(R_img_o, num_points,0.5)
    #x_img_t, y_img_t, mask_img_t = FeatureMeasure2Points(R_img_t, num_points,0.5)
    #num_points=min(len(x_img_o),len(x_img_t))
    x_img_o, y_img_o, mask_img_o = FeatureMeasure2Points(R_img_o, num_points,0.8)
    x_img_t, y_img_t, mask_img_t = FeatureMeasure2Points(R_img_t, num_points,0.8)
    
    img_o_save1 = img_o3.copy();     img_t_save1 = img_t.copy()

    for i in range(len(x_img_o)):
        cv2.circle(img_o_save1, (int(x_img_o[i]), int(y_img_o[i])), 3, marker_color, -1)

    for i in range(len(x_img_t)):
        cv2.circle(img_t_save1, (int(x_img_t[i]), int(y_img_t[i])), 3, marker_color, -1)

    cv2.imwrite(sub_folder+'0301_Feature_O'+'.jpg', img_o_save1)
    cv2.imwrite(sub_folder+'0302_Feature_T'+'.jpg', img_t_save1)

    patch_size = 21

    Dlist_img_o = generateFeatureDescriptors(gray_o, x_img_o, y_img_o, patch_size)   
    img_o_save2 = show_descriptor_image(img_o_save1.copy(), x_img_o, y_img_o, Dlist_img_o, patch_size)

    Dlist_img_t = generateFeatureDescriptors(gray_t, x_img_t, y_img_t, patch_size)     
    img_t_save2 = show_descriptor_image(img_t_save1.copy(), x_img_t, y_img_t, Dlist_img_t, patch_size)

    cv2.imwrite(sub_folder+'0401_Descriptors_O' + '.jpg', img_o_save2)
    cv2.imwrite(sub_folder+'0402_Descriptors_T' + '.jpg', img_t_save2)

    Dist = computeDescriptorDistances(Dlist_img_o, Dlist_img_t)

    threshold_match_d = 3.5;    threshold_match_n = 10.2;    threshold_match_nn = 0.8

    threshold_match_d=np.mean(np.amin(Dist, axis=1))*threshold_match_d
    MatchList1 = Distance2Matches_DistThresh(Dist, threshold_match_d)
    MatchList2 = Distance2Matches_NearestMatch(Dist, threshold_match_n)
    MatchList3 = Distance2Matches_NearestRatio(Dist, threshold_match_nn)

    result_array_d, TP_matches_d   =eval_TFPN(x_img_o, y_img_o, x_img_t, y_img_t, Dist, MatchList1, TMatrix)
    result_array_n, TP_matches_n   =eval_TFPN(x_img_o, y_img_o, x_img_t, y_img_t, Dist, MatchList2, TMatrix)
    result_array_nn, TP_matches_nn =eval_TFPN(x_img_o, y_img_o, x_img_t, y_img_t, Dist, MatchList3, TMatrix)
    
    result_image_d  =draw_correspondences(img_o_save1.copy(),img_t_save1.copy(),TP_matches_d, (0,0,255),(0,255,0))
    result_image_n  =draw_correspondences(img_o_save1.copy(),img_t_save1.copy(),TP_matches_n, (0,0,255),(0,255,0))
    result_image_nn =draw_correspondences(img_o_save1.copy(),img_t_save1.copy(),TP_matches_nn,(0,0,255),(0,255,0))

    cv2.imwrite(sub_folder+'0501_Matches_d' + '.jpg', result_image_d)
    cv2.imwrite(sub_folder+'0502_Matches_n' + '.jpg', result_image_n)
    cv2.imwrite(sub_folder+'0503_Matches_nn' + '.jpg', result_image_nn)

    string1 = print_Rarray("FixDist", result_array_d)
    string2 = print_Rarray("Nearest", result_array_n)
    string3 = print_Rarray("N N D R", result_array_nn)
    print(string1+string2+string3)
    
    # sift
    kp1, Dlist1 = DetectAndCompute(gray_o, 500)
    kp2, Dlist2 = DetectAndCompute(gray_t, 500)

    All_Matchs = computeDescriptorMatchs_s(Dlist1, Dlist2)

    threshold_match_d = 120;    threshold_match_n = 0.6;    threshold_match_nn = 0.7
    MatchList1_s = Distance2Matches_DistThresh_s(All_Matchs, threshold_match_d)
    MatchList2_s = Distance2Matches_NearestMatch_s(All_Matchs, threshold_match_n)
    MatchList3_s = Distance2Matches_NearestRatio_s(All_Matchs, threshold_match_nn)

    result_array_s_d , TP_matches_s_d  = eval_TFPN_s(kp1, kp2, MatchList1_s, All_Matchs, TMatrix)
    result_array_s_n , TP_matches_s_n  = eval_TFPN_s(kp1, kp2, MatchList2_s, All_Matchs, TMatrix)
    result_array_s_nn, TP_matches_s_nn = eval_TFPN_s(kp1, kp2, MatchList3_s, All_Matchs, TMatrix)

    result_image_s_d  =draw_correspondences(img_o3.copy(),img_t.copy(),TP_matches_s_d, (0,255,0),(0,0,255))
    result_image_s_n  =draw_correspondences(img_o3.copy(),img_t.copy(),TP_matches_s_n, (0,255,0),(0,0,255))
    result_image_s_nn =draw_correspondences(img_o3.copy(),img_t.copy(),TP_matches_s_nn,(0,255,0),(0,0,255))

    cv2.imwrite(sub_folder+'0601_Sift_Matches_d'+'.jpg', result_image_s_d)
    cv2.imwrite(sub_folder+'0602_Sift_Matches_n'+'.jpg', result_image_s_n)
    cv2.imwrite(sub_folder+'0603_Sift_Matches_nn'+'.jpg', result_image_s_nn)

    string1 = print_Rarray("FixDist_s", result_array_s_d)
    string2 = print_Rarray("Nearest_s", result_array_s_n)
    string3 = print_Rarray("N N D R_s", result_array_s_nn)
    print(string1+string2+string3)

    Result=[result_array_d,result_array_n,result_array_nn,result_array_s_d,result_array_s_n,result_array_s_nn]
    df = pd.DataFrame(Result,columns=['P', 'N', 'TP', 'FN', 'FP', 'TN', 'TPR', 'FPR', 'PPV', 'ACC'],
                      index=[ 'FixDist', 'Nearest', 'N N D R', 'FixDistS', 'NearestS', 'N N D RS'])

    df.to_excel(sub_folder+'result.xlsx')

    
