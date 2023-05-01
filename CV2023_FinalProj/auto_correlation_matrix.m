function [auto_rr_matrix_xx, auto_rr_matrix_yy, auto_rr_matrix_xy] = auto_correlation_matrix(image, window_size)
% Compute the auto-correlation matrix of an image using the structure tensor
% method.

    image = uint16(image);

    % Compute the derivatives of the image using Sobel kernels
    Ix = imfilter(image, [-1 0 1; -2 0 2; -1 0 1], 'replicate');
    Iy = imfilter(image, [-1 -2 -1; 0 0 0; 1 2 1], 'replicate');

    % Compute the elements of the structure tensor
    auto_rr_matrix_xx = imgaussfilt(Ix.^2, window_size);
    auto_rr_matrix_yy = imgaussfilt(Iy.^2, window_size);
    auto_rr_matrix_xy = imgaussfilt(Ix.*Iy, window_size);
    auto_rr_matrix_xy = abs(auto_rr_matrix_xy - mean(auto_rr_matrix_xy(:)));

end
