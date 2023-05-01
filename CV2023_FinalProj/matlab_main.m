close all
IMAGE_DIR = "./testData_1/";
% Step 1: Image Acquisition
images = load_images(IMAGE_DIR);

% Set up figure window

% Loop over each image
for i = 1:length(images)
    % Load the image and convert to grayscale
    fig = figure('Position', [0 0 1200 960]);
    img = images{i};
    grayimg = rgb2gray(img);

    % Apply Gaussian filter to compute auto-correlation matrix
    [auto_rr_xx, auto_rr_yy, auto_rr_xy] = auto_correlation_matrix(grayimg, 3);

    % Define sharpening filter
    %sharpen_filter = [0 -1 0; -1 5 -1; 0 -1 0];
    sharpen_filter = [0 0 0 0 0; 0 0 -1 0 0; 0 -1 5 -1 0; 0 0 -1 0 0; 0 0 0 0 0];

    % Apply sharpening filter to auto-correlation matrices
    auto_rr_xx_sharp = imfilter(auto_rr_xx, sharpen_filter);
    auto_rr_yy_sharp = imfilter(auto_rr_yy, sharpen_filter);
    auto_rr_xy_sharp = imfilter(auto_rr_xy, sharpen_filter);

    fft_rows_xx = zeros(size(grayimg));
    fft_rows_yy = zeros(size(grayimg));
    fft_rows_xy = zeros(size(grayimg));
    fft_rows = zeros(size(grayimg));

    % Compute FFT of each row of the auto-correlation matrix for xx component
    for j = 1:size(grayimg, 1)
        fft_rows_xx(j, :) = fft(auto_rr_xx_sharp(j, :));
        fft_rows_xy(j, :) = fft(auto_rr_xy_sharp(j, :));
    end

    % Compute FFT of each column of the auto-correlation matrix for yy component
    for j = 1:size(grayimg, 2)
        fft_rows_yy(:, j) = fft(auto_rr_yy_sharp(:, j));
    end

    % Compute the average magnitude spectrum across all rows for xx component
    fft_xx = sum(real(fft_rows_xx), 1) / size(grayimg, 1);
    fft_xy = sum(real(fft_rows_xy), 1) / size(grayimg, 1);
    % Compute the average magnitude spectrum across all columns for yy component
    fft_yy = sum(real(fft_rows_yy), 2) / size(grayimg, 2);

    % Smooth the xx component curve
    SML = 11; % smoothing length
    fft_xx_s = smoothdata(fft_xx, 'gaussian', SML);
    fft_xy_s = smoothdata(fft_xy, 'gaussian', SML);
    % Smooth the yy component curve
    fft_yy_s = smoothdata(fft_yy, 'gaussian', SML);

    fft_combined = (fft_xx_s(size(fft_xx_s(:)) - 200:end) / size(grayimg, 1) * size(grayimg, 2) ...
        + fft_yy_s(size(fft_yy_s(:)) - 200:end)' + fft_xy_s(size(fft_xy_s(:)) - 200:end)) / 3;

    % Plot the original image
    subplot(2, 2, 1);
    imshow(img);
    title(sprintf('Image %d', i));

    % Plot the auto-correlation matrix
    subplot(2, 2, 2);
    imshow(auto_rr_xx, []);
    title(sprintf('Auto-correlation Matrix of Image %d', i));

    % Plot the FFT spectra
    fpn = 49; % frequency points number
    GGC = [1:fpn + 1]; % gray gradient compensation
    subplot(2, 2, [3, 4]);
    plot((0:fpn) / 2 + 1, abs(fft_combined(size(fft_combined(:)):-1:size(fft_combined(:)) - fpn)) .* GGC, 'LineWidth', 3);
    hold on;
    plot((0:fpn) / 2 + 1, abs(fft_yy_s(size(fft_yy_s(:)):-1:size(fft_yy_s(:)) - fpn))' .* GGC, 'LineWidth', 0.5);
    plot((0:fpn) / 2 + 1, abs(fft_xy_s(size(fft_xy_s(:)):-1:size(fft_xy_s(:)) - fpn)) .* GGC, 'LineWidth', 0.5);
    plot((0:fpn) / 2 + 1, abs(fft_xx_s(size(fft_xx_s(:)):-1:size(fft_xx_s(:)) - fpn)) .* GGC, 'LineWidth', 0.5);

    hold off;
    ylim([0 1e6]);
    xlim([1 25]);

    title(sprintf('FFT Spectrum of Image %d', i));
    legend('combined fft', 'yy fft', 'xy fft', 'xx fft', 'Location', 'NorthEast', 'FontSize', 16);

    fig_name = sprintf('./figure/Res_%d.svg', i);
    print(fig, fig_name, '-dsvg');
    func_svg_transparent(fig_name);
    close(fig.Number);
end
