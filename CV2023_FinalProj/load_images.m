function images = load_images(directory)
    % Load images from the directory
    files = dir(fullfile(directory, '*.jpg'));
    files = [files; dir(fullfile(directory, '*.jpeg'))];
    files = [files; dir(fullfile(directory, '*.png'))];
    images = cell(length(files), 1);

    for i = 1:length(files)
        img_path = fullfile(directory, files(i).name);
        img = imread(img_path);
        images{i} = img;
    end

end
