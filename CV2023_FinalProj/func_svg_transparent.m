function func_svg_transparent(filename)
    % Read the input SVG file
    file_content = fileread(filename);

    % Replace all occurrences of 'white' with 'none'
    modified_content = strrep(file_content, 'white', 'none');

    % Write the modified content to a new SVG file
    file_id = fopen(filename, 'w');

    if file_id == -1
        error('Could not create the output file');
    end

    fprintf(file_id, '%s', modified_content);
    fclose(file_id);
end
