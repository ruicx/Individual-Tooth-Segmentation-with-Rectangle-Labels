% name:       draw_contour_in_image.m
% usage:      --
% date:       2020-11-29 16:29:58
% version:    1.0
% Env.:       MATLAB R2019b, WIN10


function img_draw = draw_contour_in_image(img, u, line_color, line_width)
%draw_contour_in_image - Draw the zero level contour in image matrix
%
% Syntax: img_draw = draw_contour_in_image(img, u, line_color, line_width)
%
% Draw the zero level contour in image matrix
    mask = edge(u);

    if line_width > 1
        se = strel('sphere', line_width);
        mask = imdilate(mask, se);
    end

    if length(size(img)) == 2
        img_draw = repmat(img, 1, 1, 3);
    else
        img_draw = img;
    end

    img_r = img_draw(:, :, 1);
    img_r(mask > 0) = line_color(1);
    img_g = img_draw(:, :, 2);
    img_g(mask > 0) = line_color(2);
    img_b = img_draw(:, :, 3);
    img_b(mask > 0) = line_color(3);
    img_draw = cat(3, img_r, img_g, img_b);
    img_draw = uint8(img_draw);

end