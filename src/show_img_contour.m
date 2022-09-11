% name:       show_img_contour.m
% usage:      --
% author:     Ruicheng
% date:       2020-11-24 14:12:44
% version:    1.0
% Env.:       MATLAB R2019b, WIN10

function fhandle = show_img_contour(img, u, line_color, fhandle)
%myFun - Show segmentation contour
%
% Syntax: fhandle = show_img_contou(img, u, line_color, fhandle)
%
% Show segmentation contour
% new figure handle if not given
    if ~exist('fhandle', 'var')
        fhandle = figure();
        imshow(img, [0, 255]);
    else
        figure(fhandle);
    end

    hold on;
    contour(double(u), [-1, 1], line_color, 'LineWidth', 1.1);
    hold off;

end
