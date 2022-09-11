% name:       initial_level_set.m
% usage:      --
% date:       2020-11-24 08:37:12
% version:    1.0
% Env.:       MATLAB R2019b, WIN10


function initialLSF = initial_level_set(initial_shape, img_shape)
%initial_level_set - initial level set
%
% Syntax: initialLSF = initial_level_set(initial_shape)
%
% initial level set
    nrow = img_shape(1);
    ncol = img_shape(2);
    switch initial_shape
        case 'rectangle'
            initialLSF = -2 * ones([nrow, ncol]);
            initialLSF(50:nrow - 50, 50:ncol - 50) = 2;
        case 'circle'
            ic = nrow / 2;
            jc = ncol / 2;
            r = 100;
            initialLSF = sdf2circle(nrow, ncol, ic, jc, r);
        case 'random'
            initialLSF = rand([nrow, ncol]);
        otherwise
            error("most given initial_shape from ['rectangle', 'circle', 'random']")
    end
end


function f = sdf2circle(nrow, ncol, ic, jc, r)
%   sdf2circle(nrow,ncol, ic,jc,r) computes the signed distance to a circle
%   input:
%       nrow: number of rows
%       ncol: number of columns
%       (ic,jc): center of the circle
%       r: radius of the circle
%   output:
%       f: signed distance to the circle
%
%   created on 04/26/2004
%   author: Chunming Li
%   email: li_chunming@hotmail.com
%   Copyright (c) 2004-2006 by Chunming Li

    [X, Y] = meshgrid(1:ncol, 1:nrow);

    f = sqrt((X - jc).^2 + (Y - ic).^2) - r;
end