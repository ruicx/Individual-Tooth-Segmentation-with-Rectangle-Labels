% name:       demo_generator.m
% usage:      --
% date:       2020-11-24 08:26:41
% version:    1.0
% Env.:       MATLAB R2019b, WIN10

close all
% clearvars -except bboxes_holder
% addpath('...')

fnum = '7';
inum = '279';
fig_path_prefix = ['../figures/', fnum, '_', inum];

% read image
img = imread(['../data/CBCT_PNG/', fnum, '/', inum, '.png']);
img = double(img);

% LIC parameter
param.nu            = .03 * 255.^2; % coefficient of arc length term
param.sigma         = 3;            % scale parameter that specifies the size of the neighborhood
param.iter_outer    = 50;           % outer iteration for level set evolution
param.iter_inner    = 10;           % inner iteration for level set evolution
param.alpha         = 600;          % coefficient of prior term
param.timestep      = .1;           % iteration time step
param.mu            = 1;            % coefficient for distance regularization term (regularize the level set function)
param.beta          = 40;           % coefficient for curvature term
param.epsilon       = 1;            % smooth constant in Heaviside and Dirac function
param.draw_step     = 1;            % stap gap of drawing zero level in segmentation iteration
param.initial_shape = 'rectangle';  % ['rectangle', 'circle', 'random', 'custom']

% other parameter
bluge.RADIUS_THRES  = 10;           % negative radius threshold for finding bottleneck
bluge.SIZE_THRES    = 0.25;         % negative center distance threshold for finding bottleneck
bluge.NEAREST_THRES = 1.3;          % tolerance ratio of point distance for finding bottleneck
bluge.SEG_BLUGE     = 4;            % bluge length and edge length ratio threshold for finding bottleneck
param.bluge         = bluge;
param.oblique       = 'inside';     % oblique of prior ellipse, which in ['regular', 'inside', 'outside']
param.ellipse_ratio = 1.0;          % size ratio for generating ellipse prior

% load detection box
bboxes_holder = DetBboxFcos('../data/fcos_box/fcos_bbox.json', '../data/fcos_box/name_map.json');
bboxes_holder.bbox_threshold = 0.3;
% bboxes_holder = resetDetBbox([...
%     "../data/faster_rcnn_box/train_result_faster_rcnn.json", ...
%     "../data/faster_rcnn_box/val_result_faster_rcnn.json"]);
bbox_block = bboxes_holder.get_cycle_block(fnum, inum, size(img), param.oblique, param.ellipse_ratio);

% show detection box
bboxes_holder.show_box(fnum, inum, img);
print([fig_path_prefix, '_detect.eps'], '-depsc2', '-r600')
title('Detection result')

figure; imshow(img, [0, 255]); hold on;
print([fig_path_prefix, '_img.eps'], '-depsc2', '-r600')

% initial contour
u_init = -initial_level_set(param.initial_shape, size(img));
show_img_contour(img, u_init, 'r');
print([fig_path_prefix, '_initial.eps'], '-depsc2', '-r600')
title('Initial contour')

% initial restriction
prior = double(~bbox_block);
show_img_contour(img, prior, 'r');
print([fig_path_prefix, '_prior.eps'], '-depsc2', '-r600')
title('Prior contour')

% start level set evolution
[u, b] = LIC(img, u_init, prior, param);

% corrected Img
Mask = (img > 10);
Img_corrected = Mask .* img ./ (b + (b == 0));
Img_corrected = normalize01(Img_corrected) * 255;

% fill holes
seg = medfilt2(double(u < 0));
seg = imfill(seg, 'holes');
seg = bwareaopen(seg, 400);
% seg = seg_watershed(seg, bboxes_holder.get_box(fnum, inum));
% seg = edge_curvature(seg, param.bluge);
seg = seg_split(seg, param.bluge);

show_img_contour(img, seg, 'r');
print([fig_path_prefix, '_seg_result.eps'], '-depsc2', '-r600')
title('Segmentation result')

img_draw = draw_contour_in_image(img, seg, [0, 255, 0], 1);

% check save dir
if ~exist('../results', 'dir')
    mkdir('../results')
    mkdir('../results/seg_mask')
    mkdir('../results/seg_params')
    mkdir('../results/seg_contour')
    mkdir('../results/corrected_img')
end

% check save dir
if ~exist(['../results/seg_mask/', fnum], 'dir')
    mkdir(['../results/seg_mask/', fnum])
    mkdir(['../results/seg_params/', fnum])
    mkdir(['../results/seg_contour/', fnum])
    mkdir(['../results/corrected_img/', fnum])
end

if ~exist(['../demo_seg/', fnum], 'dir')
    mkdir(['../demo_seg/', fnum])
end

% save final levelset and papameters
mat_path = ['../results/seg_params/', fnum, '/', inum, '.mat'];
save(mat_path, 'u', 'param', 'seg', 'b', 'Img_corrected')

% save save segmentation contour
contour_path = ['../results/seg_contour/', fnum, '/', inum, '.png'];
imwrite(img_draw, contour_path)

% save segmentation mask
seg_path = ['../results/seg_mask/', fnum, '/', inum, '.png'];
imwrite(seg, seg_path)

% save corrected img
corrected_path = ['../results/corrected_img/', fnum, '/', inum, '.png'];
imwrite(uint8(Img_corrected), corrected_path)

% copy code
copy_code(fnum, inum)

if ismember(fnum, {'4', '6', '7'})
    gt = imread(['../data/semantic_mask/', fnum, '/', inum, '.png']);
    dice(gt > 0, seg > 0)
    figure
    imshowpair(gt > 0, seg > 0)
end
