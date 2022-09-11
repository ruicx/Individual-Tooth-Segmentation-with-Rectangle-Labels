% name:       seg_split.m
% usage:      --
% author:     Ruicheng
% date:       2021-01-10 10:44:26
% version:    1.0
% Env.:       MATLAB R2019b, WIN10


function clear_seg = seg_split(seg, BLUGE_THRES, debug_flag)
%seg_split - Clear region segmentation mask according to curvature
%
% Syntax: clear_seg = seg_split(seg, BLUGE_THRES, debug_flag)
%
% Clear region segmentation mask according to curvature

    if nargin == 0
        clear
        close all
        fnum = '3';
        inum = '362';
        load(['../data/edge_temp/edge_temp_', fnum, inum, '.mat'], 'seg');
        debug_flag = true;
        img = imread(['../data/CBCT_PNG/', fnum, '/', inum,'.png']);
        bottleneck_ploter = BottleneckPlot(img, seg);
        bottleneck_ploter.EXPAND_SIZE = 0;
    elseif nargin == 3
        % debug_flag = debug_flag;
    else
        debug_flag = false;
    end

    if debug_flag
        point_plot_handle = figure;
    end

    if exist('BLUGE_THRES', 'var')
        param = valid_param(BLUGE_THRES);
    else
        param = valid_param();
    end

    % segment independent region
    [img_label, n_label] = bwlabel(seg);

    img_size = size(seg);

    img_draw = zeros(img_size);

    for cur_bw_label = 1:n_label
        img_piece = (img_label == cur_bw_label);

        % image edge
        piece_edge = edge(img_piece);

        edge_loc = sort_edge_line(piece_edge);

        % Circularity: S = \pi r^2, C = 2 \pi r ==> S / C^2 = 1 / (4 \pi)
        % Eccentricity: sqrt(k * b^2 / a^2)
        s = regionprops(img_piece, 'Circularity', 'Eccentricity', 'Area');
        if s.Eccentricity > param.ECCENTRICITY_THRES || s.Area > param.AREA_THRES
            % pass
        elseif s.Circularity < param.CIRCULARITY_THRES
            img_draw(img_piece) = cur_bw_label;
            fprintf('Tooth %d is odd, whose Circularity is %.2f and Eccentricity is %.2f\n', ...
                cur_bw_label, s.Circularity, s.Eccentricity);
            continue
        end

        % curvature of points in curve
        [circle_rsl, radius_sign] = fit_curvature(edge_loc, img_piece);

        % negative curvature points
        negative_idx = (radius_sign == -1) & circle_rsl(:, 3) < param.RADIUS_THRES;
        negative_edge = edge_loc(negative_idx, :);

        if debug_flag
            figure(point_plot_handle)
            plot(edge_loc(:, 1), edge_loc(:, 2))
            hold on
            plot(negative_edge(:, 1), negative_edge(:, 2), 'o')
        end

        % split negative curvature points
        [edge_neg_center, edge_neg_group] = split_negative_edge(negative_edge);

        if isempty(edge_neg_center)
            img_draw(img_piece) = cur_bw_label;
            continue
        end

        if debug_flag
            figure(point_plot_handle)
            plot(edge_neg_center(:, 1), edge_neg_center(:, 2), 'b*')
            text(edge_loc(1, 1)-5, edge_loc(1, 2), num2str(cur_bw_label))
            axis ij
        end

        % find segmentation line of adjoint region and interp bottleneck line
        seg_line = find_seg_line(edge_loc, edge_neg_center, edge_neg_group, param);

        if debug_flag && nargin == 0
            bottleneck_ploter.plot(edge_neg_center, edge_neg_group, seg_line, edge_loc);
        end

        % separte region
        img_draw_piece = seprate_region(seg_line, img_piece);
        img_draw = img_draw | img_draw_piece;

    end

    if debug_flag && nargin == 0
        % bottleneck_ploter.legend();
        figure(bottleneck_ploter.fh_img);
        print(['../figures/', fnum, '_', inum, '_kappa_bottleneck.eps'], '-depsc2', '-r600')
        figure;
        imshowpair(seg, img_draw)
        figure
        imshow(bottleneck_ploter.img_neg)
        print(['../figures/', fnum, '_', inum, '_neg_edge.eps'], '-depsc2', '-r600')
    end

    if debug_flag
        figure
        imshow(img_draw)
    end

    if nargout ~= 0
        clear_seg = img_draw;
    end

end

function img_piece = seprate_region(seg_line, img_piece)
    %seprate_region - Seprage region according to given fit line
    %
    % Syntax: img_piece = seprate_region(seg_line, img_piece)
    %
    % Seprage region according to given fit line

    for aa = 1:size(seg_line, 1)
        loc1 = seg_line{aa, 1}(:, 2);
        loc2 = seg_line{aa, 1}(:, 1);
        % +1 to expand the line
        ind = sub2ind(size(img_piece), [loc1; loc1 + 1], [loc2; loc2]);
        img_piece(ind) = 0;
    end

end


function fited_points = fit_line(cur_points, partner_points, pt1, pt2)
%fit_line - Fit dense lines
%
% Syntax: fited_point = fit_line(cur_points, partner_points, pt1, pt2)
%
% Fit dense lines
    delta_x = pt2(1) - pt1(1);
    delta_y = pt2(2) - pt1(2);

    if abs(delta_y) > abs(delta_x)
        % y -> x
        fit_points = [cur_points; partner_points];
        [fit_curve, ~] = fit(fit_points(:, 2), fit_points(:, 1), 'poly1');
        yy = pt1(2):(sign(delta_y) * 0.5):pt2(2);
        fit_x = round(fit_curve(yy));
        yy = round(yy);

        % patch points at joint
        interp_start_x = pt1(1):sign(fit_x(1)-pt1(1)):fit_x(1);
        interp_start_y = pt1(2) * ones(length(interp_start_x), 1);
        interp_end_x = fit_x(end):sign(pt2(1)-fit_x(end)):pt2(1);
        interp_end_y = pt2(2) * ones(length(interp_end_x), 1);
        interp_x = [interp_start_x'; fit_x; interp_end_x'];
        interp_y = [interp_start_y; yy'; interp_end_y];
    else
        % x -> y
        fit_points = [cur_points; partner_points];
        [fit_curve, ~] = fit(fit_points(:, 1), fit_points(:, 2), 'poly1');
        xx = pt1(1):(sign(delta_x) * 0.5):pt2(1);
        fit_y = round(fit_curve(xx));
        xx = round(xx);

        % patch points at joint
        interp_start_y = pt1(2):sign(fit_y(1)-pt1(2)):fit_y(1);
        interp_start_x = pt1(1) * ones(length(interp_start_y), 1);
        interp_end_y = fit_y(end):sign(pt2(2)-fit_y(end)):pt2(2);
        interp_end_x = pt2(1) * ones(length(interp_end_y), 1);
        interp_x = [interp_start_x; xx'; interp_end_x];
        interp_y = [interp_start_y'; fit_y; interp_end_y'];
    end
    interp_points = unique([interp_x, interp_y], 'stable', 'rows');
    fited_points = {interp_points, [pt1; pt2]};
end


function edge_loc = sort_edge_line(piece_edge)
%sort_edge_line - get edge line and sort in a cycle order
%
% Syntax: edge_loc = sort_edge_line(piece_edge)
%
% get edge line and sort in a cycle order
    [y, x] = find(piece_edge);

    % distance matrix
    dist_matrix = sqrt((x - x').^2 + (y - y').^2);
    dist_matrix(dist_matrix == 0) = inf;

    % sort edge point according to distance
    n_pts = length(x);
    loc_find = [x, y];
    edge_loc = nan(n_pts, 2);
    edge_loc(1, :) = loc_find(1, :);

    cur_idx = 1;

    for aa = 2:n_pts
        [~, min_idx] = min(dist_matrix(cur_idx, :));
        edge_loc(aa, :) = loc_find(min_idx, :);
        dist_matrix(:, cur_idx) = inf;
        cur_idx = min_idx;
    end
end


function [circle_rsl, radius_sign] = fit_curvature(edge_loc, img_piece)
%fit_curvature - Computer curvature by fitting circle
%
% Syntax: [circle_rsl, radius_sign] = fit_curvature(edge_loc, img_piece)
%
% Computer fit_radius by fitting circle


    n_pts = size(edge_loc, 1);
    circle_rsl = nan(n_pts, 3);

    for aa = 1:n_pts
        sample_pts = edge_loc(mod(aa - 1 + (-7:7), n_pts) + 1, :);
        Par = CircleFitByPratt(sample_pts);
        circle_rsl(aa, :) = Par;
    end

    % subscripts of fited center
    [h, w] = size(img_piece);
    loc1 = circle_rsl(:, 2);
    loc2 = circle_rsl(:, 1);

    judge_lines1 = repmat(1:5, n_pts, 1) .* sign(loc1 - edge_loc(:, 2));
    judge_lines1 = judge_lines1 + edge_loc(:, 2);
    judge_lines2 = repmat(1:5, n_pts, 1) .* sign(loc2 - edge_loc(:, 1));
    judge_lines2 = judge_lines2 .* (loc2 - edge_loc(:, 1)) ./ (loc1 - edge_loc(:, 2));
    judge_lines2 = judge_lines2 + edge_loc(:, 1);

    judge_lines1 = round(judge_lines1);
    judge_lines1(judge_lines1 > h) = h;
    judge_lines1(judge_lines1 < 1) = 1;
    judge_lines2 = round(judge_lines2);
    judge_lines2(judge_lines2 > w) = w;
    judge_lines2(judge_lines2 < 1) = 1;

    judge_lines1(isnan(judge_lines1)) = 1;
    judge_lines2(isnan(judge_lines2)) = 1;

    judge_lines1 = reshape(judge_lines1, [], 1);
    judge_lines2 = reshape(judge_lines2, [], 1);
    center_ind = sub2ind(size(img_piece), judge_lines1, judge_lines2);

    % whether fited center in tooth
    fit_center_menbership = img_piece(center_ind);
    fit_center_menbership = reshape(fit_center_menbership, n_pts, 5);
    fit_center_menbership = sum(fit_center_menbership, 2);
    radius_sign = sign((fit_center_menbership == 5) - 0.5);

    % set very point in very small radius to nagative
    radius_sign(circle_rsl(:, 3) < 6) = -1;

end


function [edge_neg_center, edge_neg_group] = split_negative_edge(negative_edge)
%split_negative_edge - Group negative edge
%
% Syntax: [edge_neg_center, edge_neg_group] = split_negative_edge(negative_edge)
%
% Group negative edge
    % distance between negative curvature points
    edge_diff = diff(negative_edge, 1);
    edge_diff = sqrt(sum(edge_diff.^2, 2));

    % find bottleneck
    edge_gap_loc = [0; find(edge_diff > 5); size(negative_edge, 1)];
    edge_neg_center = nan(length(edge_gap_loc) - 1, 2);
    edge_neg_group = cell(length(edge_gap_loc) - 1, 1);

    for aa = 2:length(edge_gap_loc)
        % negative curvature edge sequence
        edge_seq = negative_edge(edge_gap_loc(aa - 1) + 1:(edge_gap_loc(aa)), :);

        if size(edge_seq, 1) > 2
            seq_median = round(size(edge_seq, 1) / 2);
            edge_neg_center(aa - 1, :) = edge_seq(seq_median, :);
            edge_neg_group(aa - 1) = {edge_seq};
        end

    end

    edge_neg_center = rmmissing(edge_neg_center);
    edge_neg_group(cellfun(@isempty, edge_neg_group)) = [];
end


function fit_lines = find_seg_line(edge_loc, edge_neg_center, edge_neg_group, param)
%find_seg_line - Find segmentation line of adjoint regions
%
% Syntax: fit_lines = find_seg_line(edge_loc, edge_neg_center, edge_neg_group, param)
%
% Find segmentation line of adjoint regions

% distance between center of negative curvature points
    edge_center_dist = sqrt((edge_neg_center(:, 1) - edge_neg_center(:, 1)').^2 ...
        + (edge_neg_center(:, 2) - edge_neg_center(:, 2)').^2);
    edge_center_dist = edge_center_dist + diag(inf(size(edge_neg_center, 1), 1));

    % length of object diagonal line
    obj_size = norm(max(edge_loc) - min(edge_loc));
    edge_center_dist(edge_center_dist > param.SIZE_THRES * obj_size) = inf;

    % group bottlenecks
    nearest_dist = min(edge_center_dist, [], 2);
    edge_center_dist(edge_center_dist > param.NEAREST_THRES * nearest_dist) = inf;

    bluge_ratio_m = zeros(size(edge_center_dist));
    for aa = 1:size(edge_center_dist, 1)
        for bb = 1:size(edge_center_dist, 2)
            if isinf(edge_center_dist(aa, bb))
                continue
            end
            bluge_ratio = points_bluge_ratio(edge_neg_center(aa, :), edge_neg_center(bb, :), edge_loc);
            if bluge_ratio > param.SEG_BLUGE
                bluge_ratio_m(aa, bb) = bluge_ratio;
            end
        end
    end

    [max_bluge, partner_map] = max(bluge_ratio_m, [], 2);

    % fit line
    fit_lines = cell(length(partner_map), 2);
    fited_tab = zeros(length(partner_map), 1);

    for aa = 1:size(partner_map, 1)
        % skip edge fited by partner and far away pair
        if fited_tab(aa) == 1 || max_bluge(aa) == 0
            continue;
        end

        fited_tab(aa) = 1;
        partner_idx = partner_map(aa);

        % also the nearest neighbor of partner, set partner flag to fitted
        if aa == partner_map(partner_idx)
            fited_tab(partner_idx) = 1;
        end

        % center point of nearest neighbor
        cur_points = edge_neg_group{aa};
        partner_points = edge_neg_group{partner_idx};
        pt1 = edge_neg_center(aa, :);
        pt2 = edge_neg_center(partner_idx, :);

        % swap point if parter is before current point
        if partner_idx < aa
            [pt1, pt2] = swap(pt1, pt2);
            [cur_points, partner_points] = swap(cur_points, partner_points);
        end

        % fit line
        fited_points = fit_line(cur_points, partner_points, pt1, pt2);
        fit_lines(aa, :) = fited_points;

    end

    fit_lines(all(cellfun(@isempty, fit_lines), 2), :) = [];
    if isempty(fit_lines)
        fit_lines = [];
    end

end


function bluge_ratio = points_bluge_ratio(pt1, pt2, edge_loc)
%points_bluge_ratio - calculate point bluge ratio
%
% Syntax: bluge_ratio = points_bluge_ratio(pt1, pt2, edge_loc)
%
% calculate point bluge ratio
    start_loc = find(all(edge_loc == pt1, 2));
    end_loc = find(all(edge_loc == pt2, 2));
    % interp function `fit_line` may lead duplicate points
    start_loc = start_loc(1);
    end_loc = end_loc(1);

    if start_loc > end_loc
        [start_loc, end_loc] = swap(start_loc, end_loc);
    end

    % distance on edge index
    direct_dist = norm(pt1 - pt2);
    loc_dist = min(end_loc - start_loc, mod(start_loc - end_loc, size(edge_loc, 1)));
    bluge_ratio = loc_dist / direct_dist;
end


function [out1, out2] = swap(in1, in2)
%swap - Swap item
%
% Syntax: [out1, out2] = swap(in1, in2)
%
% Swap item
    out1 = in2;
    out2 = in1;
end


function param = valid_param(param)
    if nargin == 0
        param = struct();
    end
    if ~isfield(param, 'SIZE_THRES')
        param.SIZE_THRES = 0.3;
    end
    if ~isfield(param, 'NEAREST_THRES')
        param.NEAREST_THRES = 1.3;
    end
    if ~isfield(param, 'RADIUS_THRES')
        param.RADIUS_THRES = 10;
    end
    if ~isfield(param, 'SEG_BLUGE')
        param.SEG_BLUGE = 4;
    end
    if ~isfield(param, 'ECCENTRICITY_THRES')
        param.ECCENTRICITY_THRES = 0.9;
    end
    if ~isfield(param, 'AREA_THRES')
        param.AREA_THRES = 4000;
    end
    if ~isfield(param, 'CIRCULARITY_THRES')
        param.CIRCULARITY_THRES = 0.5;
    end
end
