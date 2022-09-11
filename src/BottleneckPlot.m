% name:       BottleneckPlot.m
% usage:      --
% author:     Ruicheng
% date:       2021-02-04 20:12:06
% version:    1.0
% Env.:       MATLAB R2019b, WIN10


classdef BottleneckPlot < handle
    properties
        fh_img;
        img_neg;
        img_size;
        EXPAND_SIZE = 0;
        temp = -1;
        plot_all_neg = false;
    end % ! END properties

    methods
        function obj = BottleneckPlot(img, seg, EXPAND_SIZE)
            obj.img_size = size(img);
            obj.img_neg = zeros(obj.img_size);
            obj.fh_img = figure;
            imshow(img);
            hold on
            contour(seg, [0, 1], 'r')
            if nargin == 3
                obj.EXPAND_SIZE = EXPAND_SIZE;
            end
        end

        function [edge_neg_center_s, edge_neg_group_s] = find_seg_edge_group(obj, edge_neg_center, edge_neg_group, seg_line)
            edge_neg_center_s = zeros(2 * size(seg_line, 1), 2);
            edge_neg_group_s = cell(2 * size(seg_line, 1), 1);
            for aa = 1:size(seg_line, 1)
                line_select = ismember(edge_neg_center, seg_line{aa, 2}, 'rows');
                edge_neg_center_s(2*aa+(-1:0), :) = edge_neg_center(line_select, :);
                edge_neg_group_s(2*aa+(-1:0)) = edge_neg_group(line_select);
            end

        end % ! END find_seg_edge_group

        function edge_neg_group = expand_neg_group(obj, edge_neg_group, edge_loc)
            if isempty(edge_neg_group)
                return;
            end

            for aa = 1:size(edge_neg_group, 1)
                start_loc = find(all(edge_loc == edge_neg_group{aa}(1, :), 2));
                assert(length(start_loc) == 1)
                expand_start = mod(start_loc + (-obj.EXPAND_SIZE:0), size(edge_loc, 1));
                edge_neg_group{aa} = [edge_loc(expand_start, :); edge_neg_group{aa}];

                end_loc = find(all(edge_loc == edge_neg_group{aa}(end, :), 2));
                assert(length(end_loc) == 1)
                expand_end = mod(end_loc + (0:obj.EXPAND_SIZE), size(edge_loc, 1)) + 1;
                edge_neg_group{aa} = [edge_neg_group{aa}; edge_loc(expand_end, :)];
            end

        end % ! END expand_neg_group

        function plot(obj, edge_neg_center, edge_neg_group, seg_line, edge_loc)
            [edge_neg_center_s, edge_neg_group_s] = obj.find_seg_edge_group(edge_neg_center, edge_neg_group, seg_line);

            % plot all negative edge or just plot bottleneck
            if obj.plot_all_neg == false
                edge_neg_group = edge_neg_group_s;
            end

            % expand the length of negative edge for plot
            edge_neg_group = obj.expand_neg_group(edge_neg_group, edge_loc);

            for ee = 1:length(edge_neg_group)

                if length(edge_neg_group) == 1
                    break;
                end

                obj.img_neg(sub2ind(obj.img_size, edge_neg_group{ee}(:, 2), edge_neg_group{ee}(:, 1))) = 1;
                figure(obj.fh_img);
                plot(edge_neg_group{ee}(:, 1), edge_neg_group{ee}(:, 2), 'g', 'LineWidth', 1.5);

                if ee == 1
                    plot(edge_neg_center_s(:, 1), edge_neg_center_s(:, 2), 'o', 'Color', '#436EEE')
                end

            end
        end

        function legend(obj)
            figure(obj.fh_img);
            legend('Seg Contour', 'Negative Contour', 'Bottleneck', 'AutoUpdate', 'off');
        end

    end % ! END methods

end % ! END class