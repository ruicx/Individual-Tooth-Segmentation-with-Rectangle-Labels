% name:       resetDetBbox.m
% usage:      --
% date:       2021-01-07 16:46:39
% version:    1.0
% Env.:       MATLAB R2019b, WIN10

classdef resetDetBbox

    properties
        box_map
        img_path = '../data/CBCT_PNG/';
        mat_path = "../data/faster_rcnn_box/box_map.mat";
    end

    methods

        function obj = resetDetBbox(json_path, mat_path)

            if isempty(json_path)
                return
            end

            if exist("mat_path", "var")
                obj.mat_path = mat_path;
            else
                mat_path = obj.mat_path;
            end

            if nargin == 0 || exist(mat_path, "file")
                fprintf('Temporary file exists, will ignore json file and load from %s\n', mat_path);
                load(mat_path, "box_map");
                obj.box_map = box_map;
                return;
            end

            % load json file
            if ischar(json_path)
                json_text = fileread(json_path);
                struct_data = jsondecode(json_text);
            else
                struct_data = [];

                for json_name = json_path
                    json_text = fileread(json_name);
                    struct_data = [struct_data; jsondecode(json_text)];
                end

            end

            % json_text = fileread('../data/faster_rcnn_box/train_result_faster_rcnn_Nov07_10_57_0.2244.json');

            box_map = cell(10, 512);

            % filename if from 0
            for item = struct_data'
                name_sp = strsplit(item.file_name, '/');
                fnum = str2double(name_sp{2});
                inum = str2double(name_sp{3}(1:end - 4)) + 1; % filename if from 0
                box_map{fnum, inum} = [box_map{fnum, inum}; round(item.bbox')];
            end

            obj.box_map = box_map;
            save(mat_path, "box_map")

        end % ! end resetDetBbox(json_path)

        function bboxes = get_box(obj, fnum, inum)

            if ~isnumeric(fnum)
                fnum = str2double(fnum);
            end

            if ~isnumeric(inum)
                inum = str2double(inum);
            end

            % filename if from 0
            bboxes = obj.box_map{fnum, inum + 1};

        end % ! end get_box

        function f_handle = show_box(obj, fnum, inum, img)
            if nargin == 3
                % read image
                img = imread([obj.img_path, fnum, '/', inum, '.png']);
                img = double(img);
            end
    
            bboxes = obj.get_box(fnum, inum);
            f_handle = figure;
            imshow(img, [])

            for aa = 1:size(bboxes, 1)
                bbox = bboxes(aa, :);
                rectangle('Position', bbox, 'EdgeColor', '#D95319', 'LineWidth', 1.2);
            end

        end

        function phi_init = get_box_block(obj, fnum, inum, img_size)
            phi_init = zeros(img_size);
            bboxes = obj.get_box(fnum, inum);

            for aa = 1:size(bboxes, 1)
                bbox = bboxes(aa, :);
                phi_init(bbox(2) + (1:bbox(4)), bbox(1) + (1:bbox(3))) = 1;
            end

            phi_init = phi_init > 0;

        end % ! end get_box_block

        function phi_init = get_cycle_block(obj, fnum, inum, img_size, oblique, size_ratio)

            if nargin == 5
                size_ratio = 1;
            end

            phi_init = false(img_size);
            bboxes = obj.get_box(fnum, inum);
            if size(bboxes, 1) == 0
                phi_init = double(phi_init);
                return
            end
            bboxes = sortrows(bboxes, 1);

            for aa = 1:size(bboxes, 1)
                bbox = bboxes(aa, :);
                bbox(4) = min(bbox(4), 512 - bbox(2));
                bbox(3) = min(bbox(3), 512 - bbox(1));
                idx1 = bbox(2) + (1:bbox(4));
                idx2 = bbox(1) + (1:bbox(3));

                tooth_horizontal_loc = bbox(1) + (bbox(3) / 2);
                if (tooth_horizontal_loc > (img_size(2) / 3) && tooth_horizontal_loc < (img_size(2) * 2/3)) ...
                    || (aa < 3 || aa > size(bboxes, 1) - 2)
                    % tooth in center 1 / 3, prior cycle is regular
                    tooth_shape = obj.get_circle_mask(bbox, img_size, 'regular', size_ratio);
                else
                    % otherwise, prior cycle is oblique
                    tooth_shape = obj.get_circle_mask(bbox, img_size, oblique, size_ratio);
                end
                phi_init(idx1, idx2) = phi_init(idx1, idx2) | tooth_shape;

            end

            phi_init = double(phi_init);

        end % ! end get_cycle_block

        function output = get_circle_mask(~, bbox, img_size, oblique, size_ratio)

            if ~ismember(oblique, ["regular", "inside", "outside"])
                error("oblique should in ['regular', 'inside', 'outside'], but now is '%s'", oblique)
            end

            h = bbox(4);
            w = bbox(3);

            % whether tooth is in the left part of image
            left_oblique = bbox(1) + (bbox(3) / 2) < (img_size(2) / 2);

            output = zeros(h, w);
            [xx, yy] = meshgrid(1:w, 1:h);

            if oblique == "regular"
                f = @(x, y) (x - (1 + w) / 2).^2 / (size_ratio * w)^2 + (y - (1 + h) / 2).^2 / (size_ratio * h)^2;
                output(f(xx, yy) < 0.25) = 1;
            else
                a = w; b = h;

                if a > b
                    b = 0.8 * b;
                else
                    a = 0.9 * a;
                end

                A = a * a * h * h + b * b * w * w;
                B = 2 * (a * a - b * b) * h * w;
                C = a * a * w * w + b * b * h * h;
                F = a * a * b * b * (size_ratio^2 * (h^2 + w^2)) / 4;

                if left_oblique
                    B = abs(B);
                else
                    B = -abs(B);
                end

                if oblique == "inside"
                    B = -B;
                end

                f = @(x, y) A * (x - (1 + w) / 2).^2 + C * (y - (1 + h) / 2).^2 +B * (x - (1 + w) / 2) .* (y - (1 + h) / 2);
                output(f(xx, yy) < F) = 1;
            end
        end % ! end get_circle_mask

    end % ! end methods

end % ! end classdef
