% name:       DetBboxFcos.m
% usage:      --
% author:     Ruicheng
% date:       2021-01-31 10:41:05
% version:    1.0
% Env.:       MATLAB R2019b, WIN10


classdef DetBboxFcos < resetDetBbox
    properties
        score_map
        bbox_threshold = 0.4;
    end

    methods
        function obj = DetBboxFcos(bbox_json_path, file_name_map)
            % initial superclass
            obj@resetDetBbox([]);

            temp_path = "../data/fcos_box/box_map.mat";
            if nargin == 0 || exist(temp_path, "file")
                fprintf('Temporary file exists, will ignore json file and load from ''%s''\n', temp_path);
                load(temp_path, "box_map", "score_map");
                obj.box_map = box_map;
                obj.score_map = score_map;
                return;
            end

            % read detected bbox
            bbox_json_text = fileread(bbox_json_path);
            bbox_data = jsondecode(bbox_json_text);
            
            % load name map
            name_json_text = fileread(file_name_map);
            name_map = jsondecode(name_json_text);
            assert(length(name_map) == name_map(end).id)

            box_map = cell(10, 512);
            score_map = cell(10, 512);

            % filename if from 0
            for item = bbox_data'
                id = item.image_id;
                name_sp = strsplit(name_map(id).file_name, '/');
                fnum = str2double(name_sp{2});
                inum = str2double(name_sp{3}(1:end - 4)) + 1; % filename if from 0
                box_map{fnum, inum} = [box_map{fnum, inum}; round(item.bbox')];
                score_map{fnum, inum} = [score_map{fnum, inum}; item.score];
            end

            obj.box_map = box_map;
            obj.score_map = score_map;
            save(temp_path, "box_map", "score_map")
        end

        function bboxes = get_box(obj, fnum, inum)

            if ~isnumeric(fnum)
                fnum = str2double(fnum);
            end

            if ~isnumeric(inum)
                inum = str2double(inum);
            end

            % filename if from 0
            bboxes = obj.box_map{fnum, inum + 1};
            scores = obj.score_map{fnum, inum + 1};
            bboxes(scores < obj.bbox_threshold, :) = [];
        end


    end

end