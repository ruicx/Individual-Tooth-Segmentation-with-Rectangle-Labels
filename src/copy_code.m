% name:       copy_code.m
% usage:      --
% date:       2020-12-04 15:40:44
% version:    1.0
% Env.:       MATLAB R2019b, WIN10


function copy_code(fnum, inum)
%copy_code - copy code
%
% Syntax: copy_code(fnum, inum)
%
% Copy code
    dst_name = ['demo_', fnum, '_', inum, '.m'];
    dst_path = ['../demo_seg/', fnum, '/', dst_name];

    copyfile('demo_generator.m', dst_path)

    fp1 = fopen('./demo_generator.m', 'r');
    fp2 = fopen(dst_path, 'w');

    cur_line = fgets(fp1);
    while ~contains(cur_line, '% copy code')
        % comment title
        cur_line = strrep(cur_line, "demo_generator.m", dst_name);
        cur_line = strrep(cur_line, "2020-11-24 08:26:41", datestr(datetime, "yyyy-mm-dd HH:MM:SS"));
        % file number
        cur_line = regexprep(cur_line, "fnum = '\d+';", "fnum = '"+ fnum +"';");
        cur_line = regexprep(cur_line, "inum = '\d+';", "inum = '"+ inum +"';");
        % path
        cur_line = strrep(cur_line, "'../", "'../../");
        cur_line = strrep(cur_line, "% addpath('...')", "addpath('../../src')");
        fprintf(fp2, '%s', cur_line);
        cur_line = fgets(fp1);
    end

    fclose(fp1);
    fclose(fp2);

end
