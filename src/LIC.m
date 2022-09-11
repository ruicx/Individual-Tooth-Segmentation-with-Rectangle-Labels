function [u,b] = LIC(Img, u, u_prior, param)

    param = set_default_param(param);

    if all(length(unique(u_prior(:))) == 1)
        u = u_prior;
        b = ones(size(Img));
    else

        b = ones(size(Img)); % initialize bias field

        K = fspecial('gaussian', round(2 * param.sigma) * 2 + 1, param.sigma); % Gaussian kernel
        KONE = conv2(ones(size(Img)), K, 'same');

        % smooth signed distance function
        smooth_sdf = abs(Mask2Sdf(u_prior > 0));
        % smooth_sdf = log(smooth_sdf) .* u_prior;
        log_sdf_loc = 1:(size(smooth_sdf, 1) / 2);
        smooth_sdf(log_sdf_loc, :) = log(smooth_sdf(log_sdf_loc, :));
        smooth_sdf = smooth_sdf .* u_prior;
        % smooth_sdf(~u_prior) = 0.1 * smooth_sdf(~u_prior);

        for aa = 1:param.iter_outer
            [u, b, ~] = lse_bfe(u, Img, b, K, KONE, u_prior, smooth_sdf, param);

            if param.draw_step > 0 && mod(aa, param.draw_step) == 0

                if ~exist('fhandle_draw', 'var')
                    fhandle_draw = figure;
                end

                figure(fhandle_draw);
                imshow(Img, [0, 255]);
                colormap(gray); hold on; axis off, axis equal
                contour(u, [0 0], 'r');
                title([num2str(aa), ' iterations']);
                hold off;
            end

            % evaluate segmentation and report key dice level
            seg_evaluation(u, param, aa)

        end

    end

end


function [u, b, C] = lse_bfe(u0, Img, b, Ksigma, KONE, u_prior, smooth_sdf, param)
% This code implements the level set evolution (LSE) and bias field estimation
% proposed in the following paper:
%      C. Li, R. Huang, Z. Ding, C. Gatenby, D. N. Metaxas, and J. C. Gore,
%      "A Level Set Method for Image Segmentation in the Presence of Intensity
%      Inhomogeneities with Application to MRI", IEEE Trans. Image Processing, 2011
%
% Note:
%    This code implements the two-phase formulation of the model in the above paper.
%    The two-phase formulation uses the signs of a level set function to represent
%    two disjoint regions, and therefore can be used to segment an image into two regions,
%    which are represented by (u>0) and (u<0), where u is the level set function.
%
%    All rights researved by Chunming Li, who formulated the model, designed and
%    implemented the algorithm in the above paper.
%
% E-mail: lchunming@gmail.com
% URL: http://www.engr.uconn.edu/~cmli/
% Copyright (c) by Chunming Li
% Author: Chunming Li

    u = u0;
    KB1 = conv2(b, Ksigma, 'same');
    KB2 = conv2(b.^2, Ksigma, 'same');
    C = updateC(Img, u, KB1, KB2, param.epsilon);

    KONE_Img = Img.^2 .* KONE;
    u = updateLSF(Img, u, C, KONE_Img, KB1, KB2, u_prior, smooth_sdf, param);

    Hu = Heaviside(u, param.epsilon);
    M(:, :, 1) = Hu;
    M(:, :, 2) = 1 - Hu;
    b = updateB(Img, C, M, Ksigma);

end

% update level set function
function u = updateLSF(Img, u0, C, KONE_Img, KB1, KB2, u_prior, smooth_sdf, param)
    u = u0;
    Hu = Heaviside(u, param.epsilon);
    M(:, :, 1) = Hu;
    M(:, :, 2) = 1 - Hu;
    N_class = size(M, 3);
    e = zeros(size(M));
    u = u0;

    for kk = 1:N_class
        e(:, :, kk) = KONE_Img - 2 * Img .* C(kk) .* KB1 + C(kk)^2 * KB2;
    end

    for kk = 1:param.iter_inner
        u = NeumannBoundCond(u);
        K = curvature_central(u); % div()
        DiracU = Dirac(u, param.epsilon);
        ImageTerm = -DiracU .* (e(:, :, 1) - e(:, :, 2));
        penalizeTerm = param.mu * (4 * del2(u) - K);
        lengthTerm = param.nu .* DiracU .* K;

        if param.restriction_type == "heaviside"
            Hu = Heaviside(u, param.epsilon);
            priorTerm = -param.alpha * 2 .* smooth_sdf .* (Hu - u_prior) .* DiracU;
        else
            priorTerm = -param.alpha * (u - u_prior);
        end

        if param.beta == 0
            % without curvature term
            u = u + param.timestep * (lengthTerm + penalizeTerm + ImageTerm + priorTerm);
        else
            % with curvature term
            csi = curvature_sign_indicator(K, param.epsilon);
            curvatureTerm = -param.beta * (1 - csi) .* K .* DiracU;
            u = u + param.timestep * (csi .* (lengthTerm + penalizeTerm + ImageTerm + priorTerm) + curvatureTerm);
        end

    end

end


% update b
function b = updateB(Img, C, M, Ksigma)

    PC1 = zeros(size(Img));
    PC2 = PC1;
    N_class = size(M, 3);

    for kk = 1:N_class
        PC1 = PC1 + C(kk) * M(:, :, kk);
        PC2 = PC2 + C(kk)^2 * M(:, :, kk);
    end

    KNm1 = conv2(PC1 .* Img, Ksigma, 'same');
    KDn1 = conv2(PC2, Ksigma, 'same');

    b = KNm1 ./ KDn1;

end


% Update C
function C_new = updateC(Img, u, Kb1, Kb2, epsilon)
    Hu = Heaviside(u, epsilon);
    M(:, :, 1) = Hu;
    M(:, :, 2) = 1 - Hu;
    N_class = size(M, 3);
    C_new = zeros(1, N_class);

    for kk = 1:N_class
        Nm2 = Kb1 .* Img .* M(:, :, kk);
        Dn2 = Kb2 .* M(:, :, kk);
        C_new(kk) = sum(Nm2(:)) / sum(Dn2(:));
    end

end


% Make a function satisfy Neumann boundary condition
function g = NeumannBoundCond(f)
    [nrow, ncol] = size(f);
    g = f;
    g([1 nrow], [1 ncol]) = g([3 nrow - 2], [3 ncol - 2]);
    g([1 nrow], 2:end - 1) = g([3 nrow - 2], 2:end - 1);
    g(2:end - 1, [1 ncol]) = g(2:end - 1, [3 ncol - 2]);
end


function k = curvature_central(u)
    % compute curvature for u with central difference scheme
    [ux, uy] = gradient(u);
    normDu = sqrt(ux.^2 + uy.^2 + 1e-10);
    Nx = ux ./ normDu;
    Ny = uy ./ normDu;
    [nxx, ~] = gradient(Nx);
    [~, nyy] = gradient(Ny);
    k = nxx + nyy;
end


function h = Heaviside(x, epsilon)
    h = 0.5 * (1 + (2 / pi) * atan(x ./ epsilon));
end


function f = Dirac(x, epsilon)
    f = (epsilon / pi) ./ (epsilon^2. + x.^2);
end

function beta = curvature_sign_indicator(kappa, epsilon)
    beta = Heaviside(kappa, epsilon);
end


function param = set_default_param(param)
    if nargin == 0
        param = struct();
    end
    if ~isfield(param, 'nu')
        param.nu = 0.03;
    end
    if ~isfield(param, 'sigma')
        param.sigma = 3;
    end
    if ~isfield(param, 'iter_outer')
        param.iter_outer = 50;
    end
    if ~isfield(param, 'iter_inner')
        param.iter_inner = 10;
    end
    if ~isfield(param, 'alpha')
        param.alpha = 600;
    end
    if ~isfield(param, 'timestep')
        param.timestep = 0.1;
    end
    if ~isfield(param, 'mu')
        param.mu = 1;
    end
    if ~isfield(param, 'beta')
        param.beta = 40;
    end
    if ~isfield(param, 'epsilon')
        param.epsilon = 1;
    end
    if ~isfield(param, 'draw_step')
        param.draw_step = 1;
    end
    if ~isfield(param, 'initial_shape')
        param.initial_shape = 'rectangle';
    end
    if ~isfield(param, 'restriction_type')
        param.restriction_type = 'heaviside';
    end
end


function seg_evaluation(u, param, aa)
    persistent flage_dice_it9 flage_dice_it8
    if isempty(flage_dice_it9)
        flage_dice_it8 = false;
        flage_dice_it9 = false;
    end
    if isfield(param, 'gt')
        seg = medfilt2(double(u < 0));
        seg = imfill(seg, 'holes');
        seg = bwareaopen(seg, 200);
        seg_dice = dice(seg, param.gt);

        if seg_dice > 0.9 && flage_dice_it9 == false
            flage_dice_it9 = true;
            fprintf('Dice over 0.9 in %d itereation\n', aa)
        elseif seg_dice > 0.8 && flage_dice_it8 == false
            flage_dice_it8 = true;
            fprintf('Dice over 0.8 in %d itereation\n', aa)
        end
    end
end
