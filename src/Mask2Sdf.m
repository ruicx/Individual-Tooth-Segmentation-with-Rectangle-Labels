function SDF = Mask2Sdf(binary_image)
    % This function compute the initial Signed Distance Function from binary mask
    if all(binary_image(:))
        SDF = binary_image;
    else
        % N = 2 .* binary_image - 1;
        SDF = bwdist(binary_image < 0.5) - bwdist(binary_image >= 0.5);
    end

end
