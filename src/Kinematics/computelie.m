% Compute joints given Lie parameters
function joint_xyz = computelie(lie_params)
    njoints = size(lie_params,1);
    for j = 1 : njoints
        if (j==1)
            A(j,:,:) = lietomatrix(lie_params(j,1:3)', lie_params(j,4:6)');
        else
            A(j,:,:) = squeeze(A(j-1,:,:)) * lietomatrix(lie_params(j,1:3)', lie_params(j,4:6)');
        end
    end
    for j = 1: njoints
        xyz = squeeze(A(j,:,:)) * [0; 0; 0; 1];
        joint_xyz(j,:) = xyz(1:3);
    end
end
