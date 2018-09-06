%% Forward Kinematics
% Input: nframes * njoints * 6 array of 6D Lie algebra parameters
% Output: nframes * njoints * 3 array of 3D joint locations

addpath('../');

joint_xyz = importdata('../../data/Human/Train/train_xyz/S1_Directions_1_xyz.mat');

% Represent human body with 5 chains
index{1} = 1:6;     %right leg
index{2} = 7:12;    %left leg
index{3} = 13:17;   %spine
index{4} = 18:22;   %left arm
index{5} = 23:27;   %right arm

file_list = dir('../../data/Human/Train/train_lie/S1_Directions_1_lie.mat');

for file_idx = 1:length(file_list)
    file_name = file_list(file_idx).name;
    
    lie_parameters = importdata([file_list(file_idx).folder '/' file_name]);
    nframes = size(lie_parameters,1);
    njoints  = size(lie_parameters,2);
    joint_xyz_f = zeros(nframes, njoints, 3);

    for i = 1:nframes
        for k = 1:length(index)
            joint_xyz_f(i,index{k},:) = computelie(squeeze(lie_parameters(i,index{k},:)));
            %joint_xyz_f(i,index{k},:) = computelie_euler(squeeze(lie_parameters(i,index{k},:)));
        end
    end
    %save(['../../data/Human/Test/x_test_xyz/' file_name(1:end-8) '.mat'],'joint_xyz_f');
end
% Visualization
Visualize(joint_xyz, joint_xyz_f);
