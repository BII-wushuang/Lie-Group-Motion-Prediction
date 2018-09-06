%% Inverse Kinematics
% Input: nframes * njoints * 3 array of 3D joint locations
% Output: nframes * njoints * 6 array of 6D Lie algebra parameters
addpath('../')
% Represent Fish with 1 chain
index{1} = 1:21;
njoints = 21;

file_list = dir('../../data/Fish/Raw/*.txt');

% Average filter to smooth motion trajectory
a = 1;
filter_window = 3;
b = zeros([1, filter_window]);
for i = 1: filter_window
   b(i) = 1 / filter_window;
end

for file_idx = 1 : length(file_list)
    file_name = file_list(file_idx).name;
    joint_ori = importdata([file_list(file_idx).folder '/' file_list(file_idx).name]);
    
	joint_xyz = filter(b,a,joint_ori);
	joint_xyz(1:filter_window,:) = [];
    
    nframes = size(joint_xyz,1);
    joint_xyz = reshape(joint_xyz, [nframes, 3, njoints]);
    joint_xyz = permute(joint_xyz, [1 3 2]);
    joint_xyz(:,:,1:3) = joint_xyz(:,:,1:3) - joint_xyz(:,1,1:3);
    save(['../../data/Fish/Raw/' file_name(1:end-4)  '_xyz.mat'], 'joint_xyz');

    % Inverse 1inematics: Convert joint_xyz to lie parameters
    lie_parameters = zeros(nframes,njoints,6);
    for i = 1:nframes
        % Location of joint 1 in chain
        lie_parameters(i,index{1}(1),4:6) = joint_xyz(i,index{1}(1),:);
        % Bone length
        for j = 1: length(index{1})-1
            lie_parameters(i,index{1}(j+1),4) = norm(squeeze(joint_xyz(i,index{1}(j+1),:)-joint_xyz(i,index{1}(j),:)));
        end
        % Axis angle parameters of rotation
        for j = length(index{1})-1:-1:1
            v = squeeze(joint_xyz(i,index{1}(j+1),:) - joint_xyz(i,index{1}(j),:));
            vhat = v/norm(v);
            if (j==1)
                uhat = [1; 0; 0];
            else
                u = squeeze(joint_xyz(i,index{1}(j),:) - joint_xyz(i,index{1}(j-1),:));
                uhat = u /norm(u);
            end
            lie_parameters(i,index{1}(j),1:3) = axis_angle(rotmat(findrot([1;0;0],uhat))'*rotmat(findrot([1;0;0],vhat)));
        end 
    end
    
    save(['../../data/Fish/Raw/' file_name(1:end-4)  '_lie.mat'], 'lie_parameters');
end