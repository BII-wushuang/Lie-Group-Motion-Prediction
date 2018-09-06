%% Inverse Kinematics
% Input: nframes * njoints * 3 array of 3D joint locations
% Output: nframes * njoints * 6 array of 6D Lie algebra parameters

addpath('../');

% Represent human body with 5 chains
index{1} = 1:6;     %right leg
index{2} = 7:12;    %left leg
index{3} = 13:17;   %spine
index{4} = 18:22;   %left arm
index{5} = 23:27;   %right arm

normalize = 1;
if (normalize)
    S5_joints = importdata('../../data/Human/standard_bone/S5.mat');
    njoints  = size(S5_joints,2);
    bone = zeros(njoints, 3);
    for j = 1: length(index)
        for k = 2: length(index{j})
            bone(index{j}(k),1) = round(norm(squeeze(S5_joints(1,index{j}(k),:)-S5_joints(1,index{j}(k-1),:)))/5)*5; 
        end
    end
    save('../../data/Human/standard_bone/bone.mat','bone');
end

mkdir('../../data/Human/Train/train_lie')
mkdir('../../data/Human/Train/train_xyz')
mkdir('../../data/Human/Test/x_test_lie')
mkdir('../../data/Human/Test/y_test_lie')
mkdir('../../data/Human/Test/x_test_xyz')
mkdir('../../data/Human/Test/y_test_xyz')

file_list = dir('../../data/Human/Raw/*.mat');

A = zeros(length(index{3}),3);

for file_idx = 1 : length(file_list)
    file_name = file_list(file_idx).name;
    if (contains(file_name,'lie'))
        continue
    end
    joint_xyz = importdata([file_list(file_idx).folder '/' file_list(file_idx).name]);
    
    % Zero centre data
    joint_xyz(:,:,1:3) = joint_xyz(:,:,1:3) - joint_xyz(:,1,1:3);
    save(['../../data/Human/Train/train_xyz/' file_name(1:end-4) '_xyz.mat'], 'joint_xyz');

    nframes = size(joint_xyz,1);
    
    % Inverse kinematics: Convert joint_xyz to lie parameters
%     lie_parameters = zeros(nframes,njoints,6);
%     for i = 1:nframes
%         lie_parameters(i,index{k}(1),4:6) = joint_xyz(i,index{k}(1),:);
%         for k = 1:length(index)
%             % Location of joint 1 in chain
%             
%             % Bone length
%             for j = 1: length(index{k})-1
%                 lie_parameters(i,index{k}(j+1),4) = norm(squeeze(joint_xyz(i,index{k}(j+1),:)-joint_xyz(i,index{k}(j),:)));
%             end
%             % Axis angle parameters of rotation
%             for j = length(index{k})-1:-1:1
%                 v = squeeze(joint_xyz(i,index{k}(j+1),:) - joint_xyz(i,index{k}(j),:));
%                 vhat = v/norm(v);
%                 if (j==1)
%                     uhat = [1; 0; 0];
%                 else
%                     u = squeeze(joint_xyz(i,index{k}(j),:) - joint_xyz(i,index{k}(j-1),:));
%                     uhat = u /norm(u);
%                 end
%                 lie_parameters(i,index{k}(j),1:3) = axis_angle(rotmat(findrot([1;0;0],uhat))'*rotmat(findrot([1;0;0],vhat)));
%             end
%         end
%     end
    
    lie_parameters_normalized = zeros(nframes,njoints,6);
    for i = 1:nframes
        % Bone length
        lie_parameters_normalized(i,:,4:6) = bone;
        for k = 1:length(index)
            if (k>3)
                A = computelie(squeeze(lie_parameters_normalized(i, index{3}, :)));
                %A = computelie_euler(squeeze(lie_parameters_normalized(i, index{3}, :)));
                lie_parameters_normalized(i,index{k}(1),4:6) = A(3,:);
            end
            % Axis angle parameters of rotation
            for j = length(index{k})-1:-1:1
                v = squeeze(joint_xyz(i,index{k}(j+1),:) - joint_xyz(i,index{k}(j),:));
                vhat = v/norm(v);
                if (j==1)
                    uhat = [1; 0; 0];
                else
                    u = squeeze(joint_xyz(i,index{k}(j),:) - joint_xyz(i,index{k}(j-1),:));
                    uhat = u /norm(u);
                end
                lie_parameters_normalized(i,index{k}(j),1:3) = axis_angle(rotmat(findrot([1;0;0],uhat))'*rotmat(findrot([1;0;0],vhat)));
                %lie_parameters_normalized(i,index{k}(j),1:3) = RotMat2Euler(rotmat(findrot([1;0;0],uhat))'*rotmat(findrot([1;0;0],vhat)));
            end
        end
    end
    save(['../../data/Human/Train/train_lie/' file_name(1:end-4)  '_lie.mat'], 'lie_parameters_normalized');
end