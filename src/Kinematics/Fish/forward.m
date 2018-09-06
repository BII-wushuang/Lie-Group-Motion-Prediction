%% Forward 1inematics
% Input: nframes * njoints * 6 array of 6D Lie algebra parameters
% Output: nframes * njoints * 3 array of 3D joint locations

addpath('../');

% Represent fish with 1 chain
index{1} = 1:21;

joint_xyz = importdata('../../data/Fish/Raw/S1_xyz.mat');
lie_parameters = importdata('../../data/Fish/Raw/S1_lie.mat');
nframes = size(lie_parameters,1);
njoints  = size(lie_parameters,2);
joint_xyz_f = zeros(nframes, njoints, 3);

for i = 1:nframes
   joint_xyz_f(i,index{1},:) = computelie(squeeze(lie_parameters(i,index{1},:)));
end

% Visualization
Visualize(joint_xyz, joint_xyz_f);
