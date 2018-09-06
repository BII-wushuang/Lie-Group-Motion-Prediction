data_1 = importdata('./Train/train_xyz/S1_walking_1.mat');
data_2 = importdata('./Train/train_xyz/S1_walking_2.mat');

joint_xyz = [data_1; data_2];
% Visualization
index{1} = 1:6;     %right leg
index{2} = 7:12;    %left leg
index{3} = 13:17;   %spine
index{4} = 18:22;   %left arm
index{5} = 23:27;   %right arm
nframes = size(joint_xyz,1);

linecolors = {'blue', 'green', 'black', 'magenta', 'cyan'};
x_loc = joint_xyz(:,:,1);
y_loc = joint_xyz(:,:,2);
z_loc = joint_xyz(:,:,3);
x_min = -800;
x_max = 800;
y_min = -800;
y_max = 800;
z_min = -1000;
z_max = 1000;

for i = 1:nframes
    chain = cell(1,5);
    for k = 1:length(index)
        chain{k} = squeeze(joint_xyz(i,index{k},:));
    end
    chain_points = squeeze(joint_xyz(i,:,:));
    
    scatter3(chain_points(:,1),chain_points(:,2),chain_points(:,3),'filled','b'); 
    %scatter3(chain_points_f(:,1),chain_points_f(:,2),chain_points_f(:,3),'filled','r');
    
    for k = 1:length(index)
        line(chain{k}(:,1),chain{k}(:,2),chain{k}(:,3),'Color', linecolors{k});
    end
    
    axis([x_min x_max y_min y_max z_min z_max])
    pause(0.04)
end