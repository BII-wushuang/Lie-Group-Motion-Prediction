% Visualization
function Visualize(joint_xyz, joint_xyz_f)
index{1} = 1:21;
nframes = size(joint_xyz,1);

x_loc = joint_xyz(:,:,1);
y_loc = joint_xyz(:,:,2);
z_loc = joint_xyz(:,:,3);
x_min = floor(min(x_loc(:))/500)*500;
x_max = ceil(max(x_loc(:))/500)*500;
y_min = floor(min(y_loc(:))/500)*500;
y_max = ceil(max(y_loc(:))/500)*500;
z_min = floor(min(z_loc(:))/500)*500;
z_max = ceil(max(z_loc(:))/500)*500;

for i = 1:nframes
    chain = cell(1);
    chain_f = cell(1);
    
    for k = 1:length(index)
        chain{k} = squeeze(joint_xyz(i,index{k},:));
        chain_f{k} = squeeze(joint_xyz_f(i,index{k},:));
    end
    
    chain_points = squeeze(joint_xyz(i,:,:));
    chain_points_f = squeeze(joint_xyz_f(i,:,:));
    
    scatter3(chain_points(:,1),chain_points(:,2),chain_points(:,3),'filled','b');
    % scatter3(chain_points_f(:,1),chain_points_f(:,2),chain_points_f(:,3),'filled','r');
    
    line(chain_f{1}(:,1),chain_f{1}(:,2),chain_f{1}(:,3),'Color', 'red');
    
    axis([x_min x_max y_min y_max z_min z_max])
    pause(0.04)
end