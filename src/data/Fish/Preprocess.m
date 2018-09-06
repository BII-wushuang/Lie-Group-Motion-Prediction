% Seperate data into test set and train set
% Test sets are taken from S6
xyz_data = importdata('./Raw/S6_xyz.mat');
lie_data = importdata('./Raw/S6_lie.mat');

num_test = 8;
num_x_test = 50;
num_y_test = 100;

start = [1, 201, 401, 601, 801, 1001, 1201, 1401];

try
    mkdir('./Test/x_test_xyz');
    mkdir('./Test/y_test_xyz');
    mkdir('./Test/x_test_lie');
    mkdir('./Test/y_test_lie');
catch
end

for i = 1:num_test
    x_test_start = start(i);
    x_test_end = x_test_start+num_x_test-1;
    y_test_start = x_test_end+1;
    y_test_end = y_test_start+num_y_test-1;
    
    x_test_xyz = xyz_data(x_test_start:x_test_end,:,:);
    y_test_xyz = xyz_data(y_test_start:y_test_end,:,:);
    
    x_test_lie = lie_data(x_test_start:x_test_end,:,:);
    y_test_lie = lie_data(y_test_start:y_test_end,:,:);
    
    save(['./Test/x_test_xyz/test_' num2str(i-1) '.mat'],'x_test_xyz');
    save(['./Test/y_test_xyz/test_' num2str(i-1) '.mat'],'y_test_xyz');
    save(['./Test/x_test_lie/test_' num2str(i-1) '_lie.mat'],'x_test_lie');
    save(['./Test/y_test_lie/test_' num2str(i-1) '_lie.mat'],'y_test_lie');
end