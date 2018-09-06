%% Preprocess the raw CDF data from Human3.6m
% Input: raw CDF dataset 
% Output: nframes * njoints * 3 joint xyz coordinates

% Kinematic Chain Configuration
% joint 14 and 25 and 17 are the same point
% joint 1 and 12 are the same point
% joint 20 and 21 are the same point
% joint 23 and 24 are the same point
% joint 28 and 29 are the same point

subjects = {'S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11',};

for i = 1: length(subjects)
    file_list = dir(['../../data/Human/Raw/' subjects{i} '/*.cdf']);
    
    for j = 1: length(file_list)
        file_name = file_list(j).name;
        file_name = file_name(1:end-4);
        
        data = importdata([file_list(j).folder '/' file_list(j).name]);
        data = data{1};

        nframes = size(data,1);

        data = reshape(data, [nframes 3 size(data,2)/3]);
        data = permute(data, [1 3 2]);

        discarded = [12, 17, 21, 22, 24, 25, 29, 31, 32];
        data(:, discarded,:) = [];

        % Represent human body with 5 chains
        index{1} = [1 2:6];     %right leg
        index{2} = [1 7:11];    %left leg
        index{3} = [1 12:15];   %spine
        index{4} = [13 16:19];  %left arm
        index{5} = [13 20:23];  %right arm

        njoints = 0;
        for k = 1:length(index)
            njoints = njoints + numel(index{k});
        end

        joint_xyz = zeros(nframes,njoints,3);

        counter = 0;
        for k = 1:length(index)
            joint_xyz(:,counter+1:counter+numel(index{k}),:) = data(:, index{k},:);
            counter = counter+numel(index{k});
        end

        save(['../../data/Human/Raw/' subjects{i} '_' file_name '.mat'],'joint_xyz');
    end
end