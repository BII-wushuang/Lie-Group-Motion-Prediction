actions = {'directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting', 'sittingdown', 'smoking', 'photo', 'waiting', 'walking', 'dog', 'together'};
subjects = {'S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'};

name_idx = 0;
% for k = 1: length(subjects)
%     file_list = dir(['./Train/train_lie/' subjects{k} '_*.mat']);
%     
%     current_idx = zeros(length(actions),1);
%     for i = 1:length(file_list)
%         idx = cell(15,1);
%         file_name = file_list(i).name;
%         for j = 1:length(actions)
%             idx{j} = strfind(upper(file_name), upper(actions{j}));
%         end
%         non_empty = find(~cellfun(@isempty, idx));
%         
%         action_idx = non_empty(end);
%         action = actions{action_idx};
%         
%         current_idx(action_idx) = current_idx(action_idx) + 1;
%         name_idx = num2str(current_idx(action_idx));
%         new_filename = ['./Train/train_lie/' subjects{k} '_' action '_' name_idx '_lie.mat'];
%         
%         movefile([file_list(i).folder '/' file_name], new_filename);
%     end
% end

name_idx = 0;
for k = 1: length(subjects)
    file_list = dir(['./Train/train_xyz/' subjects{k} '_*.mat']);
    
    current_idx = zeros(length(actions),1);
    for i = 1:length(file_list)
        idx = cell(15,1);
        file_name = file_list(i).name;
        for j = 1:length(actions)
            idx{j} = strfind(upper(file_name), upper(actions{j}));
        end
        non_empty = find(~cellfun(@isempty, idx));
        
        action_idx = non_empty(end);
        action = actions{action_idx};
        
        current_idx(action_idx) = current_idx(action_idx) + 1;
        name_idx = num2str(current_idx(action_idx));
        new_filename = ['./Train/train_xyz/' subjects{k} '_' action '_' name_idx '_xyz.mat'];
        
        movefile([file_list(i).folder '/' file_name], new_filename);
    end
end