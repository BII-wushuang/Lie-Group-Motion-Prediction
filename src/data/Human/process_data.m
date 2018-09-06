%% Batch rename D3_positions
foldernum = [1, 5, 6, 7, 8, 9, 11];
folder_prefix = './Raw/S';

for k = 1:length(foldernum)
    folderdir = [folder_prefix num2str(foldernum(k))];
    if(~exist(folderdir))
        mkdir(folderdir);
    end
end

for k = 1:length(foldernum)
    files = dir(['./Raw/S' num2str(foldernum(k)) '/MyPoseFeatures/D3_Positions/*.cdf']);
    
    for i=1:length(files)
        filename = files(i);
        names = strsplit(files(i).name,'.');
        names = strsplit(names{1}, ' ');
        action = names{1};
        
        newname = [folder_prefix num2str(foldernum(k)) '/' action '_2.cdf'];
        if (exist(newname))
            newname = [folder_prefix num2str(foldernum(k)) '/' action '_1.cdf'];
        end
        copyfile([files(i).folder '/' files(i).name], newname);
    end
end
