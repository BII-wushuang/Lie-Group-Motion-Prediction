files = {'gt', 'erd', '3lr', 'gru', 'hmr'};
frames = [1, 16, 31, 46, 61];
border = 3;
for j = 1: length(files)
    picture = [];
    file = files{j};
    for i = frames
        img = imread(['./' file '/output_' num2str(i) '.png'], 'png');
        img(:,1:border,:) = 0;
        img(:,end-border+1:end,:) = 0;
        img(1:border,:,:) = 0;
        img(end-border+1:end,:,:) = 0;
        
        %img = imcrop(img,[20 15 440 330]);
        %[oldHeight, oldWidth, oldNumberOfColorChannels] = size(img);
        %newWidth = int32(oldHeight * 4/4);
        %newImage = imresize(img, [oldHeight newWidth]);
        picture = cat(2,picture,img);    
    end
    
    imshow(picture)
    imwrite(picture, ['./' file '.png'])
end
    