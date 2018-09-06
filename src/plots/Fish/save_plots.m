files = {'gt', 'erd', '3lr', 'resgru', 'hmr'};
frames = [3, 23, 43, 63, 83];

for j = 1: length(files)
    picture = [];
    file = files{j};
    for i = frames
        img = imread(['./' file '/output_' num2str(i) '.png'], 'png');
        img = imcrop(img,[105.5 85.5 240 270]);
        [oldHeight, oldWidth, oldNumberOfColorChannels] = size(img);
        newWidth = int32(oldHeight * 4/4);
        newImage = imresize(img, [oldHeight newWidth]);
        picture = cat(2,picture,newImage);    
    end
    
    imshow(picture)
    imwrite(picture, ['./' file '.png'])
end
    