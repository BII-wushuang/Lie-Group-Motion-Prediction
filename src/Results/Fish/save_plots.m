files = {'gt', 'erd', 'resgru', 'hmr'};
frames = [0,5,25,45,65,85];

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
    