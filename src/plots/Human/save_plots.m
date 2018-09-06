files = {'gt', 'erd', '3lr', 'resgru', 'xyz', 'hmr'};
frames = [25, 50, 75, 100, 110, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1250];

for j = 1: length(files)
    picture = [];
    file = files{j};

    for i = frames
        if i == 110
            [oldHeight, oldWidth, oldNumberOfColorChannels] = size(newImage);
            newWidth = int32(oldHeight * 1/5);
            newImage = imresize(img, [oldHeight newWidth]);
            newImage = ones(size(newImage))*255;
            picture = cat(2,picture,newImage);
        else
            img = imread(['./' file '/output_' num2str(i) '.png'], 'png');
            img = imcrop(img,[90.5 85.5 271 282]);
            [oldHeight, oldWidth, oldNumberOfColorChannels] = size(img);
            newWidth = int32(oldHeight * 2.8/4);
            newImage = imresize(img, [oldHeight newWidth]);
            picture = cat(2,picture,newImage);
        end
    end

    imshow(picture)
    imwrite(picture, ['./walking_' file '.png'])
end