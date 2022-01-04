files = {'gt', '3lr', 'erd', 'gru', 'hmr'};
titles = {'Ground Truth', 'LSTM-3LR (2015)', 'ERD (2015)', 'Res-GRU (2017)', 'HMR (Ours)'};

fig = figure(1);
fig.Position = [0 0 1920 1080];
fig.Resize = 'on';
traj(1) = getframe(fig);

for i = 1: 75
    for j = 1: length(files)    
        subplot(2,3,j)
        picture = [];
        file = files{j};
        img = imread(['./' file '/output_' num2str(i) '.png'], 'png');
        imshow(img);
        title(titles{j}, 'FontSize', 16);
    end
    
    axes('Position', [0, 0.95, 1, 0.05]);
    set(gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None');
    text(0.5, 0, 'Forecasting for Mouse', 'FontSize', 22, 'FontWeight', 'Bold', 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom');
    drawnow
    traj(i) = getframe(fig);
end

myVideo = VideoWriter('./Mouse.mp4', 'MPEG-4');
myVideo.FrameRate = 10;
myVideo.Quality = 100;
open(myVideo);
writeVideo(myVideo, traj);
close(myVideo);