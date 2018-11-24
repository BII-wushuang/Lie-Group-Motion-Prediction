% Evaluate the prediction based on the mean joint error metric as well as
% the mean angle error metric (both axis angle geodesic and euler l2)

function [] = Evaluation()
    dataset = 'Human';
    dir_name = ['./output/' dataset];

    processfolder(dir_name);
end

function [] = processfolder(folder)
    actions = {'directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting', 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether'};
    out_frames = [1, 3, 7, 9, 13, 15, 17, 24]+1;
    dir_name = dir(folder);
    dir_name = dir_name(3:end);
    for i = 1:length(dir_name)
        if (~dir_name(i).isdir)
            continue
        end
        
        folder = [dir_name(i).folder '/' dir_name(i).name];
        if (isempty(dir([folder '/*.mat'])))
            processfolder(folder);
        else
            idx = strfind(folder, '/');
            folder_name = folder(idx(end-2)+1:idx(end-1)-1);
            fprintf('%s \n', folder_name);
            filename = [folder '/Errors.csv'];
            all_errors = fopen([folder '/Errors.csv'], 'w');
            fprintf('\n');
            for k = 1:length(actions)
                gt_xyz_list = dir([folder '/gt_xyz_' actions{k} '_*.mat']);
                gt_lie_list = dir([folder '/gt_lie_' actions{k} '_*.mat']);
                pred_xyz_list = dir([folder '/prediction_xyz_' actions{k} '_*.mat']);
                pred_lie_list = dir([folder '/prediction_lie_' actions{k} '_*.mat']);
                if isempty(gt_xyz_list)
                    continue
                end

                gt_xyz = importdata([gt_xyz_list(1).folder '/' gt_xyz_list(1).name]);
                nframes = size(gt_xyz, 1);
                frames = (1:nframes)';
                l2_metric = zeros([nframes,1]);
                angle_metric = zeros([nframes,1]);
                euler_metric = zeros([nframes,1]);
                nfiles = length(gt_xyz_list);

                for j = 1:nfiles
                    gt_xyz = importdata([gt_xyz_list(j).folder '/' gt_xyz_list(j).name]);
                    gt_lie = importdata([gt_lie_list(j).folder '/' gt_lie_list(j).name]);
                    pred_xyz = importdata([pred_xyz_list(j).folder '/' pred_xyz_list(j).name]);
                    pred_lie = importdata([pred_lie_list(j).folder '/' pred_lie_list(j).name]);

                    l2_metric = l2_metric + l2(gt_xyz,pred_xyz);
                    angle_metric = angle_metric + angle(gt_lie,pred_lie);
                    euler_metric = euler_metric + euler(gt_lie,pred_lie);
                end
                l2_metric = l2_metric./nfiles;
                angle_metric = angle_metric./nfiles;
                euler_metric = euler_metric./nfiles;

                filename = [gt_xyz_list(1).folder '/Errors_' actions{k} '.csv'];
                fileID = fopen(filename,'w');
                fprintf(fileID,'%s , %s , %s, %s \n', 'Frame #', 'Mean Joint Error', 'Mean Angle Error', 'Mean Angle Error (Euler)');
                fclose(fileID);
                dlmwrite(filename, [frames, l2_metric, angle_metric, euler_metric], '-append');

            end
            fclose(all_errors);
        end
    end

end

function l2_metric = l2(gt, pred)
    A = gt - pred;
    A = A.^2;
    l2_metric = sqrt(sum(A,3));
    l2_metric = sum(l2_metric,2)/size(l2_metric,2);
end

function angle_metric = angle(gt, pred)
    try
        gt = reshape(gt, [size(gt,1), size(gt,2)/3, 3]); 
        pred = reshape(pred, [size(pred,1), size(pred,2)/3, 3]);
    catch
        
    end
    nframes  = size(gt,1);
    njoints = size(gt,2);
    angle_metric = zeros([nframes, 1]);
    for i = 1: nframes
        for j = 1: njoints
            R = rotmat(squeeze(gt(i,j,:)))'*rotmat(squeeze(pred(i,j,:)));
            angle_metric(i) = angle_metric(i) + (acos((trace(R)-1)/2))^2;
        end
    end
    angle_metric = sqrt(angle_metric)/njoints;
end

function R = rotmat(v)
    v_norm = norm(v);
    v = v/v_norm;
    if (v_norm<1e-6)
        R = eye(3);
    else
        v_cross = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
        R = eye(3) + sin(v_norm)*v_cross + (1-cos(v_norm))*v_cross*v_cross;
    end
end

function euler_metric = euler(gt, pred)
    nframes  = size(gt,1);
    njoints = size(gt,2);
    euler_metric = zeros([nframes, 1]);
    gt(:,1:6) = 0;
    pred(:,1:6) = 0;
    for i = 1: nframes
        for j = 4: 3: njoints
            gt_R = rotmat(squeeze(gt(i,j:j+2)));
            pred_R = rotmat(squeeze(pred(i,j:j+2)));
            euler_metric(i) = euler_metric(i) + sum(RotMat2Euler(gt_R) - RotMat2Euler(pred_R))^2;
        end
    end
    euler_metric = sqrt(euler_metric);
end

function Eul = RotMat2Euler(R)
    if R(1,3) == 1 || R(1,3) == -1
      %special case
      E3 = 0; %set arbitrarily
      dlta = atan2(R(1,2),R(1,3));
      if R(1,3) == -1
        E2 = pi/2;
        E1 = E3 + dlta;
      else
        E2 = -pi/2;
        E1 = -E3 + dlta;
      end
    else
      E2 = - asin(R(1,3));
      E1 = atan2(R(2,3)/cos(E2), R(3,3)/cos(E2));
      E3 = atan2(R(1,2)/cos(E2), R(1,1)/cos(E2));
    end

    Eul = -[E1 E2 E3];
end