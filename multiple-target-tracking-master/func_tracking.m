function [] = func_tracking(input_file_path)
%FUNC_TRACKING Summary of this function goes here
%   Detailed explanation goes here

    addpath('/Users/andres.perez/source/DCASE2020/multiple-target-tracking-master/matlab_packages/rbmcda/')
    addpath('/Users/andres.perez/source/DCASE2020/multiple-target-tracking-master/matlab_packages/ekfukf/')

    %% I/O

%     Y_gt = csvread('/Users/andres.perez/source/DCASE2020/APRI/filter_input_gt/mi_primerito_dia_postfilter_Q!/fold1_room1_mix027_ov2.csv')';
%     Y = csvread('/Users/andres.perez/source/DCASE2020/APRI/filter_input/mi_primerito_dia_postfilter_Q!/fold1_room1_mix027_ov2.csv')';

    Y = csvread(input_file_path)';
    [filepath,name,~] = fileparts(input_file_path);
    output_file = strcat(filepath,'/',name,'.mat');
%     output_file = '/Users/andres.perez/source/DCASE2020/APRI/filter_output/fold1_room1_mix027_ov2.mat';

    spatial_resolution = 1;
    % % Tuned variables
    V_azi = 20;
    V_ele = 10;
    in_sd = 5;
    in_sdn = 50;
    in_cp = 0.25;

    T = Y(1, 1:end);
    Y = Y(2:3, 1:end)*spatial_resolution;
    %% PARAMETERS

    dt = 0.1; % Hop length of frames

    % Sound signal prior - statistics of the sound sources. 
    % We tell the tracker where all a sound source can exist in the spatial 
    % grid, and how fast can it move around.    

    M0 = [0;0;0;0]; % M0(1, 2) Mean position of the sound sources in the 
                    % respective 2D axes. 
                    % We are assuming the sources have a mean azimuth and 
                    % elevation of 0 and 0 respectively.
                    % M0(3, 4) Mean velocity of the sound sources in the 
                    % respective 2D axes. 
                    % We are assuming the overall velocity average along 
                    % azimuth and elevation is 0 and 0 respectively.

    P0 = diag([360^2 180^2 V_azi^2 V_ele^2]); 
        % P0(0, 1) is the variance of the signal along azimuth and elevation.
        % p0(0)= 360 means that the sound sources can exist anywhere in 0:360
        % degrees in azimuth and similarly 0:180 for elevation P(1)
        %
        % P0(2, 3) is the derivative/velocity of the signal in azimuth and 
        % elevation respectively. For example if a sound source moves 50 degrees 
        % in roughly 40 frames( 8 seconds) then P0(2) has to be set to 50/8 = 6
        % 
        % IDEALLY the velocity along the two axes will have to be tuned on your
        % dataset


    % Noise prior
    sd = in_sd; % standard deviation of measurement noise - [1 50] range is good
               % Approximately the width of Y axis data spread. 
               % This says that any DOA in the range of gt +/- sd belongs to the
               % same track. Lets say that the gt of the current track is 30 deg
               % and sd is 5, then if the DOA in the next frame is in the range of
               % 30-5 and 30+5 it will be part of the current track.
               % When two sources are very close by, if sd is large it can merge
               % this sources into one, and the final DOA track will be the average
               % of these two tracks.

    R = diag([sd^2 (sd/2)^2]);

    % noise spectral density along x and y axis
    % q along with sd decides how smooth the tracked signal is. 
    qx = in_sdn;
    qy = qx/2;

    % Probability of birth and death - Tuning not mandatory
    init_birth = 0.1; % value between [0 1] - Prior probability of birth 
    alpha_death = 1; % always >= 1; 1 is good  
    beta_death = 1; % always >= 1; 1 is good 


    % Prior probabilities of noise
    cd = 1/(360*180);      % Density of noise [max_azi - min_azi] *[max_ele - min_ele]
                           % Assuming the noise is uniformly distributed in the entire spatial grid

    cp = in_cp;      % Noise prior - estimate of percentage of noise in the 
                     % measurement data


    % Initialize filter
    N = 100; %Number of Monte Carlo samples - [10 100] range good
    S = kf_nmcda_init(N,M0,P0,dt);


    % Dont need to tune this.    
    F = [0 0 1 0;
        0 0 0 1;
        0 0 0 0;
        0 0 0 0];
    [A,Q] = lti_disc(F,[],diag([0 0 qx qy]),dt);
    H = [1 0 0 0; 0 1 0 0];

    %% TRACKING SCRIPT STARTS
    % Tracking unknown number of sources
    SS = cell(size(Y,2),size(S,2));
    for k=1:size(Y,2)
        S = kf_nmcda_predict_dp(S,A,Q,[],[],T(k),alpha_death,beta_death);
        [S,E] = kf_nmcda_update_dp(S,Y(:,k),T(k),H,R,cp,cd,init_birth);

        fprintf('%d/%d: %s\n',k,size(Y,2),E{1});

        SS(k,:) = S;

        %
        % Resample if needed
        %
        S = normalize_weights(S);
        W = get_weights(S);
        n_eff = eff_particles(W);
        if n_eff < N/4
            ind = resampstr(W);
            S = S(ind);
            SS = SS(:,ind);
            W = ones(1,N)/N;
            S = set_weights(S,W);
            fprintf('Resampling done on time step %d\n',k);
        end
    end
    [FM,FP,SM,SP,Times] = kf_nmcda_collect(SS,A,Q);


    %% VISUALIZATON
%     ttt = cell(size(Times));
%     for j=1:size(Times,1)
%         for k=1:size(Times,2)
%         ttt{j,k} = Times{j,k} *  1.7668; % moving
%         end
%     end
% 
%     ttt = Times

    %% VISUALIZATON

    % Visualize Azimuth and elevation separately
    % for isazi=1:2

    c = 1;
%     isazi = 1;
%     figure,
%     hold on;
%     h = plot(Y_gt(1, :), Y_gt(1+isazi,:)*spatial_resolution, 'ro');        
%     h=plot(T,Y(isazi,:),'bx');
%     set(h,'markersize',2);    

    tmp = [SS{end,:}];
    W = [tmp.W];
    [~,ind] = max(W);
    for j=1:size(SM,1)
        if ~isempty(SM{j,ind})
            t = Times{j,ind};
%             t = ttt{j,ind};
            m = SM{j,ind};

            tracks(c).t = t;        
            tracks(c).m = m;
            c = c +1;

%             h=plot(t,m(isazi,:),'g-');
%             set(h(1),'linewidth',2);
        end
    end
%     grid on;    
%     h = xlabel('Time (seconds) -->');    
%     if isazi == 1
%         h = ylabel('Azimuth (degree) -->');
%         h = title('AZIMUTH - RED : Groundtruth, BLUE: Predicted, GREEN: Tracked');
%         ylim([0 360])
%     else        
%         h = ylabel('Elevation (degree) -->');
%         h = title('ELEVATION - RED : Groundtruth, BLUE: Predicted, GREEN: Tracked');
%         ylim([0 180])
%     end

    save(output_file, 'tracks');
end

