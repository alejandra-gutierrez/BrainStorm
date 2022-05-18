function [modelParameters] = positionEstimatorTraining(training_data)
% Arguments:

% - training_data:
%     training_data(n,k)              (n = trial id,  k = reaching angle)
%     training_data(n,k).trialId      unique number of the trial
%     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
%     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

% ... train your model

% Return Value:

% - modelParameters:
%     single structure containing all the learned parameters of your
%     model and which can be used by the "positionEstimator" function.

t_start = tic;
tic;
[N_trials_tr, N_angles] = size(training_data);
N_neurons = size(training_data(1).spikes, 1);

windowsize = 15;
t_mvt = 210;
t_pre_mvt = 300;
t_step = windowsize/2;
t_step = ceil(t_step);
proportion = 2/100; % th for selection of principal components

fprintf("\nFinding spike rates and velocities...");
[velx_tr, vely_tr, ~] = getvel2(training_data, windowsize, t_step, t_mvt);
spike_rate = get_spike_rates2(training_data, windowsize, t_step, t_mvt);
fprintf("Spike_Rate done...\n");
toc;

%% TRAIN NN MODEL

spikes_train = zeros(640,98);
direction_train = zeros(640,1);
[trials,angle]=size(training_data);

neurons=length(training_data(1,1).spikes(:,1));
spike_angle = zeros(trials,neurons);

run_no = 0;
for a = 1:angle
    for t = 1:trials
        run_no = run_no + 1;
        direction_train(run_no) = a;
        for n = 1:neurons
            spikes_train(run_no,n) = sum(training_data(t,a).spikes(n,1:320)); % Feature reduction %Number of spikes in neuron in all time
        end
        direction_train(run_no) = a;
    end
end

r1 = randperm(640,640);
spikes_train_s = spikes_train(r1, :);
direction_train_s = direction_train(r1, :);


validation_data = load('monkeydata_training.mat');
validation_data = validation_data.trial;
validation_data = validation_data(81:100, :);

[trials,angle]=size(validation_data);

spikes_val = zeros(160,98);
direction_val= zeros(160,1);
neurons=length(validation_data(1,1).spikes(:,1));
spike_angle = zeros(trials,neurons);

run_no = 0;
for a = 1:angle
    for t = 1:trials
        run_no = run_no + 1;
        direction_val(run_no) = a;
        for n = 1:neurons
                spikes_val(run_no,n) = sum(validation_data(t,a).spikes(n,1:320)); % Feature reduction %Number of spikes in neuron in all time
        end
        direction_val(run_no) = a;
    end
end

ValidationData = {};
ValidationData{1} = spikes_val;
ValidationData{2} = direction_val;

nn = fitcnet(spikes_train, direction_train, 'ValidationData', ValidationData, 'ValidationPatience', 6, 'LayerSizes', 100);

for k_it = 1:N_angles+1
    modelParameters(k_it).nn = nn

end
fprintf("NN model done. "); toc;

%% TRAIN POSITION ESTIMATOR
fprintf("Extracting Principal component vectors from data...");

%   spike_rate_av_trials = make_av_spike_rate(spike_rate);
%   [principal_spikes, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, 0.05);
%   principal_spikes_tr = cell(N_trials_tr, N_angles+1);
% 
%   modelParameters(9).M = M;
%   modelParameters(9).dir = 0;
%   modelParameters(9).Vs = Vs;
%   modelParameters(9).Ds = Ds;
%   modelParameters(9).V_red = Vs(:,1:M);


for k_it =0:N_angles
    spike_rate_av_trials = make_av_spike_rate(spike_rate, k_it);
    [~, Vs, Ds, M] = spikes_PCA(spike_rate_av_trials, proportion);
    dir = k_it;
    if k_it == 0
        k_it = N_angles+1;
    end
    V_red = Vs(:, 1:M);
    modelParameters(k_it).M = M; % keep same M for all
    modelParameters(k_it).dir = dir;
    modelParameters(k_it).Vs = Vs;
    modelParameters(k_it).Ds = Ds;
    modelParameters(k_it).V_red = V_red;
    modelParameters(k_it).MdlnetX = [];
    modelParameters(k_it).MdlnetY = [];

    for n_it = 1:N_trials_tr
        if k_it == N_angles+1
            for k = 1:N_angles
                % make a specific array for non-specific training
                spikes_mean = mean(spike_rate{n_it, k}, 2);
                principal_spikes_0{n_it, k} = V_red'*(spike_rate{n_it, k}-spikes_mean);
            end
        else
           spikes_mean = mean(spike_rate{n_it, k_it}, 2);
           principal_spikes_tr{n_it, k_it} = V_red'*(spike_rate{n_it, k_it}-spikes_mean);
        end
        fprintf(".");
    end
    fprintf("\n"); 
end
fprintf("Extracted PCA parameters.\n"); toc;

fprintf("Starting Linear Regression.\t");

for k_it = 0:N_angles
    fprintf("k=%g.\t", k_it);
    if (k_it ==0) % non-direction specific training
        [input_datax, output_datax] = linearizeInputOutput(principal_spikes_0, velx_tr, k_it);
        [input_datay, output_datay] = linearizeInputOutput(principal_spikes_0, vely_tr, k_it);

        PCA_components_weights_x = linearRegression(input_datax, output_datax);
        PCA_components_weights_y = linearRegression(input_datay, output_datay);

%         MdlnetX = fitrnet(input_datax, output_datax);
%         MdlnetY = fitrnet(input_datay, output_datay);
        k_it = N_angles+1;
    else  % direction specific training
        [input_datax, output_datax] = linearizeInputOutput(principal_spikes_tr, velx_tr, k_it);
        [input_datay, output_datay] = linearizeInputOutput(principal_spikes_tr, vely_tr, k_it);
        PCA_components_weights_x = linearRegression(input_datax, output_datax);
        PCA_components_weights_y = linearRegression(input_datay, output_datay);
%         MdlnetX = fitrnet(input_datax, output_datax);
%         MdlnetY = fitrnet(input_datay, output_datay);
    end
    
    modelParameters(k_it).PCAweightsX = PCA_components_weights_x;
    modelParameters(k_it).PCAweightsY = PCA_components_weights_y;
%     modelParameters(k_it).MdlnetX = MdlnetX;
%     modelParameters(k_it).MdlnetY = MdlnetY;

end
fprintf("\n Done.\n");
fprintf("Model Parameters:\n");
% print model parameters
for k_it = 1:N_angles+1
    M = modelParameters(k_it).M;
    dir = modelParameters(k_it).dir;
    V_red = modelParameters(k_it).V_red;
    Vs = modelParameters(k_it).Vs;
    Ds = modelParameters(k_it).Ds;
    wX = modelParameters(k_it).PCAweightsX;
    wY = modelParameters(k_it).PCAweightsY;
    fprintf("dir=%g, M=%g,  size V_red=[%g, %g], size wX=[%g,%g], size wY=[%g,%g]\n",...
    dir, M, size(V_red,1),size(V_red,2), size(wX,1), size(wX, 2), size(wY,1), size(wY, 2));
end
fprintf("\nFinished Training.\n");
toc; fprintf("\n");

end
