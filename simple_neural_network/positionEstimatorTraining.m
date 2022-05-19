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
[N_trials_tr, N_angles] = size(training_data);
N_neurons = size(training_data(1).spikes, 1);

windowsize = 15;
t_mvt = 210;
t_pre_mvt = 300;
t_step = windowsize/2;
t_step = ceil(t_step);
proportion = 2/100; % th for selection of principal components

fprintf("\nFinding spike rates and velocities...");
[velx_tr, vely_tr, ~] = getvel2(training_data, windowsize, t_step, t_mvt); % one array with all three axis 
spike_rate = get_spike_rates2(training_data, windowsize, t_step, t_mvt);
fprintf("Spike_Rate done...\n");
toc;

%% TRAIN KNN MODEL

fprintf("Training KNN model...");
spikesr = zeros(N_angles*N_trials_tr, N_neurons);
labels = zeros(1, N_angles*N_trials_tr);
for k_it = 1:N_angles
    for n_it = 1:N_trials_tr
         spikesr( (k_it-1)*N_trials_tr + n_it, :) = sum(training_data(n_it, k_it).spikes(:, 1:t_pre_mvt), 2)';           
        labels( (k_it-1)*N_trials_tr + n_it) = k_it;
    end
end

knn = fitcknn(spikesr, labels, 'NumNeighbors',21);
for k_it = 1:N_angles+1
    modelParameters.KNNSpikesr = spikesr;
    modelParameters.KNNLabels = labels;
    modelParameters.knn = knn;
    modelParameters.n_neighbours = 21;
end
fprintf("KNN model done. "); toc;

%% TRAIN POSITION ESTIMATOR
fprintf("Extracting Principal component vectors from data...");

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

fprintf("Starting Neural Networks Training.\t");

for k_it = 0:N_angles
    fprintf("k=%g.\t", k_it);
    if (k_it ==0) % non-direction specific training
        [input_datax, output_datax] = linearizeInputOutput(principal_spikes_0, velx_tr, k_it);
        [input_datay, output_datay] = linearizeInputOutput(principal_spikes_0, vely_tr, k_it);

% % ------------------ Hyperoptimised parameters:  -------------------- % 
        x_activation = 'relu';
        y_activation = 'relu';
        x_lambda = 5e-04;
        y_lambda = 5e-04;
        x_standardize = false;
        y_standardize = false;
        x_layers = [10];
        y_layers = [8];
% % --------------------------------------------------------- % 


        mdl_x = fitrnet(input_datax, output_datax,'Activations', x_activation, 'Lambda', x_lambda, 'Standardize', x_standardize, 'LayerSizes', x_layers, 'ObservationsIn','columns');
        mdl_y = fitrnet(input_datay, output_datay,'Activations', y_activation, 'Lambda', y_lambda, 'Standardize', y_standardize, 'LayerSizes', y_layers, 'ObservationsIn','columns');
        

        k_it = N_angles+1;
    else  % direction specific training
        [input_datax, output_datax] = linearizeInputOutput(principal_spikes_tr, velx_tr, k_it);
        [input_datay, output_datay] = linearizeInputOutput(principal_spikes_tr, vely_tr, k_it);

        % model parameters from non-direction specific case used for all
        % directions
        mdl_x = fitrnet(input_datax, output_datax,'Activations', x_activation, 'Lambda', x_lambda, 'Standardize', x_standardize, 'LayerSizes', x_layers, 'ObservationsIn','columns');
        mdl_y = fitrnet(input_datay, output_datay,'Activations', y_activation, 'Lambda', y_lambda, 'Standardize', y_standardize, 'LayerSizes', y_layers, 'ObservationsIn','columns');

    end

    modelParameters(k_it).x = mdl_x;
    modelParameters(k_it).y = mdl_y;

end
fprintf("\n Done.\n");
fprintf("\nFinished Training.\n");
end



function [x_vel, y_vel, z_vel] = getvel2(trials , windowsize, t_step, t_start)

% x_vel: [max_t x N_trials x N_angles]

if ~exist('t_step', 'var') || isempty(t_step)
        t_step = 1;
elseif t_step <=0
    t_step = ceil(windowsize/2);
end
if ~exist('t_start', 'var') || isempty(t_start) || t_start<1
    t_start = 1;
end
  
t_start = t_start - 1; % offset for iteration

% make sure these are integers!
t_start = floor(max([t_start, 0]));
windowsize = ceil(windowsize) ;
t_step = round(max([t_step, 1]));

[N_trials, N_angles] = size(trials);
N_neurons = size(trials(1,1).spikes, 1);


x_vel = zeros(N_trials, N_angles, 2000); % make it bigger than estimated max size
y_vel = zeros(N_trials, N_angles, 2000);
z_vel = zeros(N_trials, N_angles, 2000);

max_t=0;
for n = 1:N_trials
    for k = 1:N_angles
        handPos = trials(n,k).handPos;
        timesteps = size(handPos, 2);
        if timesteps>max_t
            max_t = timesteps;
        end
        for t = t_start+windowsize:t_step:timesteps
            x_vel(n, k, ceil((t-t_start)/t_step)) = (handPos(1,t)- handPos(1,t-windowsize+1))/windowsize;
            y_vel(n, k, ceil((t-t_start)/t_step)) = (handPos(2,t)- handPos(2,t-windowsize+1))/windowsize;
            z_vel(n, k, ceil((t-t_start)/t_step)) = (handPos(3,t)- handPos(3,t-windowsize+1))/windowsize;
            
        end
    end

end


% crop extra zeros at the end
x_vel(:,:,max_t+1:end) = [];
y_vel(:,:,max_t+1:end) = [];
z_vel(:,:,max_t+1:end) = [];



end

function spike_rates = get_spike_rates2(trials, windowsize, t_step, t_start)
    % output size spike_rates: cell [N_trials x N_angles]
%       each is a double, size [N_neurons  x (t_max_each/t_step)]

% input trials - trial(n, k), spikes size(N_neurons, t_max)
%input t_step: not there = assume keep all time steps
% if t_step == 0 -> assume want default size reduction = windowsize/2


    [N_trials, N_angles]= size(trials);
    N_neurons = size(trials(1,1).spikes,1);
    
    
    if ~exist('t_step', 'var') || isempty(t_step)
        t_step = 1;
    elseif t_step <=0
        t_step = ceil(windowsize/2);
    end
    if ~exist('t_start', 'var') || isempty(t_start)
        t_start = 1;
    end
    
    t_start = t_start - 1; % offset for iteration
    
    % make sure these are integers!
    t_start = floor(max([t_start, 0]));
    windowsize = ceil(windowsize) ;
    t_step = ceil(max([t_step, 1]));

    spike_rates = cell(N_trials, N_angles);
    
    max_t = 0;

    for n = 1:N_trials
        for k = 1:N_angles
            spikes = trials(n,k).spikes;
            timesteps = size(spikes, 2);
            spike_rates{n, k} = zeros(N_neurons, ceil(timesteps/t_step));
            
            if timesteps>max_t
                max_t = timesteps;
            end
            
            for t = t_start+windowsize:t_step:timesteps
                rate = sum(spikes(:, t-windowsize+1:t), 2)/windowsize*1000;
                spike_rates{n, k}(:, ceil((t-t_start)/t_step)) = rate;

            end
        end
    end
    
end

function spike_rate_all_trials = make_av_spike_rate(spike_rates, dir)

% input format:
% neuronal spike rate data
    %   cell array {N_trials, k_dir} x [N_neurons x t_max_each]
% output:
%   spike_rate_all_trials, format [N_neurons  x t_max_all]
    
    [N_trials, N_angles] = size(spike_rates);
    N_neurons = size(spike_rates{1}, 1);
    
    if ~exist('dir', 'var') || isempty(dir) || dir == 0
        spike_rates = spike_rates(:); % no determined direction
        dir = 1;
    end
    if (size(spike_rates, 2) == 1)
        dir = 1; % deal with the function if direction already filtered out so no bug when accessing dir
    end
    
    % initialize memory allocation
    size_alloc = 2000;
    spike_rate_all_trials = zeros(N_neurons, size_alloc);
    
    max_t = 0;
    for n = 1:N_trials
        s = spike_rates{n, dir};
        timesteps = size(s, 2);
        if timesteps > max_t
            max_t = timesteps; % maximal time step
        end
        s(N_neurons, size_alloc) = 0; % zero pad
        
        spike_rate_all_trials  = spike_rate_all_trials + s;
    end
    spike_rate_all_trials(:, max_t+1:end)=[]; % reduce extra unused space
    
    spike_rate_all_trials = spike_rate_all_trials/N_trials;
end
function [principal_spikes, Vs, Ds, M] = spikes_PCA(spike_rate, cutoff)
    % function:
    % input:
    %    spike rates
    %       format: [N_neurons x t]
    %   cutoff - 0 < proportion < 1
    %       cutoff to decide how many significant eigenvalues
    % output:
    %   principal_spikes = spikes along principal component coordinates (reduced size)
    %   Vs = eigenvectors (full)
    %   Ds = eigenvalues  (full)
    %   M = number of principal elements conserved in spikes
    
    
    % check if cutoff on eigenvalues within correct range
    if cutoff<0 || cutoff >1
        cutoff = 0.2;
    end
    
    spikes_mean = mean(spike_rate, 2);

    A = cov(spike_rate'); % 98x98 covariance matrix
    [V,D] = eig(A);
    [d,ind] = sort(diag(D), 'descend');
    Ds = D(ind, ind);
    Vs = V(:, ind);
    
    
    Is = find(diag(Ds)<max(Ds,[],'all')*cutoff);
    M = Is(1);
    V_red = Vs(:,1:M); % principal component vectors

    principal_spikes = V_red'*(spike_rate - spikes_mean);
end

function [training_input, training_output] = linearizeInputOutput(training_input, training_output, dir)
    % axis: 1, 2, 3 for x, y, z respectively
    
    % does not assume prior knowledge on direction of motion
    % (if did assume previous knowledge we would train model with only
    % specific k_dir)
    
    % training input: neuronal spike rate data
    %   {N_trials, k_dir} x [N_neurons x t_max]
    % N_neurons might be a different value if using dimension reduction
    % training output: hand velocity in specific axis
    %   [N_trials, N_angle, t_max]
    
    % We can also input an averaged spike rate
    [N_trials, N_angles] = size(training_input);
    

    
    if ~exist('dir', 'var') || isempty(dir) || dir == 0
        % do we want to use prior assumption on direction of movemnet?
        % % if dir undefined: make cell vector linear(instead of 2D for each dir)!
        % and then make mean!!
        % otherwise select only dir valid (and will be linear vect)

        dir = 1;
        if size(training_input, 2)>1 % only resize if needed
            training_input = training_input(:); % no specific direction: linearize from [N_trials x N_angles] to [K x 1] cell array
        end
        % new format is a [N_trials*N_angles x 1] cell containing [M x t_max_each] matrices 
                
        if size(training_output, 2) > 1
            [M,N,P] = size(training_output);
            training_output = reshape(training_output, [N*M, P]);
                % new size: vel_axis = [N_trial*N_angles x max_t] 
        end        
    else    % we have a defined desired direction and make linear
        if size(training_input, 2)> 1 % select correct dir only if needed
            training_input = training_input(:, dir);
                % new size: [N_trials x 1] cell, [N_neurons x t_max_each]
        end
        if size(training_output, 2) > 1 % select correct dir only and squeeze matrix shape
            training_output = squeeze(training_output(:, dir, :));
                % new size: [N_trials x max_t]
        end
    end
    
    if (size(training_input, 1) ~= size(training_output, 1))
        error("wrong size!\nSize output:%g\nSize input: %g", size(training_output,1), size(training_input,2));
    end
    N_neurons = size(training_input{1}, 1);
    t_max_all = size(training_output, 2);
    
    % linearize arrays in time and trials
    training_input2 = zeros(N_neurons, N_trials*t_max_all);


    for n=1:size(training_input,1)
        spikes_trial = training_input{n};
        spikes_trial(N_neurons, t_max_all) = 0; % zeros padding
        training_input2(:, 1+(n-1)*t_max_all : n*t_max_all) = spikes_trial;
    end
    training_output = reshape(training_output', [size(training_output,1)*t_max_all, 1]);
    % flatten array back to back

    if (length(training_output) ~= size(training_input2, 2))
        error("wrong size!\nSize output:%g\nSize input: %g", size(training_output, 1), size(training_input2, 2));
    end
    
    training_input = training_input2;
    training_output = training_output';
end
