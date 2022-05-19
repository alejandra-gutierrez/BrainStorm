%% population encoding
clear;

%% start
data = load('monkeydata_training.mat');
trial = data.trial;

t_pre_mvt = 150;

N_trials = size(trial, 1);
N_angles = size(trial, 2);
N_neurons = size(trial(1,1).spikes, 1);

k_list = 1:N_angles;
theta = (40*k_list-10)/180*pi;
unit_vect_list = [cos(theta); sin(theta)];

ix = randperm(length(trial));
trainingData = trial(ix(1:80), :);
N_trials_tr = size(trainingData, 1);
testData = trial(ix(81:end), :);
N_trials_test = size(testData, 1);

spike_rates_list = zeros(N_trials_tr, N_neurons);
mean_sr = zeros(N_angles, N_neurons);
for k_it = 1:N_angles
    for n_it = 1:N_trials_tr
        spikes = trainingData(n_it, k_it).spikes(:, 1:t_pre_mvt);
        sr = sum(spikes, 2)/t_pre_mvt;
        spike_rates_list(n_it, :) = sr(:);
    end
    mean_sr(k_it, :) = sum(spike_rates_list, 1);
end

%we have obtained tuning curves mean_sr

figure;
plot(mean_sr(:, 50));
[sr_max, ind_sr_max] = max(mean_sr);
[sr_min, in_sr_min] = min(mean_sr);
prefered_vectors = unit_vect_list(:, ind_sr_max);

%% testing
trial_1 = testData(1,1);


sr = sum(trial_1.spikes(:, 1:t_pre_mvt), 2)/t_pre_mvt;

dir_vector = prefered_vectors*((sr-sr_min)/(sr_max-sr_min));
dir_vector = dir_vector/norm(dir_vector);

theta_test = atan2(dir_vector(2),dir_vector(1));
if theta_test<0
    theta_test = theta_test+2*pi;
end
angle_num_test = round((180/pi*theta_test+10)/40);

if angle_num_test < 0
    angle_num_test = angle_num_test+ 8;
elseif angle_num_test>8
    angle_num_test = angle_num_test - 8;
end

%% testing many
figure; hold on;

t_start =1;
t_pre_mvt = 150;

for k_it = 1:N_angles
    for n_it = 1:N_trials_test
        labels(n_it, k_it) = k_it;
        trial_1 = testData(n_it, k_it);
        sr = sum(trial_1.spikes(:, t_start:t_pre_mvt), 2)/(t_pre_mvt-t_start);
%         dir_vector = prefered_vectors*(sr);

        dir_vector = prefered_vectors*(sr./sr_max);
       dir_vector= dir_vector/norm(dir_vector);
        
        theta_test(n_it, k_it) = atan2(dir_vector(2),dir_vector(1));
        if theta_test(n_it, k_it)<0
            theta_test(n_it, k_it) = theta_test(n_it, k_it)+2*pi;
        end
        angle_num_test(n_it, k_it) = ((180/pi*theta_test(n_it, k_it)+10)/40);
        
        if angle_num_test(n_it, k_it) < 0.5
            angle_num_test(n_it, k_it) = angle_num_test(n_it, k_it)+ 8;
        elseif angle_num_test(n_it, k_it)>8.5
            angle_num_test(n_it, k_it) = angle_num_test(n_it, k_it) - 8;
        end
       
        
    end
    plot(labels(:, k_it), angle_num_test(:, k_it), '*');
end

xlim([0,9]);
ylim([0,9]);
xlabel("true label");
ylabel("Estimated label");

title("Classification of reaching angles using population decoding");

error = sqrt(mean((abs(angle_num_test(:) - labels(:)).^2)));
figure;
confusionchart(labels(:), round(angle_num_test(:)))

%% estimate error for different values of t_pre_mvt
t_pre_mvt_list = 20:10:301;
err_list =zeros(size(t_pre_mvt_list));

for i =1:length(t_pre_mvt_list)
        t_pre_mvt = t_pre_mvt_list(i);

    for k_it = 1:N_angles
        for n_it = 1:N_trials_test
            labels(n_it, k_it) = k_it;
            trial_1 = testData(n_it, k_it);
            sr = sum(trial_1.spikes(:, 1:t_pre_mvt), 2)/t_pre_mvt;
            dir_vector = prefered_vectors*(sr);
    
%             dir_vector = prefered_vectors*((sr)./(sr_max));
           dir_vector= dir_vector/norm(dir_vector);
            
            theta_test(n_it, k_it) = atan2(dir_vector(2),dir_vector(1));
            if theta_test(n_it, k_it)<0
                theta_test(n_it, k_it) = theta_test(n_it, k_it)+2*pi;
            end
            angle_num_test(n_it, k_it) = ((180/pi*theta_test(n_it, k_it)+10)/40);
            
            if angle_num_test(n_it, k_it) < 0
                angle_num_test(n_it, k_it) = angle_num_test(n_it, k_it)+ 8;
            elseif angle_num_test(n_it, k_it)>8
                angle_num_test(n_it, k_it) = angle_num_test(n_it, k_it) - 8;
            end
           
            
        end
    end
    err_list(i) = sqrt(mean((abs(round(angle_num_test(:)) - labels(:)).^2)));
end

figure;
plot(t_pre_mvt_list, err_list, '.');

err_smooth = smooth(err_list);
hold on;
plot(t_pre_mvt_list, err_smooth);
title("Classification error using population encoding before movement starts", "using averages from t=1 to t_c")
xlabel('time t_c (ms) since recording started');
ylabel('Classification error');

%accuracy
non_true_list = ~(labels(:)==round(angle_num_test(:)));
wrongpercent =mean(double(non_true_list));

