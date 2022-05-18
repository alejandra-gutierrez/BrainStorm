%% 
training_data = load('monkeydata_training.mat');
training_data = training_data.trial;
training_data = training_data(1:80, :);

[trials,angle]=size(training_data);

spikes_train = zeros(640,98);
direction_train = zeros(640,1);

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

%%

validation_data = load('monkeydata_training.mat');
validation_data = validation_data.trial;
validation_data = validation_data(81:100, :);

[trials,angle]=size(validation_data);

spikes_val = zeros(160,98);
direction_val = zeros(160,1);

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

%%

knn = fitcknn(spikes_train,direction_train, 'NumNeighbors', 21);
ypred = predict(knn,spikes_val);

s = ypred==direction_val; %this is a boolean 
    similarity = sum(s)/numel(s); 



%%
expected_angle = [];
for angle = 1:8
    for a = 1:20
        expected_angle = [expected_angle, angle];
    end
end

%%
expected_angle = transpose(expected_angle);
error = [];
neighbour_range = 1:10:300;
for num_neighbours = neighbour_range
    knn = fitcknn(spikes_train,direction_train, 'NumNeighbors', num_neighbours);
    modelParameters.knn=knn;

    ypred = predict(knn,spikes_val);
    s = ypred==expected_angle; %this is a boolean vector that will be 1 if the entries are the same and 0 if different
    similarity = sum(s)/numel(s);

    error = [error, similarity];
    aaa = [ypred, expected_angle];
end

% 
plot(neighbour_range,error)

