%% 
training_data = load('monkeydata_training.mat');
training_data = training_data.trial;
training_data = training_data(1:80, :);
rng(0)
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

ypred = predict(nn,spikes_val);

s = ypred==direction_val; % This is a boolean vector that will be 1 if the entries are the same and 0 if different
similarity = sum(s)/numel(s)


figure;
confusionchart(direction_val, ypred)
figure;
plot(nn.TrainingHistory.Iteration, nn.TrainingHistory.ValidationLoss)
