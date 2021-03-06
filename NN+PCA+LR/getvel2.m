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
            x_vel(n, k, ceil((t-t_start)/t_step)) = (handPos(1,t)- handPos(1,t-windowsize+1))/windowsize*2;
            y_vel(n, k, ceil((t-t_start)/t_step)) = (handPos(2,t)- handPos(2,t-windowsize+1))/windowsize*2;
            z_vel(n, k, ceil((t-t_start)/t_step)) = (handPos(3,t)- handPos(3,t-windowsize+1))/windowsize*2;

        end
    end

end


% crop extra zeros at the end
% x_vel(max_t+1:end,:,:) = [];
% y_vel(max_t+1:end,:,:) = [];
% z_vel(max_t+1:end,:,:) = [];
x_vel(:,:,max_t+1:end) = [];
y_vel(:,:,max_t+1:end) = [];
z_vel(:,:,max_t+1:end) = [];



end
