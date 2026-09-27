%% Q-learning with epsilon-greedy exploration Algorithm for stochastic Cleaning Robot V1
%% The stochastic cleaning-robot MDP
% a cleaning robot has to collect a used can also has to recharge its
% batteries. the state describes the position of the robot and the action
% describes the direction of motion. The robot can move to the left or to
% the right. The first (1) and the final (6) states are the terminal
% states. The goal is to find an optimal policy that maximizes the return
% from any initial state. Here the Q-learning epsilon-greedy exploration
% algorithm (in Reinforcement learning) is used.
% Algorithm 2-3, from:
% @book{busoniu2010reinforcement,
%   title={Reinforcement learning and dynamic programming using function approximators},
%   author={Busoniu, Lucian and Babuska, Robert and De Schutter, Bart and Ernst, Damien},
%   year={2010},
%   publisher={CRC Press}
% }
% notice: the code is written in 1-indexed instead of 0-indexed
%
% V1 the initial evaluation of the algorithm 
%
%% this is the main function including the initialization and the algorithm
% the inputs are: initial Q matrix, set of actions, set of states,
% discounting factor, learning rate, exploration probability,
% number of iterations, and the initial state.
function stochastic_robot_cleaning_q
% learning parameters
gamma = 0.9;    % discount factor  % TODO : we need learning rate schedule
alpha = 0.5;    % learning rate    % TODO : we need exploration rate schedule
epsilon = 0.9;  % exploration probability (1-epsilon = exploit / epsilon = explore)
% states
state = [1,2,3,4,5,6];
% actions
action = [-1,1];
% initial Q matrix
Q = zeros(length(state),length(action));
K = 10^4;     % maximum number of the iterations
K = 10^2;     % maximum number of the iterations (sometimes does not converge)
state_idx = 3;  % the initial state to begin from
%% the main loop of the algorithm
for k = 1:K
    disp(['iteration: ' num2str(k)]);
    r=rand; % get 1 uniform random number
    x=sum(r>=cumsum([0, 1-epsilon, epsilon])); % check it to be in which probability area
    
    % choose either explore or exploit
    if x == 1   % exploit
        [~,umax]=max(Q(state_idx,:));
        current_action = action(umax);
    else        % explore
        current_action=datasample(action,1); % choose 1 action randomly (uniform random distribution)
    end
    
    action_idx = find(action==current_action); % id of the chosen action
    % observe the next state and next reward ** there is no reward matrix
    [next_state,next_reward] = model(state(state_idx),action(action_idx));
    next_state_idx = find(state==next_state);  % id of the next state
    % print the results in each iteration
    disp(['current state : ' num2str(state(state_idx)) ' next state : ' num2str(state(next_state_idx)) ' taken action : ' num2str(action(action_idx))]);
    disp([' next reward : ' num2str(next_reward)]);
    % update the Q matrix using the Q-learning rule
    Q(state_idx,action_idx) = Q(state_idx,action_idx) + alpha * (next_reward + gamma* max(Q(next_state_idx,:)) - Q(state_idx,action_idx));

    % if the robot is stuck in terminals
    if (next_state_idx == 6 || next_state_idx == 1)
        state_idx = datasample(2:length(state)-1,1); % we just restart the episode with a new state
    else
        state_idx = next_state_idx;
    end
    disp(Q);  % display Q in each level
end
% display the final Q matrix
disp('Final Q matrix : ');
disp(Q)
[V,I]=max(Q,[],2);                              % finding the max values
disp('V(s):');
disp(V);
disp('Optimal Policy');
disp('*');
disp([action(I(2,1));action(I(3,1));action(I(4,1));action(I(5,1))]);
disp('*');
end

%% This function is used as an observer to give the next state and the next reward using the current state and action
function [x_next,r_next] = model(x,u)
if (x == 1 || x == 6)
    xp = [x x x];
else
    xp = [x-u x x+u];
end
p = [prob(x,u,x-u) prob(x,u,x) prob(x,u,x+u)];   %p(x_next|x,u)
r = [reward(x,u,x-u) reward(x,u,x) reward(x,u,x+u)];

[x_next, idx] = datasample(xp,1,'Weights',p);        %drawn x_next from p(x_next|x,u)
r_next = r(idx); 

end
%% this function is the transition model of the robot
% the inputs are: the current state, the chosen action, and the next desired state
% the output is the probability for going to the next state considering the stochasity
%
% ** in the deterministic case the model gives the next state
% ** but in the stochastic case the model gives the probability for going
% to the next state
function f = prob(x,u,xp)
if (x == 1 || x == 6)
    if (xp == x)
        f = 1;
    else
        f = 0;
    end
elseif (x <= 5 && x>=2)
    if (xp == x)
        f = 0.1;
    elseif (xp == x + u)
        f = 0.8;
    elseif (xp == x - u)
        f = 0.1;
    else
        f = 0;
    end
end
end

%% this function is the reward function for the task (stochastic)
% the inputs are: the current state, the chosen action, and the next state
% the output is the expected reward in the next state
% the reward actually doesn't depend on the chosen action, in this case
function r = reward(x,u,xp)
if (x ~= 6 && xp == 6)
    r = 5; %check randomness of final outcome
elseif (x ~= 1 && xp == 1)
    r = 1;
else
    r = -1;
end
end 



