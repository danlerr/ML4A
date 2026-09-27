%% Model-based value iteration Algorithm for Deterministic Cleaning Robot
%% The deterministic cleaning-robot MDP
% a cleaning robot has to collect a used can also has to recharge its
% batteries. the state describes the position of the robot and the action
% describes the direction of motion. The robot can move to the left or to
% the right. The first (1) and the final (6) states are the terminal
% states. The goal is to find an optimal policy that maximizes the return
% from any initial state. Here model-based value iteration
% Algorithm 2-1, from:
% @book{busoniu2010reinforcement,
%   title={Reinforcement learning and dynamic programming using function approximators},
%   author={Busoniu, Lucian and Babuska, Robert and De Schutter, Bart and Ernst, Damien},
%   year={2010},
%   publisher={CRC Press}
% }

%
%
%% this is the main function including the initialization and the algorithm
% the inputs are: initial Q matrix, set of actions, set of states,
% discounting factor, number of iterations, the epsilon criteria for
% breaking the algorithm
function stochastic_robot_cleaning_value
% initialization
state = [1,2,3,4,5,6];                      % set of states
action = [-1,1];                            % set of actions
Q = zeros(length(state),length(action));    % initial Q can be chosen arbitrariliy
V = zeros(size(state));
Vold = V;                                   % save a backup to compare later
L = 15;                                     % Max number of iterations
gamma = 0.5;                                % discounting factor
%gamma = 0.99;                              % discounting factor (right state changes decisionx)
epsilon = 0.001;                            % final error to stop the algorithm
% deterministic Q-iteration algorithm

for l = 1:L
    disp(['iteration: ' num2str(l)]);
    for ii = 1:length(state)
        for jj = 1:length(action)
            [StateP,pr,rew] = model(state(ii),action(jj));
            Q_temp = Q(StateP,:);
            Q(ii,jj) = pr*(rew' + gamma * max(Q_temp,[],2));
        end
        V(ii) = max(Q(ii,:));
    end
    if norm(V-Vold) < epsilon
        disp('Epsilon criteria satisfied!');
        break;
    else
        Vold = V;
    end
end
disp(Q)                                       % show the final Q matrix
[C,I]=max(Q,[],2);                              % finding the max values
% show the results
disp('V(s)(optimal):');
disp(V');
disp('Optimal Policy');
disp('*');
disp([action(I(2,1));action(I(3,1));action(I(4,1));action(I(5,1))]);
disp('*');
end

%% find the possible next states together with their probability value
function [xp,p,r] = model(x,u)
if (x == 1 || x == 6)
    xp = [x x x];
else
    xp = [x-u x x+u];
end
%p = [prob(x,u,xp(1)) prob(x,u,xp(2)) prob(x,u,xp(3))];
%r = [reward(x,u,xp(1)) reward(x,u,xp(2)) reward(x,u,xp(3))];
p = [prob(x,u,x-u) prob(x,u,x) prob(x,u,x+u)];
r = [reward(x,u,x-u) reward(x,u,x) reward(x,u,x+u)];
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
    if (u == -1 && xp == x)
        f = 0.1;
        f = 0.8;
    elseif (u == -1 && xp == x + u) 
        f = 0.8;
        f = 0.1;
    elseif (u == -1 && xp == x - u)
        f = 0.1;
    elseif (u == +1 && xp == x)
        f = 0.1;
    elseif (u == +1 && xp == x + u)
        f = 0.8;
    elseif (u == +1 && xp == x - u)
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
    r = 1;
elseif (x ~= 1 && xp == 1)
    r = 100;
else
    r = 0;
end
end
