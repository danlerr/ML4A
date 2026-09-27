%% Model-based value iteration Algorithm for Deterministic Cleaning Robot
%% The deterministic cleaning-robot MDP
% a cleaning robot has to collect a used can and also has to recharge its
% batteries. the state describes the position of the robot and the action
% describes the direction of motion. The robot can move to the left or to
% the right. The first (1) and the final (6) states are the terminal
% states. The goal is to find an optimal policy that maximizes the return
% from any initial state. Here the model-based value iteration 
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
function deterministic_robot_cleaning_value
% initialization
state = [1,2,3,4,5,6];                      % set of states
action = [-1,1];                            % set of actions
Q = zeros(length(state),length(action));    % initial Q can be chosen arbitrariliy
V = zeros(size(state));
Vold = V;                                   % save a backup to compare later
L = 15;                                     % Max number of iterations
gamma = 0.99;                                % discounting factor
epsilon = 0.001;                            % final error to stop the algorithm
% deterministic Q-iteration algorithm

for l = 1:L
    disp(['iteration: ' num2str(l)]);
    for ii = 1:length(state)
        for jj = 1:length(action)
            Q(ii,jj) = reward(state(ii),action(jj)) + gamma * max(Q(model(state(ii),action(jj)),:));
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
disp(V);
disp('Optimal Policy');
disp('*');
disp([action(I(2,1));action(I(3,1));action(I(4,1));action(I(5,1))]);
disp('*');
end

%% this function is the transition model of the robot
% the inputs are: the current state, and the chosen action
% the output is the next state
function f = model(x,u)
if (x <= 5 && x>=2)
    f = x + u;
else
    f = x;
end
end

%% this function is the reward function for the task
% the inputs are: the current state, and the chosen action
% the output is the expected reward
function r = reward(x,u)
    if (x == 5 && u == 1)
        r = 3; %check decision according to best final reward
    elseif (x == 2 && u == -1)
        r = 2;
        %r = 4; %(changes direction)
    else
        r = 0;
    end
end
