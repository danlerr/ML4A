%% EM Algorithm - Machine Learning for automation - Prof. Alessandro D'Innocenzo, Francesco Smarra, Luis Florenzan

clc, clear, close all

%rng(2);
% number of points in each cluster
numPoints = 500; 
numClusters = 3;
% generate random data using 3 2D Normal distributions with 500 data points 
[Data_o, Mu_o, Sigma_o] = generateRandomData(numPoints,numClusters);

% reshuffle the data labels
Data_r = [Data_o(:,1:2) randi(numClusters, numClusters*numPoints, 1)];

% make some initial guess
Param = struct();
Param.mu1 = [1.5, 1.5];
Param.mu2 = [4.5, 4.5];
Param.mu3 = [7.5, 7.5];
Param.sigma1 = [1 0; 0 1];
Param.sigma2 = [1 0; 0 1];
Param.sigma3 = [1 0; 0 1];
Param.lambda = [0.3, 0.5, 0.2];

% run EM to find the parameters 

shift = 10000;  % a big number
iter = 0;       % counter
epsilon = 0.001; % percision
formatSpec = 'iteration: %d, error: %2.4f, mu1: [%2.4f %2.4f], mu2: [%2.4f %2.4f], mu3: [%2.4f %2.4f] \n';

while shift > epsilon
    iter = iter + 1;

    % E-step
    [Gamma, Data_r] = expectation(Data_r, Param);
    
    % M-step
    Param_ = maximization(Gamma,Data_r, Param);
    
    % calculate the distance from the previous set of params
    shift = calc_distance(Param, Param_);
    
    fprintf(formatSpec, iter, shift, Param_.mu1, Param_.mu2, Param_.mu3);    

    
    Param = Param_;
end
Data_f = Data_r;
Param_f = Param;

% to compare the results with the Matlab version of EM
gmm=  fitgmdist(Data_o(:,1:2),numClusters);
idx_m = cluster(gmm,Data_o(:,1:2));

%% plot the results
figure; subplot(2,2,1);
scatter(Data_o(:,1), Data_o(:,2), 10, 'MarkerEdgeColor', [0 0.5 0.5], 'MarkerFaceColor',[0 0.7 0.7], 'LineWidth',1.5);
grid on; box on;
xlabel('x'); ylabel('y');title('raw data');

Data1 = Data_o(Data_o(:,3)==1, :);
Data2 = Data_o(Data_o(:,3)==2, :);
Data3 = Data_o(Data_o(:,3)==3, :);

subplot(2,2,2);hold on
scatter(Data1(:,1), Data1(:,2), 10, 'MarkerEdgeColor', [0.5 0 0], 'MarkerFaceColor',[0.7 0 0], 'LineWidth',1.5);
scatter(Data2(:,1), Data2(:,2), 10, 'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor',[0 0.7 0], 'LineWidth',1.5);
scatter(Data3(:,1), Data3(:,2), 10, 'MarkerEdgeColor', [0 0 0.5], 'MarkerFaceColor',[0 0 0.7], 'LineWidth',1.5);
grid on; box on;
xlabel('x'); ylabel('y');title('true value');

Data_f1 = Data_f(Data_f(:,3)==1, :);
Data_f2 = Data_f(Data_f(:,3)==2, :);
Data_f3 = Data_f(Data_f(:,3)==3, :);
subplot(2,2,3); hold on
scatter(Data_f1(:,1), Data_f1(:,2), 10, 'MarkerEdgeColor', [0.5 0 0], 'MarkerFaceColor',[0.7 0 0], 'LineWidth',1.5);
scatter(Data_f2(:,1), Data_f2(:,2), 10, 'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor',[0 0.7 0], 'LineWidth',1.5);
scatter(Data_f3(:,1), Data_f3(:,2), 10, 'MarkerEdgeColor', [0 0 0.5], 'MarkerFaceColor',[0 0 0.7], 'LineWidth',1.5);

grid on; box on;
xlabel('x'); ylabel('y');title('data clustered by EM');

Data_m1 = Data_f(idx_m==1, :);
Data_m2 = Data_f(idx_m==2, :);
Data_m3 = Data_f(idx_m==3, :);

subplot(2,2,4); hold on
scatter(Data_m1(:,1), Data_m1(:,2), 10, 'MarkerEdgeColor', [0.5 0 0], 'MarkerFaceColor',[0.7 0 0], 'LineWidth',1.5);
scatter(Data_m2(:,1), Data_m2(:,2), 10, 'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor',[0 0.7 0], 'LineWidth',1.5);
scatter(Data_m3(:,1), Data_m3(:,2), 10, 'MarkerEdgeColor', [0 0 0.5], 'MarkerFaceColor',[0 0 0.7], 'LineWidth',1.5);

grid on; box on;
xlabel('x'); ylabel('y');title('data clustered by Matlab');

Sigma_o;
Param.sigma1;
Param.sigma2;
Param.sigma3;

[Mu_o(1,:)', Param_.mu1']
[Mu_o(2,:)', Param_.mu2']
[Mu_o(3,:)', Param_.mu3']



%% Functions
function d = calc_distance(Param, Param_)
 
%This function calculates the distance between two sets of parameters. 

d = norm(Param.mu1 - Param_.mu1) + norm(Param.mu2 - Param_.mu2) + norm(Param.mu3 - Param_.mu3)
end

