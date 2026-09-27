%% K-means Algorithm - Machine Learning for automation - Prof. Alessandro D'Innocenzo, Francesco Smarra, Luis Florenzan

clc, clear, close all

% number of points in each cluster
numPoints = 500;
numClusters = 3;

% generate random data with 3 clusters
[Data_o, Mu_o,Sigma_o] = generateRandomData(numPoints,numClusters);

% reshuffle data lables 
Data_r = [Data_o(:,1:2) randi(numClusters, numClusters*numPoints, 1)];

% initial guess for the means
Mu = [1.5,1.5;
    2.5,2.5;
    0.5,0.5];
Mu = rand(3,2)

% run the k-means algorithm

% This is the main K-means algorithm. It performs two steps Expectation and
% Maximization. The whole process is done in a while loop until a desired
% error has reached. 
% Input: 
%     Data: the dataset including the points and labels [x y label]
%     Mu: center of the clusters
% Output: 
%     Data_f: the dataset with updated labels
%     Mu_f: updated centers
%}

iter = 0;       % counter
epsilon = 0.0001; % precision
shift = epsilon+1;  % larger than epsilon
formatSpec = 'iteration: %d, error: %2.4f, mu1: [%2.4f %2.4f], mu2: [%2.4f %2.4f], mu3: [%2.4f %2.4f] \n';
Data = Data_r;
while shift > epsilon
    iter = iter + 1;
    
    % Expectation
    Data_u = expectation(Data, Mu);
    
    % Maximization
    Mu_u = maximization(Data_u);
    
    % calculate the difference the current parameters and the parameters at the previous step
    shift = sqrt(sum(sum((Mu - Mu_u).^2)));
    
    fprintf(formatSpec, iter, shift, Mu_u(1,:), Mu_u(2,:), Mu_u(3,:));
    
    Mu = Mu_u;
    Data = Data_u;
end
Data_f = Data;
Mu_f = Mu;



% to compare the results with the Matlab version of k-means
[idx,Mu_m] = kmeans(Data_r(:,1:2),numClusters);

%% plot the results
figure;
subplot(2,2,1);
scatter(Data_o(:,1), Data_o(:,2), 10, 'MarkerEdgeColor', [0 0.5 0.5], 'MarkerFaceColor',[0 0.7 0.7], 'LineWidth',1.5);
grid on; box on;
xlabel('x'); ylabel('y');title('raw data');

Data1 = Data_o(Data_o(:,3)==1, :);
Data2 = Data_o(Data_o(:,3)==2, :);
Data3 = Data_o(Data_o(:,3)==3, :);
axis('square')

subplot(2,2,2);hold on
scatter(Data1(:,1), Data1(:,2), 10, 'MarkerEdgeColor', [0.5 0 0], 'MarkerFaceColor',[0.7 0 0], 'LineWidth',1.5);
scatter(Data2(:,1), Data2(:,2), 10, 'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor',[0 0.7 0], 'LineWidth',1.5);
scatter(Data3(:,1), Data3(:,2), 10, 'MarkerEdgeColor', [0 0 0.5], 'MarkerFaceColor',[0 0 0.7], 'LineWidth',1.5);
for ii=1:numClusters
    plot(Mu_o(ii,1), Mu_o(ii,2),'m+','Markersize',15, 'linewidth',2);
end
grid on; box on;
xlabel('x'); ylabel('y');title('true value');
axis('square')


Data_f1 = Data_f(Data_f(:,3)==1, :);
Data_f2 = Data_f(Data_f(:,3)==2, :);
Data_f3 = Data_f(Data_f(:,3)==3, :);
subplot(2,2,3); hold on
scatter(Data_f1(:,1), Data_f1(:,2), 10, 'MarkerEdgeColor', [0.5 0 0], 'MarkerFaceColor',[0.7 0 0], 'LineWidth',1.5);
scatter(Data_f2(:,1), Data_f2(:,2), 10, 'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor',[0 0.7 0], 'LineWidth',1.5);
scatter(Data_f3(:,1), Data_f3(:,2), 10, 'MarkerEdgeColor', [0 0 0.5], 'MarkerFaceColor',[0 0 0.7], 'LineWidth',1.5);
for ii=1:numClusters
    plot(Mu_f(ii,1), Mu_f(ii,2),'m+','Markersize',15, 'linewidth',2);
end
grid on; box on;
xlabel('x'); ylabel('y');title('data clustered by k-means');
axis('square')

Data_m1 = Data_r(idx==1, :);
Data_m2 = Data_r(idx==2, :);
Data_m3 = Data_r(idx==3, :);
subplot(2,2,4); hold on
scatter(Data_m1(:,1), Data_m1(:,2), 10, 'MarkerEdgeColor', [0.5 0 0], 'MarkerFaceColor',[0.7 0 0], 'LineWidth',1.5);
scatter(Data_m2(:,1), Data_m2(:,2), 10, 'MarkerEdgeColor', [0 0.5 0], 'MarkerFaceColor',[0 0.7 0], 'LineWidth',1.5);
scatter(Data_m3(:,1), Data_m3(:,2), 10, 'MarkerEdgeColor', [0 0 0.5], 'MarkerFaceColor',[0 0 0.7], 'LineWidth',1.5);
for ii=1:numClusters
    plot(Mu_m(ii,1), Mu_m(ii,2),'m+','Markersize',15, 'linewidth',2);
end
grid on; box on;
xlabel('x'); ylabel('y');title('matlab k-means');
axis('square')

%% Functions
function Xout = expectation(X, Mu)
%{ 
This function calculates the first step of the EM algorithm for K-means.
This function associates the closest point to each center to its cluster.
Input: 
    X : nx3 (number of data points , [x, y, label])
    Mu: 3x2
Output: 
    X: the dataset with updated labels
%}
[numPoints, numClusters] = size(X);

% define a function to calculate the distance between a set of points and a
% mean
calcDistance = @(x,y) (sum((x - repmat(y, size(x,1), 1)).^2 , 2)).^0.5;
% shift = sqrt(sum(sum((Mu - Mu_u).^2)));
DV = zeros(numPoints, numClusters);
for jj = 1:numClusters
    DV(:, jj) = calcDistance(X(:,1:2), Mu(jj,:));
end
[~, Idx] = min(DV, [], 2);
X(:,3) = Idx;
Xout = X;
end



function Mu = maximization(X)
%{ 
This function calculates the second step of the EM algorithm for K-means.
This function updates the center of the clusters according to the updated
labels in the previous step
Input: 
    X: nx3 (number of data points , [x, y, label])
Output: 
    Mu: updated centers
%}

[~, numClusters] = size(X);
Mu = zeros(numClusters, 2);
for jj = 1:numClusters
    idx = X(:,3) == jj;
    data = X(idx, 1:2);
    Mu(jj, :) = sum(data) / size(data,1);
end
end