function [D, Mu,Sigma] = generateRandomData(n,K)
%{ 
This function generates random data
The points are drawn from two Normal Distributions with random parameters (mu, sigma)
Input: 
    n: number of points in each cluster
    K: number of clusters
Output: 
    D: the generated dataset [x y label]
    Mu: expectation vector
    Sigma: covariance matrix
%}
mu_max = 10;

Mu = [];
Sigma = [];
D = [];
for i = 1:K
    Mu_i = 3*rand(1,2)+3*(i-1);
    
    temp = rand(2,2);
    Sigma_i = 0.75*(temp+temp')^2 ; %Sigma_i must be simmetric and def. positive  %for GMM
    %Sigma_i = eye(2);       %for k-means
    D = [D; mvnrnd(Mu_i, Sigma_i, n), i*ones(n,1)];
    
    Mu = [Mu; Mu_i];
    Sigma = [Sigma, Sigma_i];
    
end


end