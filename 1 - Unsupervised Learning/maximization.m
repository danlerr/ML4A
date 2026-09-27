function Param = maximization(Gamma,Data, Param)
%{ 
This function calculates the second step of the EM algorithm, Maximization.
It updates the parameters of the Normal distributions according to the new 
labled dataset.

Input: 
    Gamma : nx3
    Param: (mu, sigma, lambda)

Output: 
    Param: updated parameters 
%}

n = sum(Gamma,1);

% calculate the weights
Param.lambda = n/sum(n);


% calculate the means
Param.mu1 = Gamma(:,1)'*Data(:,1:2)/n(1) ;
Param.mu2 = Gamma(:,2)'*Data(:,1:2)/n(2);
Param.mu3 = Gamma(:,3)'*Data(:,1:2)/n(3);

% calculate the variances
Param.sigma1 = (Data(:,1:2)-Param.mu1)'*((Data(:,1:2)-Param.mu1).*Gamma(:,1))/n(1);
Param.sigma2 = (Data(:,1:2)-Param.mu2)'*((Data(:,1:2)-Param.mu2).*Gamma(:,2))/n(2);
Param.sigma3 = (Data(:,1:2)-Param.mu3)'*((Data(:,1:2)-Param.mu3).*Gamma(:,3))/n(3);

end