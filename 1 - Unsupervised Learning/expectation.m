function [Gamma, Data] = expectation(Data, Param)
%{ 
This function calculates the first step of the EM algorithm, Expectation.
It calculates the probability of each specific data point belong to each
cluster or class


%}
 
    p_cluster1 = Param.lambda(1,1)*mvnpdf(Data(:,1:2), Param.mu1, Param.sigma1 );
    p_cluster2 = Param.lambda(1,2)*mvnpdf(Data(:,1:2), Param.mu2, Param.sigma2 );
    p_cluster3 = Param.lambda(1,3)*mvnpdf(Data(:,1:2), Param.mu3, Param.sigma3 );
   
    Gamma = [p_cluster1, p_cluster2, p_cluster3]./(p_cluster1+p_cluster2+p_cluster3); 
    [~, Data(:,3)] = max(Gamma,[],2); 
end
