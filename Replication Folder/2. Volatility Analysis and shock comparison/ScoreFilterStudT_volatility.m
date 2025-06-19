function [LL, Res, Save_Other] = ScoreFilterStudT_volatility(vpar,Y,X)

  
%INPUT:     
% Y - depndent variable Tx1
% X - explanatory variables (intercept plus lags of the dependent) Txk

% OUTPUT: 
% mCoeffs - kx1 matrix of nonTVP coeffs (possibly include constant) 
% vSig2 - 1xT vector of TV variance
% loglik - 1xT predictive log density
% LL - likelihood function [sum(loglik)]


T = size(Y,1);
k = size (X,2); %number of regressors 

vSig2 = zeros(1,T+1);
loglik = zeros(1,T); % include likelihood 

if size(vpar,1)<size(vpar,2)
vpar=vpar';
end
 
TrasfX=@(x) (1+(x/((1+x^2)^.5)))/2;

Omegas = exp(vpar(1));
Alfas = .3*TrasfX(vpar(2));
Betas = (1-Alfas)*TrasfX(vpar(3))-0.001;

eta = 1/(2.1+exp(vpar(4))); 

BetaCoeff = vpar(4+1:4+k);

Res.Omegas = Omegas;
Res.Alfas = Alfas;
Res.Betas = Betas;
Res.eta   = eta;
Res.BetaCoeff = BetaCoeff;
    
% initialize the recursion
sig2_t=Omegas/(1-Alfas-Betas);   %initialization of the (transformmed) variance

for t=1:T+1
   
    vSig2(:,t) = sig2_t;
    sig2 = sig2_t;
    
    %% Given Parameters Calculate Likelihood 
    if t<=T
    x = X(t,:)'; % column vector
    y = Y(t,1);  % scalar
    
    eps = y - x'*BetaCoeff;
    eps2 = eps^2;
    zz_t = eps2/sig2;
    w_t=1/(1-2*eta+eta*zz_t);
    
    
    % now calculate the likelihood 
    CC_eta= log(gamma((eta+1)/(2*eta))) - log(gamma(1/(2*eta))) -.5*log((1-2*eta)/eta) -.5*log(pi) ; 
    loglik(1,t) = CC_eta - .5*log(sig2) - ((eta+1)/(2*eta))* log(1+((eta/(1-2*eta))*(eps2/sig2)));

    
    %% Update the Parameters
    score_v = (1+3*eta)* ((1+eta)*w_t*eps2 - sig2); 
        
    Save_Other(t,:) = [w_t eps eps2 score_v];
    
    sig2_t_old=sig2_t; 
    sig2_t = Omegas + Betas * sig2_t_old + Alfas * score_v; 
    
    end
    
end
LL=-sum(loglik(1,2:end))/T;
Res.loglik = loglik';
Res.vSig2 = vSig2';

% if isnan(LL)
%     LL=10^5;
% end

 