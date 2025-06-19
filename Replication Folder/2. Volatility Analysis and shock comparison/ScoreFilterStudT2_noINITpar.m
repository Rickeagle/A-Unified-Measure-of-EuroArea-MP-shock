function [LL, mCoeffs, vSig2, loglik,Save_Other] = ScoreFilterStudT2_noINITpar(vpar,...
    Y,X,station,const,lowB,upB,numParam,vCoeffsIn,Sig2In)

 
%INPUT:     
% Y - depndent variable Tx1
% X - explanatory variables (intercept plus lags of the dependent) Txk
% vCoeffsIn - initial coefficients for the recursion kx1 (transformmed coefficients)  
%           (if included, the first element is the constant)
% Sig2In - initial variance 1x1 (transformmed variance)
% Hinit - initial Hessian matrix (e.g. inv(X'*X)): kxk matrix
% vpar - vector of parameter 4x1 (3x1 if not constant) (smoothing coefficients:
%        (1st smoot. coeff; 2nd smoot. Hessian; 3rd smoot. variance; the
%        first smoothing might be 2x1 when a constant is included) 
% station - 0 no constraint on the coeff./ 1 impose stationarity of coeff. 
% const - 0 no constant/ 1 constant (no-bounds)/ 2 constant with bounds
% lowB - lower bound for the constant
% upB - upper bound for the constant
% 
% 
% OUTPUT: 
% mCoeffs - kxT matrix of TV coeffs (possibly include constant) 
% vSig2 - 1xT vector of TV variance
% loglik - 1xT predictive log density
% LL - likelihood function [sum(loglik)]

% global Y X Hinit station const lowB upB numParam vCoeffsIn Sig2In


T = size(Y,1);
k = size (X,2); %number of regressors 
mCoeffs = zeros(k,T+1);
vSig2 = zeros(1,T+1);
loglik = zeros(1,T); % include likelihood 

cccc=1;

if size(vpar,1)<size(vpar,2)
vpar=vpar';
end
 
TrasfX=@(x) (1+(x/((1+x^2)^.5)))/2;
costant=1-.005;

if numParam==2
%     lambda_c = vpar(1);%
%     lambda_v = vpar(2);%
    lambda_c = costant*TrasfX(vpar(1)) + .005; %vpar(1);%
    lambda_v = costant*TrasfX(vpar(2)) + .005; %vpar(2);%
    if vpar(3)>5; vpar(3)=5; end
    eta = 1/(2.1+exp(cccc*vpar(3))); %vpar(3);
%     vCoeffsIn = vpar(4:end-1);
%     Sig2In = vpar(end);
    lambda_h = 1; %lambda_c; %1; %
    B_c = lambda_c*eye(k);
elseif numParam==3
    lambda_c = TrasfX(vpar(1)) + .005; %vpar(1);%
    lambda_v = TrasfX(vpar(2)) + .005; %vpar(2);%
    lambda_h = TrasfX(vpar(3)) + .005; %vpar(3); 
    eta = 1/(2+exp(cccc*vpar(4))); %vpar(3);    
%     vCoeffsIn = vpar(5:end-1);
%     Sig2In = vpar(end);
    B_c = lambda_c*eye(k);
elseif numParam==4
    lambda_const = TrasfX(vpar(1)) + .005; 
    lambda_c = TrasfX(vpar(2)) + .005; %vpar(2);%
    lambda_v = TrasfX(vpar(3)) + .005; %vpar(3);%
    lambda_h = TrasfX(vpar(4)) + .005; %vpar(4); 
    eta = 1/(2+exp(cccc*vpar(5))); %vpar(3);
%     vCoeffsIn = vpar(6:end-1);
%     Sig2In = vpar(end);
    B_c = diag([lambda_const lambda_c*ones(1,k-1)]);
end

A_c = eye(k); %random walk coeffs
A_v = eye(1); %IGARCH type
B_v = lambda_v ;

% initialize the recursion
gam_t=Sig2In;   %initialization of the (transformmed) variance
a_t=vCoeffsIn;  %initialization of the (transformmed) coefficients
% H = Hinit;      %initialization of the Hessian
% HH=H;

for t=1:T+1
   
    
    %% Apply the appropriate transformation
    %in case not transformation vphi = valpha; sig2 = delta; PSIc = eye(k); PSIs =  1;
    
    [sig2, PSIs] =  LinkFne_PosVar(gam_t);
    vphi = a_t; PSIc=eye(k);
    if station==1  
        if const==0; 
            [vvv, PSIcoeff] =  LinkFne_StCoeffs2(a_t); % Link function and its Jacobian
            PSIc = PSIcoeff;
            vphi = vvv; clear vvv PSIcoeff
        else
            [vvv, PSIcoeff] =  LinkFne_StCoeffs2(a_t(2:end,1)); % Link function and its Jacobian
            PSIc(2:end,2:end) = PSIcoeff;
            vphi(2:end,1) = vvv; clear vvv PSIcoeff
        end
    end
    if const==2 % constant  with bounds
        [vvv,PSIo_11,PSIo_12]=LinkFne_BMean2(a_t(1),lowB,upB,vphi(2:end,1)); % Link function and its Jacobian (constant) 
        PSIc(1,1) = PSIo_11; 
        
        
        PSIc(1,2:end) = PSIo_12' * PSIc(2:end,2:end);
        PSIc(2:end,1) = 0*PSIc(1,2:end)'; 
        
        vphi(1) = vvv; clear vvv PSIconst PSIo_11 PSIo_12
    end
    
    mCoeffs(:,t) = vphi;
    vSig2(:,t) = sig2;
    
    
    %% Given Parameters Calculate Likelihood 
    if t<=T
    x = X(t,:)'; % column vector
    y = Y(t,1);  % scalar
    
    eps = y - x' * vphi;
    eps2 = eps^2;
    zz_t = eps2/sig2;
    w_t=1/(1-2*eta+eta*zz_t);
    
    
    % now calculate the likelihood 
    CC_eta= log(gamma((eta+1)/(2*eta))) - log(gamma(1/(2*eta))) -.5*log((1-2*eta)/eta) -.5*log(pi) ; 
    loglik(1,t) = CC_eta - .5*log(sig2) - ((eta+1)/(2*eta))* log(1+((eta/(1-2*eta))*(eps2/sig2)));

    
    %% Update the Parameters
    H = ((x * x')/sig2);
    if t==1
        HH = (PSIc'*H*PSIc); 
%         HH = eye(size(x,1)); %(1-lambda_h) * Hinit + lambda_h * (PSIc'*H*PSIc);   %Hinit;%(PSIc'*H*PSIc); 
%         HH = (1-lambda_h) * eye(size(x,1))+ lambda_h * (PSIc'*H*PSIc);  
%         HH = (1-lambda_h) * Hinit+ lambda_h * (PSIc'*H*PSIc); 
%         HH =  .5 * eye(size(x,1))+ .5 * (PSIc'*H*PSIc);  
    else
        HH =  (1-lambda_h) * HH + lambda_h * (PSIc'*H*PSIc);   %smoothing the Hessian of coeffs
    end
    score_v = (1+3*eta)* PSIs^-1 * ((1+eta)*w_t*eps2 - sig2); 
    
    if isnan(sum(sum(HH)))==1 || sum(sum(HH))==inf
    score_c = 0*a_t; loglik(1,t)=-(10^6)*T; %NaN; %     
    else
    score_c = ((1-2*eta)*(1+3*eta))*pinv(HH) *  PSIc'* (x/sig2) * w_t* eps;
    if isnan(score_c)==1; 
    score_c = 0*a_t; loglik(1,t)=-(10^6)*T; %NaN; %     
    end
    end
    
    Save_Other(t,:) = [w_t eps eps2 score_c score_v];
    
    a_t_old=a_t; a_t = A_c * a_t_old + B_c * score_c; %if lambda_c==0; a_t = A_c * a_t_old; end; 
    gam_t_old=gam_t; gam_t = A_v * gam_t_old + B_v * score_v; %if lambda_v==0; gam_t = A_v * gam_t_old; end; 
    
    end
    

    
    
end
LL=-sum(loglik(1,2:end))/T;

% if isnan(LL)
%     LL=10^5;
% end

 