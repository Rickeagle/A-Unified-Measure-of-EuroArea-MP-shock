clc
clear
close all
function varargout = plot(varargin)
  % call built-in
  [varargout{1:nargout}] = builtin('plot',varargin{:});
  % tighten axes around the data you just drew
  axis tight
end
addpath('C:\Users\Utente\Dropbox\Thesis\Financial Econ Project\Replication folder\Replication codes')

%% LOAD DATA 
folder='C:\Users\Utente\Dropbox\Thesis\Financial Econ Project\Pictures';
data = readtable('NEW.xlsx','ReadVariableNames',true);
data2=readtable("comparison_ecb.xlsx",'ReadVariableNames',true);
varnames=data2.Properties.VariableNames;
MP=data2.ecb;
data3=data2(1:168,[14:17]); % I AM COLLECTING THE FACTORS SCALED
data3=fillmissing(data3,"constant",0);
Target=data2.ratefactor1;
Timing=data2.conffactor1;
FG=data2.conffactor2;
QE=data2.conffactor3;
Jaro_Median=data2.mp_median;
Target_scaled=data3.ratefactor1_scale;
Timing_scaled=data3.conffactor1_scale;
FG_scaled=data3.conffactor2_scale;
QE_scaled=data3.conffactor3_scale;
mdate=data2.mdate;


%% Selecting the correct MP to run
MP=MP;
mon_name='_MP.png';
plot (MP)
fname=['Plot' mon_name;];
print(gcf, fullfile(folder, fname), '-dpng', '-r300');

%% Basic Time Series Properties of MP
figure;
autocorr(MP);
title('Sample ACF of MP Shocks');
% Plot the ACF of Squared MP
figure;
autocorr(MP.^2);
title('Sample ACF of Squared MP Shocks');

%% 
clear bAIC bP bQ
P = 3; 
Q = 3; 
bAIC = Inf; % Start with a very high AIC
bP = 0; 
bQ = 0; 
% Loop over possible combinations of p and q
for p = 0:P
    for q = 0:Q
        try
            Mdl = arima(p, 0, q);
            EstMdl = estimate(Mdl, MP, 'Display', 'off');           
            % Compute AIC
            [~,~,logL] = infer(EstMdl, MP ); 
            numParams = p + q + 1; % Number of parameters (AR + MA + constant)
            AIC = -2 * logL + 2 * numParams; 
            
            
            if AIC < bAIC
                bAIC = AIC;
                bP = p;
                bQ = q;
            end
        catch
            fprintf('Model ARMA(%d, %d) failed to estimate.\n', p, q);
        end
    end
end
fprintf('Optimal ARMA Model: AR(%d), MA(%d) with AIC = %.4f\n', bP, bQ, bAIC);

% Fit the optimal ARMA model
OptimalMdl = arima(bP, 0, bQ);
OptimalEstMdl = estimate(OptimalMdl, MP);
residuals_ARMA = infer(OptimalEstMdl, MP);

squared_residuals_ARMA = residuals_ARMA.^2;
[h, pValue] = lbqtest(squared_residuals_ARMA, 'Lags', 10);

if h == 0
    fprintf('No significant ARCH effects detected (p-value = %.4f).\n', pValue);
else
    fprintf('Significant ARCH effects detected (p-value = %.4f).\n', pValue);
end
%}
%optimal arma is 0,0 ... the series is basically a white noise around
% -0.0064
%% Plot Squared Residuals Autocorrelation after fitting the the ARMA(0,0)
%{
figure;
autocorr(squared_residuals_ARMA);
title('Autocorrelation of Squared Residuals');
xlabel('Lag');
ylabel('Autocorrelation');
grid on;
%Residuals do not seem autocorrelated 

%}



%% AR Model for MP with EGARCH Volatility Specification
% AR(1) with EGARCH(5,4) volatility model
model = arima('ARLags',1, 'Distribution', 'Gaussian', 'Variance', egarch(5, 4));
options = optimoptions(@fmincon, 'Display', 'off', 'Diagnostics', 'off', 'Algorithm', 'sqp', 'TolCon', 1e-7);
% Fit the model to MP
fitEG = estimate(model, MP, 'Options', options);
% Infer residuals and variances
[residuals_EGARCH, variances_EGARCH] = infer(fitEG, MP);

figure;
subplot(2,1,1);
plot(residuals_EGARCH,'LineWidth',1.5);
grid on; xlabel('Time'); ylabel('Residual');
title('Filtered Residuals');
xlim([1 numel(residuals_EGARCH)]);
standardizedResiduals_EGARCH = residuals_EGARCH ./ sqrt(variances_EGARCH);
subplot(2,1,2);
plot(standardizedResiduals_EGARCH,'LineWidth',1.5);
grid on; xlabel('Time'); ylabel('Volatility');
title('Standardized Residuals');
fname='AR1_EGARCH(5,4)';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r300');
%Leverage(1)=γ₁ is the asymmetry from the one‐period‐ago standardized shock.
%Leverage(2)=γ₂ is the two‐period‐lag asymmetry.
%Leverage(3)=γ₃ is the three‐period‐lag asymmetry.
std_res_EGARCH= residuals_EGARCH./ sqrt(variances_EGARCH);
squared_residuals_EGARCH = std_res_EGARCH.^2;
[h, pValue] = lbqtest(squared_residuals_EGARCH, 'Lags', 15);

if h == 0
    fprintf('No significant ARCH effects detected (p-value = %.4f).\n', pValue);
else
    fprintf('Significant ARCH effects detected (p-value = %.4f).\n', pValue);
end
% The EGARCH (5,4) does not show ARCH effects, hence it captures the
% residuals well

S1 = summarize(fitEG);
logL1 = S1.LogLikelihood;
k1    = S1.NumEstimatedParameters;
% AIC/BIC
n = numel(MP);  % sample size
[aic1,bic1] = aicbic(logL1, k1, n);
fprintf('Model 1: AIC=%.2f, BIC=%.2f\n', aic1, bic1)



%% Standardize Residuals

figure;
autocorr(standardizedResiduals_EGARCH);
title('Sample ACF of Standardized Residuals');

% Plot ACF of squared standardized residuals
figure;
autocorr(standardizedResiduals_EGARCH.^2);
title('Sample ACF of Squared Standardized Residuals');
fname='ACF_EGARCH';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r300');

% Let's check autocorr in the residuals
[h_EGARCH, pValue_EGARCH] = lbqtest(standardizedResiduals_EGARCH, 'Lags', 15);
fprintf('Ljung-Box p-value (EGARCH standardized residuals): %.4f\n', pValue_EGARCH);
% We have that EGARCH (2,3) using a student t-distribution works fine

%%%%%%%%%%%%%%
%% EGARCH residuals scatter plot
% Compute lagged residuals and corresponding variances
eps_lag   = residuals_EGARCH(1:end-1);
sigma2    = variances_EGARCH(2:end);

% Scatter‐plot eps_{t-1} vs σ²_t
figure;
scatter(eps_lag, sigma2, 25, 'filled');
grid on;
xlabel('\epsilon_{t-1}');
ylabel('\sigma^2_{t|t-1}');
title('Lagged Residual vs Conditional Variance');
fname='Asymmetry_EGARCH';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r300');


%% GARCH-t                  
%  LAG SELECTION
clear AIC BestAIC BestLag AICs
lags       = 1:5;
bestAIC    = Inf;
bestLag    = [];
AICs       = NaN(length(lags),1);
options    = optimoptions(@fmincon, ...
               'Display','off', ...
               'Diagnostics','off', ...
               'Algorithm','sqp', ...
               'TolCon',1e-7);

for i = 1:length(lags)
    p = lags(i);
    q = lags(i);
    try
        Mdl = arima( ...
              'ARLags',      2, ...
              'Distribution','t', ...
              'Variance',    garch(p,q));
        [EstMdl,~,LogL] = estimate(Mdl, MP, 'Options', options);
        [~,VolatilityGARCH_t] = infer(EstMdl, MP);
        numParams = numel(EstMdl.Constant) ...
                  + numel(EstMdl.Variance.ARCH) + numel(EstMdl.Variance.GARCH)+ 1;          % t-dist DoF
        AICs(i) = -2*LogL + 2*numParams;
        if AICs(i) < bestAIC
            bestAIC = AICs(i);
            bestLag = [p, q];
        end
    catch ME
        fprintf('Failed GARCH(%d,%d): %s\n', p, q, ME.message);
    end
end
fprintf('Best t distribution model (EGARCH-t standardized residuals): %.4f\n', bestLag);

% The best Specification is the GARCH(1,1)
%% Estimate GARCH-t for MP
Mdl_GARCH33_t = arima('ARLags',1, 'Distribution', 't', 'Variance', garch(3, 3));
options = optimoptions(@fmincon,Display="off",Diagnostics="off",Algorithm="sqp",TolCon=1e-7);

fitG = estimate(Mdl_GARCH33_t,MP,Options=options);  % Fit the model
[residuals_GARCH,VolatilityGARCH_t] = infer(fitG,MP);
%
standardizedResiduals_GARCH= residuals_GARCH ./ sqrt(VolatilityGARCH_t);
%
% Optional: Plot residuals
figure(10);
subplot(2,1,1);
plot(residuals_GARCH, 'LineWidth', 1.5);
title('GARCH-t Residuals');
xlabel('Time');
ylabel('Residuals');
grid on;

subplot(2,1,2);
plot(standardizedResiduals_GARCH, 'LineWidth', 1.5);
title('GARCH-t Standardized Residuals');
xlabel('Time');
ylabel('Standardized Residuals');
grid on;
fname='GARCH.t_3.3';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r300');
%
figure;
autocorr(standardizedResiduals_GARCH.^2);
title('Sample ACF of Squared Standardized Residuals');
fname='ACF_GARCH.t';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r300');

% Does the model fit well?
[h_GARCH, pValue_GARCH] = lbqtest(standardizedResiduals_GARCH.^2, 'Lags', 10);
fprintf('Ljung-Box p-value (GARCH-t standardized residuals): %.4f\n', pValue_GARCH);
% The model is fine!
h=sqrt(VolatilityGARCH_t);
%But let's compare it to a DCS model

%%  GRAPH FOR DISTRIBUTION OF THE RESIDUALS
% assume ν was pulled from your fitted model:
nu = fitG.Distribution.DoF;

% plot histogram + KDE + t + normal
figure;
histogram(standardizedResiduals_EGARCH, 'Normalization','pdf','EdgeColor','none');
hold on;
[f,xi] = ksdensity(standardizedResiduals_EGARCH);
plot(xi, f,      'LineWidth',1.5);

% Student-t(ν)
pd_t    = makedist('tLocationScale','nu',nu,'mu',0,'sigma',1);
y_t     = pdf(pd_t, xi);
plot(xi, y_t,    '--','LineWidth',1.5);

% Normal(0,1)
pd_norm = makedist('Normal','mu',0,'sigma',1);
y_norm  = pdf(pd_norm, xi);
plot(xi, y_norm, ':','LineWidth',1.5);

hold off;
xlabel('Standardized Residual');
ylabel('Density');
title('Residuals PDF with KDE, t- and Normal fits');
legend('Empirical PDF','KDE','Fitted t','Normal(0,1)','Location','best');

% 2. Q–Q plot against N(0,1)
figure;
qqplot(standardizedResiduals_GARCH);
title('Q–Q Plot of Standardized Residuals');
nu = fitG.Distribution.DoF;
pd = makedist('tLocationScale','nu',nu,'mu',0,'sigma',1);
figure;
qqplot(standardizedResiduals_GARCH, pd);
title('Q–Q Plot vs. Student-t');


%%   OUT OF SAMPLE FORECASTING OF GARCH-t (3,3) and EGARCH (3,3)
%{
T = numel(MP);
h = 45;        
Y = MP;
yTrain = Y(1:end-h);

% preallocate
Vf_G = nan(h,1);
Vf_E = nan(h,1);
RV  = nan(h,1);

for i = 1:h
    Yi = Y(1:end-h+i-1);
    
    %— GARCH-t
    MdlG = arima('ARLags',1,'Distribution','t','Variance',garch(3,3));
    fitG = estimate(MdlG,Yi,'Options',options,'Display','off');
    [yF, VF] = forecast(fitG,1,'Y0',Yi);
    Vf_G(i) = VF;
    
    %— EGARCH
    MdlE = arima('ARLags',1,'Distribution','Gaussian','Variance',egarch(3,3));
    fitE = estimate(MdlE,Yi,'Options',options,'Display','off');
    [yF2, VE] = forecast(fitE,1,'Y0',Yi);
    Vf_E(i) = VE;
    
    %— realized variance = squared AR(1) residual
    RV(i) = (Y(end-h+i) - yF)^2;
end

%— MSE’s
mseG = mean((Vf_G - RV).^2);
mseE = mean((Vf_E - RV).^2);

fprintf('MSE GARCH-t: %.4g\nMSE EGARCH: %.4g\n', mseG, mseE);


%% PLOTTING THE OUT OF SAMPLE FORECAST
% t-axis for hold-out
t0 = (T-h+1):T;
figure; hold on;
plot(t0, Vf_G,    'b-', 'LineWidth',1.2);
plot(t0, Vf_E,    'r--','LineWidth',1.2);
plot(t0, RV,      'k:','LineWidth',1.2);
hold off;

xlabel('Time');
ylabel('Variance');
legend('GARCH-t forecast','EGARCH forecast','Realized variance','Location','Best');
title('Out-of-Sample Variance Forecasts vs Realized');
grid on;
fname='Out_of_sample_Forecast.png';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r1200');

%}

%%
% Export VolatilityGARCH_t to add to a dataset
datasetWithVolatility = table(MP, VolatilityGARCH_t, ...
    'VariableNames',{'MP','GARCHVolatility'});

% Save the dataset as an excel file 
writetable(datasetWithVolatility, 'GARCH_Estimation.xlsx');
%%
%%%%%%%%%%%%%%%%%%%%%%
% Score and Garch-t comparison
X = ones(size(MP));
start_plot = 1; 
end_plot = size(MP,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WhichVariable = MP(start_plot:end_plot,1);
WhichDates = [1:size(MP,1)]';

QANT_CUT = 8;
COLORE = 'k';
SPESSORE = 2;
%% Adjusted DCS-t Estimation Code
% Optimization options
optionsIVAN = optimset('Display', 'iter-detailed', ...
                       'LargeScale', 'off', ...
                       'MaxFunEvals', 5000, ...
                       'TolFun', 1e-6, ...
                       'TolX', 1e-6);
% Define bounds for parameters
lb = [-10; -10; -10; -10; min(MP)];  
ub = [10; 10; 10; 10; max(MP)];     

% Improved initial parameter guess
vpars_init = [randn(4, 1) * 0.1; mean(MP)];
% Loss function with error handling for complex outputs
lossT = @(vpar) ScoreFilterStudT_volatility_safe(vpar, MP, X);
% Run optimization using fmincon to apply bounds
[vpars, fval, exitflag, output] = fmincon(lossT, vpars_init, [], [], [], [], lb, ub, [], optionsIVAN);
[LL, Res, Save_Other] = ScoreFilterStudT_volatility_safe(vpars, MP, X);
VolatilityDCS_t = Res.vSig2(1:end-1, 1);

%% Safe ScoreFilterStudT_volatility Function
function [LL, Res, Save_Other] = ScoreFilterStudT_volatility_safe(vpars, MP, X)
    [LL, Res, Save_Other] = ScoreFilterStudT_volatility(vpars, MP, X);
    
    % Handle complex or invalid outputs
    if any(imag(LL) ~= 0) || isnan(LL)
        warning('Complex or NaN log-likelihood detected. Penalizing solution.');
        LL = Inf;  % Penalize invalid solutions
    end
end

%% Plotting Comparison
CUTT = 20;
WhichVariable = [VolatilityDCS_t(CUTT:end, 1), VolatilityGARCH_t(CUTT:end, 1)];
WhichDates = 1:length(WhichVariable);

figure(22);
plot1 = plot(WhichDates, WhichVariable, 'linewidth', 2);
set(plot1(1), 'Color', 'b', 'linestyle', '-');
set(plot1(2), 'Color', 'r', 'linestyle', '--');

xlim([min(WhichDates), max(WhichDates)]);
set(gca, 'Xtick', WhichDates);
title('COMPARISON VOLATILITIES');
legend('DCS-t', 'GARCH-t');
grid('on');
fname='Comparison.Volatilities';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r300');


% Figure Plot

figure(23);
StandardizedResid = Save_Other(:, 2) ./ sqrt(VolatilityDCS_t);
WeightsPlot = Save_Other(:, 1);
scatter(StandardizedResid, WeightsPlot);
xlabel('\epsilon / \sigma');
ylabel('weights');
xlim([-max(abs(StandardizedResid)) max(abs(StandardizedResid))]);

fname= 'Weights.residual_DCS'; 
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r300');

%% PROBABILITY OF BEING AN OUTLIER
% Compute the Weight for Prediction Error Equal to 0 (w(0))
w0 = WeightsPlot(StandardizedResid == 0);
% If there is no exact zero in StandardizedResid, interpolate w(0)
if isempty(w0)
    w0 = interp1(StandardizedResid, WeightsPlot, 0, 'linear', 'extrap');
end

% Compute the Probability of Observing Outliers
outlier_prob = 1 - abs(WeightsPlot / w0);

% Display Results
fprintf('Standardized Prediction Error (spe_t):\n');
disp(StandardizedResid);

fprintf('Weights (w(spe_t)):\n');
disp(WeightsPlot);

fprintf('Probability of Observing Outliers:\n');
disp(outlier_prob);

%% Plot the Standardized Residuals and Outlier Probabilities
figure(24);
subplot(2, 1, 1);
plot(StandardizedResid, 'b-', 'LineWidth', 1.5);
title('Standardized Prediction Errors ');
xlabel('Time');
ylabel('Standardized Residual');
grid on;
% Plot Outlier Probabilities
subplot(2, 1, 2);
stem(outlier_prob, 'r', 'filled', 'LineWidth', 1.5);
title('Probability of Observing Outliers');
xlabel('Time');
ylabel('Probability');
grid on;

fname='Standardized Prediction Errors and Outlier Probabilities';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r1200');

figure;
posP = outlier_prob(outlier_prob>0);
histogram(posP)
fname='Outlier_hist';
print(gcf, fullfile(folder, [fname mon_name]), '-dpng', '-r1200');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPARISON SERIES
figure;
plot(mdate(1:168), MP(1:168),     'LineWidth',1.2); hold on;
plot(mdate(1:168), Target_scaled(1:168), 'LineWidth',1.2);
hold off;
xlim(mdate([1 168]));                    % use actual date limits
datetick('x','yyyy-mm-dd','keepticks');  % or choose your format
legend('MP','Target_{norm}','Location','best');
xlabel('Time'); ylabel('Value');
title('MP vs. Target');
fname = 'MP.vs.Target.png';
print(gcf, fullfile(folder,fname), '-dpng', '-r300');

%
figure;
plot(mdate(1:168),MP(1:168), 'LineWidth',1.2); hold on;
plot(mdate(1:168),Timing_scaled(1:168), 'LineWidth',1.2);
hold off;
xlim(mdate([1 168]));                    % use actual date limits
datetick('x','yyyy-mm-dd','keepticks'); 
legend('MP','Timing_{norm}','Location','best');
xlabel('Time'); ylabel('Value');
title('MP vs. Timing');
fname='MP.vs.Timing.png';
print(gcf, fullfile(folder, [fname]), '-dpng', '-r300');
%
figure;
plot(mdate(1:168),MP(1:168), 'LineWidth',1.2); hold on;
plot(mdate(1:168),Timing_scaled(1:168), 'LineWidth',1.2); hold on;
plot(mdate(1:168),Target_scaled(1:168), 'LineWidth',1.2); hold on;
hold off;
xlim(mdate([1 168]));                    % use actual date limits
datetick('x','yyyy-mm-dd','keepticks'); 
legend('MP','Timing_{norm}','Target_{norm}','Location','best');
xlabel('Time'); ylabel('Value');
title('MP vs. Timing vs Target');
fname='MP.vs.Others.png';
print(gcf, fullfile(folder, [fname]), '-dpng', '-r300');

%
figure;
plot(mdate(1:size(MP,1)),MP, 'LineWidth',1.2); hold on;
plot(mdate(1:size(MP,1)),Jaro_Median, 'LineWidth',1.2);
hold off;
xlim(mdate([1 size(MP,1)]));                    % use actual date limits
datetick('x','yyyy-mm-dd','keepticks'); 
legend('MP','Jarocinsky','Location','best');
xlabel('Time');
ylabel('Value');
title('MP vs. Jaro Median');
fname = 'MP_vs_Jaro_Median.png';
print(gcf, fullfile(folder, fname), '-dpng', '-r300');

%
figure;
plot(mdate(1:168),MP(1:168), 'LineWidth',1.2); hold on;
plot(mdate(1:168),FG_scaled(1:168), 'LineWidth',1.2);
hold off;
xlim(mdate([1 168]));                    % use actual date limits
datetick('x','yyyy-mm-dd','keepticks'); 
legend('MP','FG_{norm}','Location','best');
xlabel('Time'); ylabel('Value');
title('MP vs. FG');
fname='MP.vs.FG.png';
print(gcf, fullfile(folder, [fname]), '-dpng', '-r300');


figure;
plot(mdate(1:168),MP(1:168), 'LineWidth',1.2); hold on;
plot(mdate(1:168),QE_scaled(1:168), 'LineWidth',1.2);
hold off;
xlim(mdate([1 168]));                    % use actual date limits
datetick('x','yyyy-mm-dd','keepticks'); 
legend('MP','QE_{norm}','Location','best');
xlabel('Time'); ylabel('Value');
title('MP vs. QE');
fname='MP.vs.QE.png';
print(gcf, fullfile(folder, [fname]), '-dpng', '-r300');

