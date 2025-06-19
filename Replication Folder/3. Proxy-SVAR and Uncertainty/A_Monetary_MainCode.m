clear
clc
close all
addpath('C:\Users\Utente\Dropbox\Thesis\macroData\NewData')
savepath  
%% 1) Load Data
data2=readtable('ECB_and_Uncertainty_FinalProxy.xlsx','ReadRowNames',true);
data3=readtable('comparison_ecb.xlsx','ReadRowNames',true);
ecb_2=data3.ecb_2;
ecb_2_rob=data3.ecb_2_robust;
ecb_5=data3.ecb_5yr;
ecb=data3.ecb;
Uncertainty=data2.GARCHVolatility;
CumUncertainty=cumsum(Uncertainty);

MPshock=-Uncertainty(:,:);
%MPshock=-MPshock/MPshock(1); %To normalize

%log_MP=100*log(MPshock);

%cleaning the MP proxy!!
%
Q1   = prctile(MPshock,25);
Q3   = prctile(MPshock,75);
IQR  = Q3 - Q1;
idx  = MPshock < Q1-1.5*IQR | MPshock > Q3+1.5*IQR;
MPshock(idx) = 0;
%}
ldata=readtable('MacroData.xlsx','ReadVariableNames',true);
%ldata=ldata(2:240,:);
CumMP=cumsum(MPshock);
%CumMP=CumMP/CumMP(1);
[P_trend, Stat_CumMP] = hpfilter(CumMP, Smoothing=129600); %% SIMPLY DE-TRENDING CumMP in order to be stationary
[h_adf,pValue] = adftest(Stat_CumMP);
%Stat_CumMP=-Stat_CumMP/Stat_CumMP(1); %To normalize
mon_dummy=ldata.monetary_dummy2(1:end-2);
mon_dummy5=ldata.monetary_dummy5(1:end-2);

% 
LOGDATA=readtable("100Log_MacroData.xlsx","ReadVariableNames",true);
Y=[LOGDATA(10:end-2,2:end-1)];

%Y = [ldata(:, [22 13 26 24 28])];
%Y = [ldata(:, [23 14 27 25 39])];
Y=table2array(Y);
Y=[Y CumUncertainty];
Y = rmmissing(Y);
var_names = {
  'Industrial Production'
  'HICP (EA) '
  'Intermediate Goods value'
  'EURO STOXX 50 return'
  'Exchange Rate'
  'Cumulative MP proxy'
};
o=size(Y,2); 
C=corrcoef(Y);

%% 2) Parameters
n=size(Y,2); 
H=48; % horizon of the IRF
c=1;
MaxBoot=200;
cl=0.9;
ind=[];
opt=1;
signif_level=0.9;
% Lag selection
maxLag = 12; 
criteria = zeros(maxLag, 3); % Columns for AIC, BIC, HQIC
for lag = 1:maxLag
    % Estimate VAR model with lag 'lag'
    [~, ~, ~, logL] = estimate_VAR(Y, lag, c);
    % Compute the criteria
    T = size(Y, 1) - lag; % Effective sample size after lags
    k = n^2 * lag + n; 
    criteria(lag, 1) = -2 * logL / T + 2 * k / T;       % AIC
    criteria(lag, 2) = -2 * logL / T + log(T) * k / T;  % BIC
    criteria(lag, 3) = -2 * logL / T + 2 * log(log(T)) * k / T; % HQIC
end

% Select the optimal lag based on minimum BIC (or choose AIC/HQIC)
[~, p_bic] = min(criteria(:, 2)); 
disp(['Optimal lag (BIC): ', num2str(p_bic)]);
[~, p_hqic] = min(criteria(:, 3)); 
disp(['Optimal lag (HQIC): ', num2str(p_hqic)]);

[~, p_aic] = min(criteria(:, 1));
disp(['Optimal lag (AIC): ', num2str(p_aic)]);
p=p_aic;
%using AIC optimal p since we are working in small samples
%addpath('C:\Users\Utente\Desktop\A-UNITO\Second Year\Monetary\Gambetti\MATLAB\USED FILE')
[wirf,res] = WoldBoot(Y,p,H,c);

%% Proxy-SVAR SVAR-IV approach
Z=MPshock; 
Z=Z(p+1:end);
Z=Z(~isnan(Z));
mon_dummy1=mon_dummy5(p+1:end);
Z1=Z.*mon_dummy1;
res1=res(~isnan(Z),:);
res1=res1.*mon_dummy1;


% Regression on epsilon onto the instrument
b=(inv(Z1'*Z1)*Z1'*res1)';
b=b/b(o);

for j=1:H
    mpirf(:,j)=wirf(:,:,j)*b;
end

%[irfs,irfsboot] = PROXYSVARRecoverabilityCase(Y,Z,p,H,1,ind,MaxBoot,4,1,0,Y);

%% Plot
figure(1);
plot(mpirf', 'LineWidth', 1.5); % Plot all impulse responses with thicker lines
grid on;
xlabel('Horizon (months)', 'FontSize', 12);
ylabel('Impulse Response', 'FontSize', 12);
title('Impulse Response Functions (IRFs)', 'FontSize', 14);
legend(var_names, 'Location', 'best', 'FontSize', 10);
saveas(figure(1),'Proxy-SVAR.png')


%% Plagborg Moller code 

plot_var=3;
plot_xticks      = 0:12:H; 
plot_band_xlabel = 'months after shock';
plot_band_ylabel = '';

numdraws_supt        = 1e5; % Number of normal draws used to compute plug-in sup-t crit. val.

numdraws_boot        = 1e4; % Set to 0 if bootstrap inference undesired

verbose = true;             % Print progress of bootstrap procedure

rng(20170114);              % Seed for random number generator

% Bands

band_list            = {'Pwise',...
                        'supt'};
                        %'Sidak',...
                        %'Bonferroni'}; %, 'thetaproj', 'muproj'}; 
                            % Cell array of bands to be plotted, 
                            % can be any combination of: 'Pwise', 'supt', 'Sidak', 'Bonferroni', 'thetaproj', 'muproj'

legend_bands         = {'Pointwise',...
                        'Sup-t: plug-in'};
                        %'Sidak',...
                        %'Bonferroni'}; %, '\theta-projection', '\mu-projection'}; 
                            % Legend for bands

linestyle_supt_bands = {'-', '--'};

%% Reduced-form VAR estimation

redf                 = iv_RedForm(Y, Z1, p); 
                                      % Struct redf contains reduced-form VAR
                                      % and IV objects
%% PLOT ALL THE IRFs with Plagborg-Moller

figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);

    
for plot_var = 1:o
    % IV estimation (if independent of plot_var, move before loop)
    [Theta, Sigmahat, p] = iv_estim(redf, H);
    %scale    = Theta(end,1);        % IRF of last variable at h=0
    %Theta_sc = Theta ./ scale;      % rescaled IRFs
    
    % select IRF and compute bands
    sel = select_IRF(o,1,H,plot_var,1);
    bands_plugin = SimInference.bands_plugin( ...
        Theta(sel)', Sigmahat(sel,sel), p, band_list, numdraws_supt, 1-signif_level);

    % subplot
    subplot(3,2,plot_var);
    plot_compare({Theta(sel)'}, bands_plugin, plot_band_xlabel, plot_band_ylabel, plot_xticks, {});
    title(var_names{plot_var}, 'FontSize', 12);
end

saveas(gcf, 'AllVars_Bands_IV_Plug_in.png');
hhh
%% PLOT ALL THE IRFs with coloured bands

figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
for plot_var = 1:o
    % IV estimation
    [Theta, Sigmahat, p] = iv_estim(redf, H);

    % select IRF and compute bands (returns a 1×2 cell of 2×T arrays)
    sel   = select_IRF(o,1,H,plot_var,1);
    bands = SimInference.bands_plugin( ...
        Theta(sel)', Sigmahat(sel,sel), p, band_list, numdraws_supt, 1-signif_level);

    % unpack bands cell
    b1   = bands{1};   % inner band: 2×T (row1=lower, row2=upper)
    b2   = bands{2};   % outer band: 2×T
    lb1  = b1(1,:);    ub1 = b1(2,:);
    lb2  = b2(1,:);    ub2 = b2(2,:);

    % IRF and x-axis
    irf = Theta(sel)';               
    x   = 0:(length(irf)-1);

    % plot
    subplot(3,2,plot_var);
    hold on;
    % inner band (blue)
    fill([x fliplr(x)], [ub1 fliplr(irf)], 'b',       'FaceAlpha', .2, 'EdgeColor','none');
    fill([x fliplr(x)], [irf fliplr(lb1)], 'b',       'FaceAlpha', .2, 'EdgeColor','none');
    % outer band (orange)
    fill([x fliplr(x)], [ub2 fliplr(ub1)], [1 .5 0],   'FaceAlpha', .2, 'EdgeColor','none');
    fill([x fliplr(x)], [lb1 fliplr(lb2)], [1 .5 0],   'FaceAlpha', .2, 'EdgeColor','none');
    % central IRF
    plot(x, irf, 'k', 'LineWidth',1.5);
    plot(x, ub1, 'b--',   'LineWidth',1);           % inner upper (blue dashed)
    plot(x, lb1, 'b--',   'LineWidth',1);           % inner lower
    plot(x, ub2, '-',     'Color',[1 .5 0], 'LineWidth',1);  % outer upper (orange solid)
    plot(x, lb2, '-',     'Color',[1 .5 0], 'LineWidth',1);  % outer lower
    xlim([0 x(end)]);
    xlabel(plot_band_xlabel);
    ylabel(plot_band_ylabel);
    title(var_names{plot_var}, 'FontSize',12);
    hold off;
end

saveas(gcf, 'OIS2_Final_IRFs.png');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LOCAL PROJECTION

% ) Estimation of LP
p=4;
[Ytemp, Xtemp] = LagY(Y, p); 
L= MPshock; 
zz = L(p+1:end); 

for i = 1:n
    for j = 1:H
        Y_sub = Ytemp(j:end, i);
        X = [ones(size(Y_sub, 1), 1), zz(1:end-j+1), Xtemp(1:end-j+1, 2:end)];
        b = (X' * X) \ (X' * Y_sub); % Regression coefficients
        irf(j, i) = b(2); 
    end
end
irf = irf / irf(1, o); % Normalize IRF

% Plot LP
plot (irf)

%% Plot SVAR-IV and LP IRFs Side by Side
figure;

% SVAR-IV Plot
subplot(1, 2, 1); % Create a subplot (1 row, 2 columns, position 1)
plot(mpirf', 'LineWidth', 1.5); % Plot all impulse responses with thicker lines
grid on;
xlabel('Horizon (months)', 'FontSize', 12);
ylabel('Impulse Response', 'FontSize', 12);
title('SVAR-IV: Impulse Response Functions (IRFs)', 'FontSize', 14);
legend(var_names, 'Location', 'best', 'FontSize', 10);

% LP Plot
subplot(1, 2, 2); 
plot(irf, 'LineWidth', 1.5); 
grid on;
xlabel('Horizon (months)', 'FontSize', 12);
ylabel('Impulse Response', 'FontSize', 12);
title('LP: Impulse Response Functions (IRFs)', 'FontSize', 14);


% Adjust layout if needed
set(gcf, 'Position', [100, 100, 1200, 500]); 
saveas(gcf, 'ProxySVAR-LP.png'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%% Proxy-SVAR SVAR-IV approach without Jarocinski
Z=MPshock; 
Z=Z(p+1:end);
Z=Z(~isnan(Z));
%mon_dummy1=mon_dummy5(p+1:end);
%Z1=Z.*mon_dummy1;
res1=res(~isnan(Z),:);
%res1=res1.*mon_dummy1;


% Regression on epsilon onto the instrument
b=(inv(Z'*Z)*Z'*res1)';
b=b/b(o);

for j=1:H
    mpirf(:,j)=wirf(:,:,j)*b;
end

%[irfs,irfsboot] = PROXYSVARRecoverabilityCase(Y,Z,p,H,1,ind,MaxBoot,4,1,0,Y);

%Plot
figure(1);
plot(mpirf', 'LineWidth', 1.5); % Plot all impulse responses with thicker lines
grid on;
xlabel('Horizon (months)', 'FontSize', 12);
ylabel('Impulse Response', 'FontSize', 12);
title('Impulse Response Functions (IRFs)', 'FontSize', 14);
legend(var_names, 'Location', 'best', 'FontSize', 10);
saveas(figure(1),'Proxy-SVAR_normal.png')


%% WHAT ABOUT USING THE GARCH-t(3,3) AS AN UNCERTAINTY MEASURE?

% THIS CODE NEEDS TO BE CHECKED. THE VARIABLE MATRIX USED NEEDS CHANGING I
% GUESS USING cumUncertainty instead of cumMP
U=Uncertainty; %FOR Uncertainty
U=U(p+1:end);
U=U(~isnan(U));


% Regression on epsilon onto the instrument
bU=(inv(U'*U)*U'*res)';
bU=bU/bU(4);

for j=1:H
    mpirfU(:,j)=wirf(:,:,j)*bU;
end

% Plot for Uncertainty as an instrument
figure(10);
plot(mpirfU', 'LineWidth', 1.5);
grid on;
xlabel('Horizon (months)', 'FontSize', 12);
ylabel('Impulse Response', 'FontSize', 12);
title('IRF using UNCERTAINTY', 'FontSize', 14);

legend(var_names, 'Location', 'best', 'FontSize', 10);

saveas(figure(10),'Proxy-SVAR_Uncertainty.png')