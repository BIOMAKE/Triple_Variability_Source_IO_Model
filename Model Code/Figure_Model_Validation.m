% This script aims to use cross-validation to show the robusteness of the
% optimisation method, i.e., PSO.
clear
resultTable = importdata("results_v1.mat");

% cv number
Kfold = length(resultTable);

%% Generate simulation data: 300 points in total
modelParameters = [1, 0.45, 60, ...
                   0.2, 3, ...
                   0.1, 0.001, 0.003];
%
Inputs = linspace(30, 100, 25);
noVx = 20;
[x_axis, y_axis] = simulateDataset(modelParameters, Inputs, noVx);

% baseline
Inputs = 0;
noVx = 1000;
[x_base, y_base] = simulateDataset(modelParameters, Inputs, noVx);
gev_para = mygevfit(y_base);

% load model
curveModel = Logistic3P3VCurveModel_GEV();

%% Figure
fig = figure('Color', [1, 1, 1]);
figWidth = 12;
figHeigth = 15;
set(gcf,'unit','centimeters','position',[1, 2, figWidth, figHeigth],...
    'PaperUnits','centimeters','PaperOrientation','portrait',...
    'PaperSize',[figWidth, figHeigth]);

%
FontSize = 12;

% set parent panels
t = tiledlayout(fig, 2, 1, "TileSpacing", "compact", "Padding", "compact");

% select a representative curve model
repIteration = resultTable(end).modelOptResults;
fitted_params = [repIteration.opti_parameters_maximumlikelihood, gev_para];


%% Subplot 1: Representative fitted Curve
h(1) = nexttile(t, 1, [1, 1]);

% settings
hold(h(1), 'on');
xrange = range(x_axis);
x_smooth = linspace(min(x_axis) - 0.05*xrange, max(x_axis) + 0.05*xrange, 1000);

% sample
scatter(h(1), x_axis, y_axis, 40,...
    'black', 'Marker', 'x', 'LineWidth', 0.6);

% noise level: mode of GEV
stat_gev = Logistic3P3VCurveModel_GEV.calculateGEVMode(modelParameters);

% logistic curve
plot(x_smooth, 10.^(Logistic3P3VCurveModel_GEV.curveModelFunction(modelParameters, x_smooth)),...
    'LineStyle', '-', 'Color', [0.8, 0, 0], 'LineWidth', 2);

% noise level
yline(stat_gev, '--', '$v_\textrm{add}$', 'Color', [0.8, 0, 0], 'LineWidth', 2, 'Interpreter', 'latex', 'FontSize', 12)

% set the ylim
ylim(h(1), [1e-3, modelParameters(1)*3])
xlim(h(1), [25, 105])


% Calculate and plot confidence intervals
Yquantile = curveModel.calculateQuantile(fitted_params, x_smooth);
fill([x_smooth, flip(x_smooth)], [Yquantile{:, '0.025'}; flip(Yquantile{:, '0.975'})], [0.8, 0, 0], 'FaceAlpha', 0.1, 'EdgeColor', 'none')
fill([x_smooth, flip(x_smooth)], [Yquantile{:, '0.075'}; flip(Yquantile{:, '0.925'})], [0.8, 0, 0], 'FaceAlpha', 0.25, 'EdgeColor', 'none')


% legend
lgd = legend(h(1), {'Sample Points', 'Model Curve', '', '$95\%$ Variability', '$85\%$ Variability'}, ...
    'Location', 'northwest', 'Interpreter', 'latex', 'Box', 'off');
xlabel('Stimulus Strength, $x$', 'Interpreter', 'latex')
ylabel('MEP (mV)', 'Interpreter', 'latex')

set(lgd, 'Position', [0.6, 0.65, 0.3, 0.15])


%
set(h(1), 'YScale', 'log', 'XGrid', 'on', 'YGrid', 'on', 'YMinorGrid', 'on')
xtickformat(h(1), '$%g \\%%$')
%
hold(h(1), 'off')


%% Subplot 2: Probability-Probability plot: cdf
h(2) = nexttile(t, 2, [1, 1]);
hold(h(2), 'on')
% fitted model
[fitted_prob, fitted_y] = calculateFittedProbabilities(curveModel, fitted_params, x_axis, y_axis);
% true model
[theoretical_prob, true_y] = calculateFittedProbabilities(curveModel, modelParameters, x_axis, y_axis);

% calculate correlation coefficient between theoretical and estimated probability
corr_PP_rep = corr(theoretical_prob, fitted_prob);
corr_PP = importdata("corr_PP.mat");

% Plot P-P plot
scatter(h(2), theoretical_prob, fitted_prob, 20, 'MarkerEdgeColor', 'none', ...
    'MarkerFaceColor', "#0072BD", 'MarkerFaceAlpha', 0.3, 'Marker', 'o')
plot(h(2), [-1, 2], [-1, 2], 'LineStyle', '--', 'Color', 'black', 'LineWidth', 1.5) 
text(h(2), 0, 0.8, sprintf('$R = %.3f$', corr_PP_rep), 'Interpreter', 'latex', 'FontSize', 14)
hold(h(2), 'off')

% Creat inset for corr_PP
insetPosition = [0.62, 0.135, 0.25, 0.2];
Inset = axes(fig, 'Position', insetPosition);
histogram(corr_PP, 'FaceColor', "#0072BD", 'EdgeColor', 'none')

%
Inset.XTick = [0.993, 0.997];
xlabel(Inset, '$R$', 'Interpreter', 'latex', 'FontSize', 9)
ylabel(Inset, 'Frequency', 'Interpreter', 'latex', 'FontSize', 9)
set(Inset, 'TickLabelInterpreter', 'Latex', 'TickDir', 'none')


%
xlim(h(2), [-0.1, 1.4])
ylim(h(2), [-0.2, 1.1])
xlabel(h(2), 'Theoretical Probability', 'Interpreter', 'latex')
ylabel(h(2), 'Estimated Probability', 'Interpreter', 'latex')

text(h(2), -0.15, 1, 'B', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'latex');



%% Figure settings
text(h(1), -0.15, 1, 'A', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'latex');
text(h(2), -0.15, 1, 'B', 'Units', 'normalized', 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'latex');
set(h, 'TickLabelInterpreter', 'Latex', 'XGrid', 'on', 'YGrid', 'on', 'Box', 'on')





% save figure
exportgraphics(fig, 'Model Validation.pdf', "ContentType", "vector")



%% Calculate median and interquantile values for the estimated parameters
estimatedPara = zeros(Kfold, 5);
for icnt = 1:Kfold
    estimatedPara(icnt, :) = resultTable(icnt).modelOptResults.opti_parameters_maximumlikelihood;
end


estimated_median = median(estimatedPara, 1);
estimated_iqr = iqr(estimatedPara, 1);

%% Calculate parameter deviation from the theoretical values

devDist = zeros(1, Kfold);
for icnt = 1:Kfold
    devDist(icnt) = sum((estimatedPara(icnt, :) - modelParameters(1:5)).^2) / 5;
end



%% Calculate correlation coefficient between theoretical and estimated probability

% corr_PP = zeros(1, Kfold);
% for icnt = 1:Kfold
%     [fitted_prob, fitted_y] = calculateFittedProbabilities(curveModel, [estimatedPara(icnt, :), gev_para], x_axis, y_axis);
%     corr_PP(icnt) = corr(theoretical_prob, fitted_prob);
% end



%% Functions
function [fitted_prob, y_sorted] = calculateFittedProbabilities(curveModel, params, x, y)
[y_sorted, sort_idx] = sort(y);
x_sorted = x(sort_idx);
n = length(y);
fitted_prob = zeros(n, 1);

for i = 1:n
    [likelihood, y_range] = curveModel.calculateLikelihoodDistribution(params, x_sorted(i), max(y)*2);
    cdf = cumsum(likelihood) * mean(diff(y_range));
    fitted_prob(i) = interp1(y_range, cdf, y_sorted(i), 'linear', 'extrap');
end
end