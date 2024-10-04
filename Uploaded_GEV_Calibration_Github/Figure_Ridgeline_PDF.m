% This script aims to plot a rigidline for the pdf given several x
clear
resultTable = importdata("results.mat");


%% Generate simulation data: 300 points in total
modelParameters = [1, 0.45, 60, ...
                   0.2, 3, ...
                   0.1, 0.001, 0.003];
%
Inputs = linspace(20, 100, 25);
noVx = 15;
[x_axis, y_axis] = simulateDataset(modelParameters, Inputs, noVx);

% baseline
Inputs = 0;
noVx = 1000;
[x_base, y_base] = simulateDataset(modelParameters, Inputs, noVx);

gev_para = mygevfit(y_base);

% load model
curveModel = Logistic3P3VCurveModel_GEV();


%% select an example
repIteration = resultTable(end).modelOptResults;
fitted_params = [repIteration.opti_parameters_maximumlikelihood, gev_para];


%% main
fig = figure('Color', [1, 1, 1]);


% subplot 2 3D Ridge Line Plot
x_values = linspace(40, 80, 10);  % Choose 10 x values from 0 to 100
y_max = max(y_axis) * 1.2;  % Set y_max to 120% of the maximum y value in the data
y_values = linspace(0, y_max, 200);  % 200 points for smooth PDF curves

hold on
for i = 1:length(x_values)
    [likelihood, y_range] = curveModel.calculateLikelihoodDistribution(fitted_params, x_values(i), y_max);
    pdf = interp1(y_range, likelihood, y_values, 'linear', 'extrap');
    pdf = pdf / max(pdf);  % Normalize PDF for better visualization
    plot3(repmat(x_values(i), size(y_values)), y_values, pdf, 'LineWidth', 1.5, 'Color', 'black')
end
hold off

view(15, 15)  % Set the view angle
xlabel('x')
ylabel('y')
zlabel('Normalized PDF')
title('Ridge Line Plot of PDF for Different x Values')