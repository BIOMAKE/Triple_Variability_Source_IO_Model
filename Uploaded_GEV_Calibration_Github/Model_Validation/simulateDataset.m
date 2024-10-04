function [x, y] = simulateDataset(modelParameters, Inputs, noVx)

rng('default')

% model parameters
% for lgrithmic sigmoidal equation
p1 = modelParameters(1); 
p2 = modelParameters(2); 
p3 = modelParameters(3);

% vx and vy 
sigma_y = modelParameters(4);
sigma_x = modelParameters(5);

% GEV
k_gev = modelParameters(6);
sigma_gev = modelParameters(7);
mu_gev = modelParameters(8);

% x range - [0, 100]
x = Inputs;
noInputs = length(Inputs);
%
x = repmat(x, noVx, 1);
x = reshape(x, 1, numel(x));

% variability along x-axis, additive
vx = normrnd(0, sigma_x, 1, noVx*noInputs);

% variability along y-axis, multiplicative
vy = normrnd(0, sigma_y, 1, noInputs*noVx);

% variability along y-axis, gev distribution
vadd = gevrnd(k_gev, sigma_gev, mu_gev, 1, noInputs*noVx);

% input of the model
xivec = x + vx;

% output of the model
yivec = log10(p1 ./ ( 1 + exp( -p2 .* (xivec - p3))));

% real outputs
y_hat = yivec + vy;

% Exp transformation
y = 10.^y_hat + vadd;

end