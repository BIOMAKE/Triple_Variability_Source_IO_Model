% This script aims to use cross-validation to show the robusteness of the
% optimisation method, i.e., PSO.
clear
addpath("Model_Validation\")


%% Generate simulation data:
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

%% Cross-validation: 30-folds
Kfold = 30;
cvIdx = crossvalind('kfold', y_axis, Kfold);
resultTable = struct('modelOptResults', [], 'BIC', []);

tic
for icnt = 1:Kfold
    % extract training dataset
    testIdx = (cvIdx == icnt);
    trainIdx = ~testIdx;
    Dtrain_y = y_axis(trainIdx);
    Dtrain_x = x_axis(trainIdx);

    % call class
    curveModel = Logistic3P3VCurveModel_GEV();
    curveModel.initialiseModel(Dtrain_x, Dtrain_y);

    % cost function
    objFunction = @(parameters) curveModel.likelihoodObjFunction([parameters, gev_para], false);

    % optimisation settings
    obj.maximumIteration = 2000;
    obj.maximumFunctionValue = 7000;
    obj.functionTolerance = 1e-6;
    obj.stopTolerance = 1e-6;
    obj.trueModelParameters = modelParameters;
    obj.iniPoints = curveModel.opti_iniPoints;

        
    % optimisation method: particle swarm
    % set the lower and upper bounds
    lb = curveModel.opti_bounds(1, 1:5); % lower bound
    ub = curveModel.opti_bounds(2, 1:5); % upper bound
    
    % the number of parameters
    nvar = length(lb);
    % optimisation method
    options = optimoptions('particleswarm', 'Display', 'iter', 'SwarmSize', 150, 'UseParallel', ...
        true, 'MaxIterations', obj.maximumIteration, 'FunctionTolerance', obj.functionTolerance);
    [obj.opti_parameters_maximumlikelihood, obj.opti_fval_maximumlikelihood,...
        obj.opti_exitflag_maximumlikelihood, obj.opti_output_maximumlikelihood] = ...
        particleswarm(objFunction, nvar, lb, ub, options);

    %
    resultTable(icnt).modelOptResults = obj;
    resultTable(icnt).BIC = 2*obj.opti_fval_maximumlikelihood + length(obj.opti_parameters_maximumlikelihood) * log(length(x_axis));


end

toc


save("results_v1.mat", "resultTable")