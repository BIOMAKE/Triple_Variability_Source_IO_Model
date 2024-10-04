classdef Logistic3P3VCurveModel_GEV < handle
    %%
    % The curve model function is
    % f(x) = p1 / ( 1 + exp(-p2*(x - p3)) )
    % with three sources of variability
    %
    % Normal distribution:
    % sigma_vy - standard deviation of vy, p4
    % sigma_vx - standard deviation of vx, p5
    %
    %
    % Generalised extreme value distribution:
    % k_gev - the shape parameter, p6
    % sigma_gev - the scale parameter, p7
    % mu_gev - the location parameter, p8
    %
    %
    % Variability Sources:
    % vx - additive noise at input side (normal distribution)
    % vy - multiplicative noise at output side (normal distribution)
    % vadd - additive noise at output side (GEV distribution)

    % dataset
    properties
        x_axis % raw data
        y_axis % raw data
        x_unique % arranged data
        y_arrange % arranged data
    end

    %%
    % find the initial points for later maximum-likelihood (MLE) optimisation
    % properties: nonlinear regression optimisation (least-square method)
    % 3 parameters in total
    properties
        opti_parameters_regression % the optimal parameters for the curve model using regresison method
        opti_fval_regression % objective function value at solution
        opti_exitflag_regression % reason why algorithm stop
        opti_output_regression % algorithm outputs
    end

    %% optimisation range
    properties
        opti_bounds % optimisation upper and lower bounds
        opti_iniPoints % the initial points for the model
    end

    %% own methods
    methods
        %% load the class
        function obj = Logistic3P3VCurveModel_GEV()
        end

        %% initialise the class and find the initial opti points for later MLE
        function initialiseModel(obj, xivec, yivec)
            % save x- and y-axis
            obj.x_axis = xivec;
            obj.y_axis = yivec;

            % Re-arrange the input data into cell array
            % Group yivec by xivec
            obj.x_unique = unique(xivec);
            obj.y_arrange = cell(1, length(obj.x_unique));
            for icnt = 1:length(obj.x_unique)
                obj.y_arrange{icnt} = yivec(xivec == obj.x_unique(icnt));
            end

            % Run linear regression for the model
            obj.runLinearRegression();
        end


        %% Run the normal linear regression
        function runLinearRegression(obj)
            % inputs
            xivec = obj.x_axis;
            yivec = obj.y_axis;

            % objective function
            objFunction_regression = @(parameters)...
                sum((10.^(Logistic3P3VCurveModel_GEV.curveModelFunction(parameters, xivec)) - yivec).^2);

            % number of parameters
            nvar = 3;
            % set the lower and upper bounds
            p1 = [min(obj.y_axis); max(obj.y_axis)]; p2 = [eps; 10];
            p3 = [eps; 150]; % the shift range should be extended since some subjects have higher cut-off point.
            bounds = [p1, p2, p3];
            lb = bounds(1, :); % lower bound
            ub = bounds(2, :); % upper bound
            % optimisation method: particle swarm algorithm
            options = optimoptions('particleswarm', 'SwarmSize', 150);
            [obj.opti_parameters_regression, obj.opti_fval_regression, obj.opti_exitflag_regression, obj.opti_output_regression] = ...
                particleswarm(objFunction_regression, nvar, lb, ub, options);

            % update new parameters with residual distribution (vy);
            obj.opti_iniPoints = [obj.opti_parameters_regression, exprnd(2), exprnd(2),...
                rand(1), min(obj.y_axis), min(obj.y_axis)];

            % update optimisation bounds
            % the mean of GEV is not finit when k >= 1, and the variance of
            % GEV is not finit when k >= 0.5
            p4 = [1e-10; 20]; % sigma for vy
            p5 = [1e-10; 20]; % sigma for vx
            % gev parameters
            p6 = [1e-10; 0.99]; % the shape parameter, k, for extreme value distribution, since x > -sigma_ev/k + mu_ev > 0
            p7 = [1e-10; 0.99]; % sigma for gev, scale parameter
            p8 = [1e-10; 1]; % mu for gev, location parameter
            obj.opti_bounds = [bounds, p4, p5, p6, p7, p8];
        end


        %% Likelihood objective function
        function loglikelihoodValue = likelihoodObjFunction(obj, parameters, stopFlag)
            % Input pair:
            % obj.x_unique and obj.y_arrange
            %
            % Variability sources:
            % vx - additive noise at the input side
            % vy - multiplicative noise at output side
            % vadd = additive noise at output side
            %
            % Options:
            % stopFlag - to stop optimisation during running
            % vx_selected: consider variability along x-axis

            %% if stop optimisation during running
            if stopFlag
                error('Stop Optimisation.')
            end

            %% Calculate the log-likelihood
            % ensure the GEV distribution strictly obeys the domain
            xivec = obj.x_unique; yivec = obj.y_arrange;
            likelihoodValue = 0;
            for icnt = 1:length(xivec)
                % calculate the probability distribution for a given x
                [likelihood_fulldistribution, Youtput_corresp] = ...
                    Logistic3P3VCurveModel_GEV.calculateLikelihoodDistribution(parameters, xivec(icnt), 2*max(obj.y_axis));
                % find the corresponding likelihood for y
                y_likelihood = interp1(Youtput_corresp, likelihood_fulldistribution, yivec{icnt}, 'linear');
                y_likelihood(isnan(y_likelihood)) = eps; % to avoid any errors
                y_likelihood(y_likelihood == 0) = eps;
                %
                likelihoodValue = likelihoodValue + sum(log(y_likelihood));
            end
            % negative log-likelihood value
            loglikelihoodValue = -likelihoodValue;
        end


        %% Visualise the model curve
        function visualisation(obj, axeHandle, modelParameters)
            % settings
            hold(axeHandle, 'on');
            xrange = range(obj.x_axis);
            x = linspace(min(obj.x_axis) - 0.05*xrange, max(obj.x_axis) + 0.05*xrange, 1000);

            % sample
            scatter(axeHandle, obj.x_axis, obj.y_axis, 40,...
                'black', 'Marker', 'x', 'LineWidth', 0.6);

            % noise level: mode of GEV
            stat_gev = Logistic3P3VCurveModel_GEV.calculateGEVMode(modelParameters);

            % logistic curve
            plot(x, 10.^(Logistic3P3VCurveModel_GEV.curveModelFunction(modelParameters, x)) + stat_gev,...
                'LineStyle', '-', 'Color', 'red', 'LineWidth', 1);
            
            % noise level
            yline(stat_gev, '--', 'Noise level', 'Color', 'red', 'LineWidth', 1, 'Interpreter', 'latex')

            % set the ylim
            ylim(axeHandle, [stat_gev/2, modelParameters(1)*2])

            % legend
            legend('Sample Points', 'Model Curve', 'Location', 'best', 'Interpreter', 'latex')
            xlabel('Stimulus Strength, x', 'Interpreter', 'latex')
            ylabel('MEP (mV)', 'Interpreter', 'latex')
            %
            set(axeHandle, 'YScale', 'log', 'XGrid', 'on', 'YGrid', 'on', 'YMinorGrid', 'on')
            %
            hold(axeHandle, 'off')
        end



    end

    %% static methods
    methods(Static)
        %% Logistic function
        function FunValue = curveModelFunction(parameters, xivec)
            % p1 - parameters(1), p2 - parameters(2), p3 - parameters(3)
            % FunValue will never be negative
            FunValue = log10(parameters(1) ./ ( 1 + exp( -parameters(2) .* (xivec - parameters(3)) ) ));
        end

        %% Inverse logistic function
        function FunValue = inverseCurveModelFunction(parameters, yivec)
            % For the inverse function, the range of yivec belongs to (0, p1)
            % p1 - parameters(1), p2 - parameters(2), p3 - parameters(3)
            FunValue = - 1/parameters(2) .* log( (parameters(1) - 10.^yivec) ./ 10.^yivec ) + parameters(3);
        end


        %% Calculate the mode of GEV
        function FunValue = calculateGEVMode(parameters)
            k_gev = parameters(6); sigma_gev = parameters(7); mu_gev = parameters(8);
            if k_gev ~= 0
                FunValue = mu_gev + sigma_gev * ((1 + k_gev) ^ - k_gev - 1) / k_gev;
            else
                FunValue = mu_gev;
            end
        end


        %% Calculate the mean of GEV
        function FunValue = calculateGEVMean(parameters)
            [FunValue, ~] = gevstat(parameters(6), parameters(7), parameters(8));
        end


        %% Calculate the variance of GEV
        function FunValue = calculateGEVVar(parameters)
            [~, FunValue] = gevstat(parameters(6), parameters(7), parameters(8));
        end


        %% Calculate the lowest boundary for GEV
        function FunValue = calculateGEVLowest(parameters)
            if parameters(6) ~= 0
                FunValue = -parameters(7)/parameters(6) + parameters(8);
            else
                FunValue = NaN;
            end
        end


        %% Likelihood distribution function: backward calculation with convolution
        function [likelihood_fulldistribution, Youtput_corresp] = calculateLikelihoodDistribution(parameters, xivec, ymax)
            % This probablity function could be used to construct the
            % likelihood function for a given xivec.
            %
            % The whole system follows:
            % x_tilde = x + vx, y_tilde = S(x_tilde),
            % y_hat = y_tilde + vy, MEP = exp(y_hat) + vadd;
            % exp transformation with base = 10;
            %
            % Parameters:
            % p(1), p(2), p(3) - model parameters
            % p(4) - sigma_vy,
            % p(5) - sigma_vx.
            % p(6) - the shape parameter, k, for gev
            % p(7) - the scale parameter, sigma, for gev
            % p(8) - the location parameter, mu, for gev
            %
            % Input argument:
            % parameters - the parameters of the model
            % xivec - the given x point

            % define the numbers of sample
            numSample_Youtput = 5000;
            numSample_Yrange = 5000;

            % Define function handles
            myinversefunction = @(y) Logistic3P3VCurveModel_GEV.inverseCurveModelFunction(parameters, y);

            % define the distribution parameters
            k_gev = parameters(6); sigma_gev = parameters(7); mu_gev = parameters(8);
            sigma_vy = parameters(4); sigma_vx = parameters(5);

            % Define the range of y output
            % Carefully! the defined range should cover the range of
            % probability distribution, otherwise numeric results will go
            % wrong!
            % NOTE: The range of Youtput and Yrange can highly affect the
            % performace of optimisation.
            Youtput = linspace(-0.1, ymax, numSample_Youtput); % this is strickly positive
            Youtput_diff = mean(diff(Youtput));

            % Define the range of y before exp transformation
            Yrange = linspace(log10(eps), log10(Youtput(end)), numSample_Yrange);
            Yrange_diff = mean(diff(Yrange));

            if length(Youtput) ~= length(unique(Youtput))
                disp('Error')
            end

            %% Part I: Model transformation
            % the corresponding vy probability
            PDF_vy = normpdf(Yrange - Yrange(end/2), 0, sigma_vy);
            PDF_vy = PDF_vy / (sum(PDF_vy) * Yrange_diff);

            % calculate the corresponding x range
            x_tilde = real(myinversefunction(Yrange));
            x_tilde(Yrange > log10(parameters(1))) = xivec + 100*sigma_vx;

            % calculate the transformed vx probability
            CDF_vx = normcdf(x_tilde - xivec, 0, sigma_vx);
            CDF_vx_transformed = [0, diff(CDF_vx)] / Yrange_diff;
            CDF_vx_transformed_correct = CDF_vx_transformed / sum(CDF_vx_transformed * Yrange_diff);

            % convolution before exp transformation
            % Note that the probability density of PDF_vy and CDF_vx_transformed should be
            % equally spaced according to the array of vy. (Equal sampling);
            % Therefore, we can ensure that the probability of y is correctly calculated
            % using the corresponding f_vx(S^{-1}(y - vy) - xivec)*f_vy(vy).
            fulldistrib = Yrange_diff * conv(PDF_vy, CDF_vx_transformed_correct, 'full');
            %fulldistrib = fulldistrib / sum(fulldistrib * Yrange_diff);
            Yrange_corresp = linspace(min(Yrange) + min(Yrange - Yrange(end/2)), ...
                max(Yrange) + max(Yrange - Yrange(end/2)), 2*numSample_Yrange - 1);

            % calculate the CDF for full distribution before exp
            CDF_fulldistrib = cumsum(fulldistrib) * Yrange_diff;


            %% Part II: exponential transformation
            % calculate the PDF for additive noise, generalised extreme
            % value distribution
            % Because gev have a domain problem, we need to correct NaN to
            % zero.
            PDF_vadd = gevpdf(Youtput, k_gev, sigma_gev, mu_gev);
            PDF_vadd = PDF_vadd / sum(PDF_vadd * Youtput_diff);

            % calculate the corresponding y_hat depending on y
            y_hat = real(log10(Youtput));
            fulldistrib_interpret = interp1(Yrange_corresp, CDF_fulldistrib, y_hat, 'linear');
            fulldistrib_interpret(Youtput <= 0) = 0;
            fulldistrib_interpret(isnan(fulldistrib_interpret)) = eps;
            fulldistrib_transformed = [eps, diff(fulldistrib_interpret)] ./ Youtput_diff;

            % convolution after exp transformation
            likelihood_fulldistribution = Youtput_diff * conv(PDF_vadd, fulldistrib_transformed, 'full');
            likelihood_fulldistribution = likelihood_fulldistribution / sum(likelihood_fulldistribution * Youtput_diff);
            Youtput_corresp = linspace(2*min(Youtput), 2*max(Youtput), 2*numSample_Youtput - 1);
        end


        %% calculate quantiles for error band
        function Yquantile = calculateQuantile(parameters, xivec)
            % Input argument:
            % parameters - model parameters;
            % xmax - define the maximum value of the x range;

            %
            % Output:
            % Yquantile - a table summarises the calculated quantile of y

            % calculate quantiles for error band
            quantile_vec = [0.05/2, 0.15/2, (1-0.15/2), (1-0.05/2)]; % 0.95 and 0.85 variability range
            quantile_y = zeros(length(xivec), length(quantile_vec)); % create an array

            % set the maximum y
            ymax = 10^Logistic3P3VCurveModel_GEV.curveModelFunction(parameters, 100) + ...
                Logistic3P3VCurveModel_GEV.calculateGEVMean(parameters);


            % calculate y distribution at each x point
            for i = 1:length(xivec)
                % calculate the cumulative probability for y
                [YprobDistribution, Youtput_corresp]  = Logistic3P3VCurveModel_GEV.calculateLikelihoodDistribution(parameters, ...
                    xivec(i), 2*ymax);
                YcumDistribution = mean(diff(Youtput_corresp)) * cumsum(YprobDistribution);

                % unique sample points
                [~, idx, ~] = unique(YcumDistribution, 'stable');

                % find corresponding y closest to Yquantile
                quantile_y(i, :) = interp1(YcumDistribution(idx), Youtput_corresp(idx), ...
                    quantile_vec, 'linear');
            end

            % arrange table
            Yquantile = array2table([xivec', quantile_y], 'VariableNames', {'x', '0.025', '0.075', '0.925', '0.975'});
        end


    end



end