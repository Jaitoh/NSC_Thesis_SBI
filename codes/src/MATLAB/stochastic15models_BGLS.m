function [aToPlot,probRchoice] = ...
    stochastic15models_BGLS(x,dt,C,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,numBruns,bias, debug)
% aToPlot is temporally discretized and also has the first time-point a(0)
if nargin < 9
    debug = false; % assign default value
end

if debug
    rng(111)
end

if ~isnan(x(4,1)) % we have sensory adaptation under this condition (starting from same)
    
    gain10 = x(4,~isnan(x(4,:))); %same
    gain12 = x(5,~isnan(x(5,:))); %opposite
    gain11 = x(6,~isnan(x(6,:))); %else
    
    % Sensory adaptation
    scaleC = max(abs(C));
    if scaleC ~= 0
        C = C/scaleC; %making sure that C is in [0;1] C would be  [0,1,-1...]
        [adaptedC] = sensoryAdaptation5(gain10, gain12, gain11, dt, C, maxClength);
        C = adaptedC*scaleC;
    end

end
        
[aToPlot,probRchoice] = ...
    deterministic_aMany_presetC_base15models(x(1:3,:),dt,C,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,...
    numBruns,bias,debug);

end

function [adaptedC] = sensoryAdaptation5(gain10, gain12, gain11, dt, C, maxClength)

if isempty(gain11)
    gain11TimeDependent = ones(1,numel(C)); %*1 no change would happen
else
    tempDiscrBins = 0:dt:dt*(maxClength-2);% because first pulse is always 0, nothing to sensory adapt there,
        % not by multiplication at least (which is what we do).

    % use Legendre or any other orthogonal polynomials on [0;dt*(numel(C)-1)]
    % use shifted Legendre or any other orthogonal polynomials on [0;dt*(numel(C)-1)]
    affineTransformationBins = -1 + tempDiscrBins*2/(dt*(numel(tempDiscrBins)-1));
    % Now polynomials will be orthogonal on [0;dt*(numel(C)-1)]
    orthogonalPolynomials = zeros(8,numel(affineTransformationBins));
    orthogonalPolynomials(1,:) = ones(size(affineTransformationBins)); %(1,:)
    % Add other rows corresponding to polynomial orders below
    order = numel(gain11) - 1;
    if order >= 1
        orthogonalPolynomials(2,:) = affineTransformationBins; % order 1
    end
    if order >= 2
        orthogonalPolynomials(3,:) = (3*affineTransformationBins.^2 - 1)/2; % order2
    end
    if order >= 3
        orthogonalPolynomials(4,:) = (5*affineTransformationBins.^3 - 3*affineTransformationBins)/2; % order3
    end
    if order >= 4
        orthogonalPolynomials(5,:) = (35*affineTransformationBins.^4 - 30*affineTransformationBins.^2 + 3)/8; %order4
    end
    if order >= 5
        orthogonalPolynomials(6,:) = (63*affineTransformationBins.^5 - 70*affineTransformationBins.^3 +...
            15*affineTransformationBins)/8; %order 5
    end
    if order >= 6
        orthogonalPolynomials(7,:) = (231*affineTransformationBins.^6 - 315*affineTransformationBins.^4 + ...
            105*affineTransformationBins.^2 - 5)/16; %order 6
    end
    if order >= 7
        orthogonalPolynomials(8,:) = (429*affineTransformationBins.^7 - 693*affineTransformationBins.^5 + ...
            315*affineTransformationBins.^3 -35*affineTransformationBins)/16; %order 7
    end
    
    gain11TimeDependent = gain11*orthogonalPolynomials(1:order+1,:);
end

adaptedC = zeros(size(C));
% adaptedC(1) = gain11TimeDependent(1)*C(1); Actually 0 stays 0, so why bother? I do not
% want to get an unconstrained first parameter in gain11TimeDependent

tempDiscrBins = 0:dt:dt*(maxClength-3);% because our sequences always start from 0 =>
    % second pulse is either 0 or 'else' ([0 1... or [0 -1...) so 'same' and
    % 'opposite' can only start working from pulse 3 in the sequence.

affineTransformationBins = -1 + tempDiscrBins*2/(dt*(numel(tempDiscrBins)-1));
% Now polynomials will be orthogonal on [0;dt*(numel(C)-1)]
orthogonalPolynomials = zeros(8,numel(affineTransformationBins));
orthogonalPolynomials(1,:) = ones(size(affineTransformationBins)); %(1,:) 
% Add other rows corresponding to polynomial orders below
order = max(numel(gain10) - 1,numel(gain12) - 1);
if order >= 1
    orthogonalPolynomials(2,:) = affineTransformationBins; % order 1
end
if order >= 2
    orthogonalPolynomials(3,:) = (3*affineTransformationBins.^2 - 1)/2; % order2
end
if order >= 3
    orthogonalPolynomials(4,:) = (5*affineTransformationBins.^3 - 3*affineTransformationBins)/2; % order3
end
if order >= 4
    orthogonalPolynomials(5,:) = (35*affineTransformationBins.^4 - 30*affineTransformationBins.^2 + 3)/8; %order4
end
if order >= 5
    orthogonalPolynomials(6,:) = (63*affineTransformationBins.^5 - 70*affineTransformationBins.^3 +...
        15*affineTransformationBins)/8; %order 5
end
if order >= 6
    orthogonalPolynomials(7,:) = (231*affineTransformationBins.^6 - 315*affineTransformationBins.^4 + ...
        105*affineTransformationBins.^2 - 5)/16; %order 6
end
if order >= 7
    orthogonalPolynomials(8,:) = (429*affineTransformationBins.^7 - 693*affineTransformationBins.^5 + ...
        315*affineTransformationBins.^3 -35*affineTransformationBins)/16; %order 7
end

order = numel(gain10) - 1;
gain10TimeDependent = gain10*orthogonalPolynomials(1:order+1,:);
order = numel(gain12) - 1;
gain12TimeDependent = gain12*orthogonalPolynomials(1:order+1,:);

count10 = 0; count11 = 0; count12 = 0;
for ii = 2:numel(C)
    if C(ii-1) == C(ii) && abs(C(ii)) > 0 % ++ or --
        count10 = count10 + 1;
        count11 = 0; count12 = 0;
        adaptedC(ii) = gain10TimeDependent(count10)*C(ii);

    elseif C(ii-1) == -C(ii) && abs(C(ii)) > 0 % +- or -+
        count12 = count12 + 1;
        count11 = 0; count10 = 0;
        adaptedC(ii) = gain12TimeDependent(count12)*C(ii);
        
    else
        if C(ii-1) == 0
            count11 = count11 + 1;
            adaptedC(ii) = gain11TimeDependent(count11)*C(ii);
        else
            count11 = 0;
            adaptedC(ii) = C(ii);
        end
        count10 = 0; count12 = 0;
    end
end

end

function [aToPlot,probRchoice] = ...
    deterministic_aMany_presetC_base15models(x,dt,C,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,...
    numBruns,bias, debug)

% temporalDiscretization = 100;

CperEachTempDiscrBin = zeros(1,numel(C)*temporalDiscretization);
a = zeros(1,numel(C)*temporalDiscretization+1);


tempDiscrBins = 0:dt:dt*(maxClength-2);% because first pulse is always 0, nothing to influence there,
    % not by multiplication at least (which is what we do). Do not want to
    % create an unconstrained situation.
    % If we get a bias term than it makes sense to have some value of polynomials for 1st pulse (since
    % we'll have something non-zero there, in the accumulator at least).

% use Legendre or any other orthogonal polynomials on [0;dt*(numel(C)-1)]
% use shifted Legendre or any other orthogonal polynomials on [0;dt*(numel(C)-1)]
affineTransformationBins = -1 + tempDiscrBins*2/(dt*(numel(tempDiscrBins)-1));
% Now polynomials will be orthogonal on [0;dt*(numel(C)-1)]
orthogonalPolynomials = zeros(8,numel(affineTransformationBins));
orthogonalPolynomials(1,:) = ones(size(affineTransformationBins)); %(1,:) 
% Add other rows corresponding to polynomial orders below
order = max(sum(~isnan(x),2) - 1);
if order >= 1
    orthogonalPolynomials(2,:) = affineTransformationBins; % order 1
end
if order >= 2
    orthogonalPolynomials(3,:) = (3*affineTransformationBins.^2 - 1)/2; % order2
end
if order >= 3
    orthogonalPolynomials(4,:) = (5*affineTransformationBins.^3 - 3*affineTransformationBins)/2; % order3
end
if order >= 4
    orthogonalPolynomials(5,:) = (35*affineTransformationBins.^4 - 30*affineTransformationBins.^2 + 3)/8; %order4
end
if order >= 5
    orthogonalPolynomials(6,:) = (63*affineTransformationBins.^5 - 70*affineTransformationBins.^3 +...
        15*affineTransformationBins)/8; %order 5
end
if order >= 6
    orthogonalPolynomials(7,:) = (231*affineTransformationBins.^6 - 315*affineTransformationBins.^4 + ...
        105*affineTransformationBins.^2 - 5)/16; %order 6
end
if order >= 7
    orthogonalPolynomials(8,:) = (429*affineTransformationBins.^7 - 693*affineTransformationBins.^5 + ...
        315*affineTransformationBins.^3 -35*affineTransformationBins)/16; %order 7
end


if key_lambdaPolynomial
    orthogonalPolynomials = orthogonalPolynomials(:,1:numel(C)-1); %no value for 1st pulse (since it's always 0)
else
    orthogonalPolynomials = orthogonalPolynomials(:,numel(C)-1); %no value for 1st pulse (since it's always 0)
end
paramsTimeDependent = ones(3,numel(C));
paramsTimeDependent(1,:) = Inf*ones(1,numel(C)); %B
% paramsTimeDependent(2,:) = ones(size(tempDiscrBins)); %G
paramsTimeDependent(3,:) = zeros(1,numel(C)); %L

paramsPerEachTempDiscrBin = zeros(size(x,1),numel(C)*temporalDiscretization);
for paramsIdx = 1:size(x,1)
    if ~isnan(x(paramsIdx,1))
        order = sum(~isnan(x(paramsIdx,:))) - 1;
        paramsTimeDependent(paramsIdx,2:end) = x(paramsIdx,~isnan(x(paramsIdx,:)))*orthogonalPolynomials(1:order+1,:); % adding lines of all orders together
    end
    
    for jj = 1:numel(C) %number of frames in a pulse, but can be any higher number to increase accuracy ????
        paramsPerEachTempDiscrBin(paramsIdx,temporalDiscretization*(jj-1)+1:temporalDiscretization*jj)...
            = paramsTimeDependent(paramsIdx,jj)*ones(1,temporalDiscretization);
    end
end
% interpolate with the same values as point before -

for jj = 1:numel(C) %number of frames in a pulse, but can be any higher number to increase accuracy
    CperEachTempDiscrBin(temporalDiscretization*(jj-1)+1:temporalDiscretization*jj)...
        = C(jj)*ones(1,temporalDiscretization);
end

[a,probRchoice] = simpleEvolution_stochastic_15models(dt/temporalDiscretization,...
    CperEachTempDiscrBin,paramsPerEachTempDiscrBin,sigmas,temporalDiscretization,numBruns,bias,debug);


% aToPlot = a(:,1:temporalDiscretization:numel(CperEachTempDiscrBin)+1);% size = (numSimulations,12): from 0th to 11th timestep
aToPlot = a;
end


function [a,probRchoice] = simpleEvolution_stochastic_15models(dt,C,BGL,sigmas,temporalDiscretization,numBruns,bias,debug)

sigmas(sigmas < 0) = 0;

sigma2a = sigmas(1);
sigma2i = sigmas(2);
sigma2s = sigmas(3);


B = BGL(1,:);
C = C.*BGL(2,:); % C.*G
lambda = BGL(3,:);

aFinal = zeros(1,numBruns);

if sum(sigmas == 0) == numel(sigmas)
    % deterministic case, compute mean:
    a = compute_mean(C, B, lambda, dt);
    
    if a(end) == bias
        probRchoice = 0.5;
    else
        probRchoice = double(a(end) > bias);
    end
    
else %stochastic cases
    if sum(~isinf(B)) > 0
        for idxBruns = 1:numBruns
            
            a = compute_stoch_trace(C, B, dt, lambda, sigma2a, sigma2i, sigma2s, temporalDiscretization, debug);
            aFinal(idxBruns) = a(end);

        end
        probRchoice = sum(aFinal >= bias)/numBruns;
        
        a = compute_mean(C, B, lambda, dt);
        
    else
        
        a = compute_mean(C, B, lambda, dt);
        sigma2dt = compute_variance(C, sigma2a, sigma2i, sigma2s, dt, lambda, temporalDiscretization);
        
        probRchoice = 1 - normcdf(bias,a(end),sqrt(sigma2dt(end)));
        
    end
end

end

function a = compute_stoch_trace(C, B, dt, lambda, sigma2a, sigma2i, sigma2s, temporalDiscretization, debug)
    if debug
        rn = norminv(rand(1,numel(C)), 0, 1);
    else
        rn = randn(1,numel(C));
    end
    dW = sqrt(dt)*rn; 
    
    
    if debug
        rn = norminv(rand(1,numel(C)), 0, 1);
    else
        rn = randn(1,numel(C));
    end
    eta = 1 + sqrt(sigma2s)*rn*sqrt(temporalDiscretization);
    
    a = zeros(1,numel(C) +1);
    da = zeros(1,numel(C));

    
    if debug
        rn = norminv(rand(1,1), 0, 1);
    else
        rn = randn(1,1);
    end
    a(1) = sqrt(sigma2i)*rn; % a(t=0), initial bias before stimulus
    
    for k = 2:numel(C)+1
        if abs(a(k-1)) >= B(k-1) % sticky boundary
            a(k-1:numel(C)+1) = B(k-1)*sign(a(k-1));
            break
        else
            da(k-1) = sqrt(sigma2a)*dW(k-1) + C(k-1)*eta(k-1)*dt + lambda(k-1)*a(k-1)*dt;
            a(k) = a(k-1)+da(k-1);
        end
    end
    if abs(a(k)) >= B(k-1) % sticky boundary at the end
        a(k) = B(k-1)*sign(a(k));
    end
    
end


function a = compute_mean(C, B, lambda, dt)

    a = zeros(1,numel(C) +1);
    da = zeros(1,numel(C));
    a(1) = 0; % a(t=0), initial bias before stimulus
    
    for k = 2:numel(C)+1
        if abs(a(k-1)) >= B(k-1)
            a(k-1:numel(C)+1) = B(k-1)*sign(a(k-1));
            break
        else
            da(k-1) = C(k-1)*dt + lambda(k-1)*a(k-1)*dt;
            a(k) = a(k-1)+da(k-1);
        end
    end
    if abs(a(k)) >= B(k-1)
        a(k) = B(k-1)*sign(a(k));
    end
end


function sigma2dt = compute_variance(C, sigma2a, sigma2i, sigma2s, dt, lambda, temporalDiscretization)
    % compute variance:
    T = numel(C);
    
    var_a = zeros(1,T);
    var_a(1) = sigma2a*dt;
    for k = 2:T
        var_a(k) = var_a(k-1)*(1+lambda(k-1)*dt)^2 + dt*sigma2a;
    end
    % Variance of initial noise accumulating with time
    var_i = zeros(1,T);
    var_i(1) = sigma2i; %at t0
    for k = 2:T+1
        var_i(k) = var_i(k-1)*(1+lambda(k-1)*dt)^2;
    end
    var_i(1) = [];
    % Variance of sensory noise accumulating with time
    var_s = zeros(1,T);
    var_s(1) = C(1)^2*dt*sigma2s;
    for k = 2:T
        var_s(k) = var_s(k-1)*(1+lambda(k-1)*dt)^2 + C(k)^2 * dt^2 * temporalDiscretization*sigma2s;
    end
    
    % Overall variance per timestep:
    sigma2dt = var_a + var_i + var_s; % gaussian adding variance
end