clear;
close all;

C = [0, -0.2, -0.2, 0.2, -0.2,  0.2,  0.2,    0, -0.2, 0.2, -0.2, 0,   -0.2, 0, -0.2];
x = [[nan, nan, nan, nan, nan, nan, nan, nan],
     [10, nan, nan, nan, nan, nan, nan, nan],
     [1, nan, nan, nan, nan, nan, nan, nan],
     [2, nan, nan, nan, nan, nan, nan, nan],
     [2, 1, nan, nan, nan, nan, nan, nan],
     [1, nan, nan, nan, nan, nan, nan, nan]];

bias = 2;
sigmas = [10, 0, 1];
% dataFitPars.allModelsList(idx);
    
dt = 11/85;
maxClength = 15;
key_lambdaPolynomial = 1;
temporalDiscretization = 100;
numBruns = 100;

debug = false;

for i = 1:100
    [aToPlot1,probRchoice1] = stochastic15models_BGLS(x,dt,C,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,numBruns,bias, debug);
end