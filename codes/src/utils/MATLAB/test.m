% code test 
% using case participant S1

clear;
close all;

load('../../../data/params/263 models fitPars/data_fitPars_S1.mat')

load('../../../data/trials.mat')
pulses = cell2mat(data(1));
subjectIDs = cell2mat(data(45));
inputs = pulses(subjectIDs==1, :);
% % inputs = inputs(1:100, :);
% inputs = inputs(:, :);
input_size = size(inputs);
input_len = input_size(1);

num_models = length(dataFitPars.allModelsList);

probRs = -1*ones(num_models, input_len);

time = zeros(10,1);
parfor idx = 1:num_models
    tic;

    x = cell2mat(dataFitPars.paramsBGLS(idx));
    bias = dataFitPars.bias(idx);
    sigmas = dataFitPars.sigmas(idx,:);
    dataFitPars.allModelsList(idx);
    
    dt = 11/85;
    maxClength = 15;
    key_lambdaPolynomial = 1;
    temporalDiscretization = 100;
    numBruns = 100;
    
    debug = true;

    probR_temp = -1*ones(1,input_len);
    for j = 1:input_len
        data_in = inputs(j,:);
        nan_idx = isnan(data_in);
        C = data_in(~nan_idx);
        [aToPlot1,probRchoice1] = stochastic15models_BGLS(x,dt,C,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,numBruns,bias, debug);
        probR_temp(1, j) = probRchoice1;
    end
    probRs(idx, :) = probR_temp;
    time(idx) = toc;
    idx
end
total_time = sum(time);


save('../../../data/compare_python_matlab/probRs.mat', 'probRs')
