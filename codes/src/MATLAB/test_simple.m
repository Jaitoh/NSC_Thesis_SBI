% code test 
% using case participant S1

clear;
close all;

load('../../../data/params/263 models fitPars/data_fitPars_S1.mat')

C1 = [0, 0.4, -0.4];
C2 = [0, 0.1, -0.1];
C3 = [0, -0.2, -0.2, 0.2, -0.2, 0.2, 0.2, 0, -0.2, 0.2, -0.2, 0, -0.2, 0, -0.2];
C4 = [0, 0.1, 0, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0, -0.1, 0.1, 0, 0, 0];

ModelName = {'G2', 'G6', 'L2', 'L6', 'S0x0x0', 'S1x1x2', 'G3L2', 'B0', 'B1G1L1S1x1'};

i = 0;
for idx = [133 137 138 142 143 161 132 212 241]
% for idx = 138
    i = i+1;

    x = cell2mat(dataFitPars.paramsBGLS(idx));
    bias = dataFitPars.bias(idx);
    sigmas = dataFitPars.sigmas(idx,:);
    display(cell2mat(dataFitPars.allModelsList(idx)));
    
    dt = 11/85;
    maxClength = 15;
    key_lambdaPolynomial = 1;
    temporalDiscretization = 100;
    numBruns = 100;
    
    debug = true;
    [aToPlot1,probRchoice1] = stochastic15models_BGLS(x,dt,C1,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,numBruns,bias, debug);
    [aToPlot2,probRchoice2] = stochastic15models_BGLS(x,dt,C2,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,numBruns,bias, debug);
    [aToPlot3,probRchoice3] = stochastic15models_BGLS(x,dt,C3,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,numBruns,bias, debug);
    [aToPlot4,probRchoice4] = stochastic15models_BGLS(x,dt,C4,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,numBruns,bias, debug);
    
    l1 = sprintf('a1 probR: %.3f', probRchoice1);
    plot(aToPlot1,'.-', 'DisplayName', l1, "LineWidth",2, "MarkerSize", 3, "Marker", "*", "Color", "#0072BD");
    hold on;
    l2 = sprintf('a2 probR: %.3f', probRchoice2);
    plot(aToPlot2,'.-', 'DisplayName', l2, "LineWidth",2, "MarkerSize", 3, "Marker", "*", "Color", "#A2142F");
    l3 = sprintf('a3 probR: %.3f', probRchoice3);
    plot(aToPlot3,'.-', 'DisplayName', l3, "LineWidth",2, "MarkerSize", 3, "Marker", "*", "Color", "#D95319");
    l4 = sprintf('a4 probR: %.3f', probRchoice4);
    plot(aToPlot4,'.-', 'DisplayName', l4, "LineWidth",2, "MarkerSize", 3, "Marker", "*", "Color", "#7E2F8E");
    grid on;
    hold off;
    titleName = sprintf('Model: %s', string(ModelName(i)));
    title(titleName, 'FontSize',24)
    xlabel('Time (sample)')
    ylabel('Motion Strength')
    lgd = legend('Location', 'southeast');
    lgd.FontSize = 32;
    lgd.FontWeight = "bold";
    set(lgd, 'Color', 'none');
    

    figureName = sprintf("../../../figures/model/comparison/MATLAB/%s.png", string(ModelName(i)));
    saveas(gcf, figureName);

    display([probRchoice1, probRchoice2, probRchoice3, probRchoice4])
end