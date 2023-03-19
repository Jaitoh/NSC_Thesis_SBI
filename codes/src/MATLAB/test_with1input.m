% code test 
% using case participant S1

clear;
close all;

load('../../../data/params/263 models fitPars/data_fitPars_S1.mat')

% C1 = [0, -0.2, -0.2, 0.2, -0.2, 0.2, 0.2, 0, -0.2, 0.2, -0.2, 0, -0.2, 0, -0.2];
C1 = [0, -0.4, -0.4, -0.4, 0, 0.4, 0.4, 0.4, 0, -0.4, 0 ];


ModelName = 'L0';

% bias = -0.01;
bias = -0.72675;
% sigmas = [0.1, 0, 0.2];
% x = [ nan, nan, nan, nan, nan, nan, nan, nan;
%       nan, nan, nan, nan, nan, nan, nan, nan;
%       0.2, nan, nan, nan, nan, nan, nan, nan;
%       nan, nan, nan, nan, nan, nan, nan, nan;
%       nan, nan, nan, nan, nan, nan, nan, nan;
%       nan, nan, nan, nan, nan, nan, nan, nan];
sigmas = [28.77, 0, 5.0986];
x = [ nan, nan, nan, nan, nan, nan, nan, nan;
      nan, nan, nan, nan, nan, nan, nan, nan;
      6, nan, nan, nan, nan, nan, nan, nan;
      nan, nan, nan, nan, nan, nan, nan, nan;
      nan, nan, nan, nan, nan, nan, nan, nan;
      nan, nan, nan, nan, nan, nan, nan, nan];
display(x);

dt = 11/85;
maxClength = 15;
key_lambdaPolynomial = 1;
temporalDiscretization = 100;
numBruns = 100;

debug = true;
[aToPlot1,probRchoice1] = stochastic15models_BGLS(x,dt,C1,maxClength,key_lambdaPolynomial,sigmas,temporalDiscretization,numBruns,bias, debug);

l1 = sprintf('a1 probR: %.3f', probRchoice1);
plot(aToPlot1,'.-', 'DisplayName', l1, "LineWidth",2, "MarkerSize", 3, "Marker", "*", "Color", "#0072BD");
grid on;
hold off;
titleName = sprintf('Model: %s', ModelName);
title(titleName, 'FontSize',24)
xlabel('Time (sample)')
ylabel('a')
lgd = legend('Location', 'southeast');
lgd.FontSize = 32;
lgd.FontWeight = "bold";
set(lgd, 'Color', 'none');


% figureName = sprintf("../../../figures/model/comparison/MATLAB/%s.png", string(ModelName(i)));
% saveas(gcf, figureName);

display(probRchoice1)
