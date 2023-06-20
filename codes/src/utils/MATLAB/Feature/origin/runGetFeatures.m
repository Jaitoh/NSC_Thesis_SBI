clear all

load('trial_for_JakobMacke.mat')

sIdx = 15; % which participant
mpR = []; % vector of model's probabilities of rightward choice for all sequences in trial

[y] = getFeatures(trial,sIdx,mpR);