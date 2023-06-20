clear all
close all

load('trial_for_JakobMacke.mat')

sIdx = 1; % which participant
mpR = []; % vector of model's probabilities of rightward choice for all sequences in trial

[y] = getFeatures(trial,sIdx,mpR);

% plot(y, '.-', LineWidth=2, MarkerSize=20)
% 
% xs = [69, 69, 15, 12, 56, 69, 69, 15, 12, 56, 69, 69, 15, 12, 56];
% a = 0;
% for x = xs
%     a = a+x;
%     xline(a, '--')
% end
% 
% xline(663/3*1)
% xline(663/3*2)
% xline(663/3*3)
% 
% grid()

% to get nSwitches (need it to be 0 for noSwitch trials, and say 1 for the
% rest)
nSwitches = ones(1,size(trial.pulses,1));
for i = 1:size(trial.pulses,1)
   temp = trial.pulses(i,:)/trial.MS(i);
   temp(isnan(temp)) = [];
   temp(temp == 0) = []; % exclude the pauses
   if abs(sum(temp)) == numel(temp)
       nSwitches(i) = 0; % no-switch trial
   end
end