function [y] = getFeatures(trial,sIdx,mpR)
% VS, 9/6/2023

if ~isnan(mpR)
    trial.cR = mpR;
end

y = [];
for idxMS = 1:3
    data = trial(trial.subjectId == sIdx,:);
    data = data(data.idxMS == idxMS,:);
    
    data.pulses = data.pulses(:,2:end);
    data.dur = data.dur-1;
    
    
    [y1,y2] = get_pRByMDMS(data);
    y = [y y1' y2];
    [y3] = get_pRByNetEvMS(data);
    y = [y y3];
    
    [y4] = get_pR_noSwitch(data);
    y = [y y4'];
    
    [y5] = getPsyKernelsFeature(data);
    y = [y y5'];
end

y(isnan(y)) = [];
end

function [y1,y2] = get_pRByMDMS(trial)

rangeNetMD = -5:5;
uniqueDur = unique(trial.dur);
perf = NaN(numel(rangeNetMD),numel(uniqueDur));

cR = trial.cR;

for iMD = 1:numel(rangeNetMD)
    MD = rangeNetMD(iMD);
    
    for iDur = 1:numel(uniqueDur)
        x = trial.dur == uniqueDur(iDur) & trial.idxMS > 0 & trial.MD == MD;
        perf(iMD,iDur) = mean(cR(x));
    end
end

y1 = [];
for iDur = numel(uniqueDur):-1:1
    x = ~isnan(perf(:,iDur));
    y0 = perf(x,iDur);
    y1 = [y1; y0];
end

y2 = [];
for iMD = 1:numel(rangeNetMD)
    x = ~isnan(perf(iMD,:));
    y0 = perf(iMD,x);
    y2 = [y2, y0];
end
end

function [y3] = get_pRByNetEvMS(trial)

rangeNetMD = -7:7;
uniqueMS = unique(trial.idxMS);
uniqueMS(uniqueMS == 0) = [];
perf = NaN(numel(rangeNetMD),numel(uniqueMS));

cR = trial.cR;

for iMD = 1:numel(rangeNetMD)
    MD = rangeNetMD(iMD);
    
    for iMS = 1:numel(uniqueMS)
        x = trial.idxMS == uniqueMS(iMS) & trial.MD == MD;
        perf(iMD,iMS) = mean(cR(x));        
    end
end

y3 = [];
for iMD = 1:numel(rangeNetMD)
    x = ~isnan(perf(iMD,:));
    y0 = perf(iMD,x);
    y3 = [y3 y0];
end
end

function [y4] = get_pR_noSwitch(trial)
    
rangeNetMD = -4:4; rangeNetMD(rangeNetMD == 0) = [];
uniqueDur = unique(trial.dur);
perf = NaN(numel(rangeNetMD),max(trial.idxMS),numel(uniqueDur));

for iMD = 1:numel(rangeNetMD)
    MD = rangeNetMD(iMD);

    cR = trial.cR;
    
    for iMS = 1:max(trial.idxMS)
        for iDur = 1:numel(uniqueDur)
            x = trial.dur == uniqueDur(iDur) & trial.idxMS == iMS & ...
                trial.MD == MD & trial.nSwitches == 0;
            perf(iMD,iMS,iDur) = mean(cR(x));
        end
    end
end

y4 = [];
for iDur = 1:2
    for iMS = 1:max(trial.idxMS)
        x = ~isnan(perf(:,iMS,iDur));
        y0 = perf(x,iMS,iDur);
        y4 = [y4; y0];
    end
end
end

function [y5] = getPsyKernelsFeature(trial)

uniqueDur = unique(trial.dur);
maxDur = max(uniqueDur);
nDur = numel(uniqueDur);

data1 = trial;

pRpulseR  = NaN(nDur,maxDur);
pRpulseL  = NaN(nDur,maxDur);

for iDur = 1:nDur
    jDur = data1.dur == uniqueDur(iDur);
    for iPulse = 1:uniqueDur(iDur)
        jR = data1.pulses(:,iPulse) > 0;
        jL = data1.pulses(:,iPulse) < 0;
        pRpulseR(iDur,iPulse) = nanmean(data1.cR(jDur&jR));
        pRpulseL(iDur,iPulse) = nanmean(data1.cR(jDur&jL));
    end
end

psyKernel = pRpulseR - pRpulseL;

y5 = psyKernel(:);
end