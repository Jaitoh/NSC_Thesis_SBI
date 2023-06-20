function [y] = getFeatures(trial,sIdx,mpR)
% VS, 9/6/2023

if ~isnan(mpR)
    trial.cR = mpR;
end

y = [];

y3s = [];
y4s = [];
figure;
for idxMS = 1:3
    
    used_var = trial.Properties.VariableNames';

    % 删选 idxMS, subject, 
    data = trial(trial.subjectId == sIdx,:);
    data = data(data.idxMS == idxMS,:);
    
    MSs = unique(data.MS);
    
    % 过滤第一个 0 的输入
    data.pulses = data.pulses(:,2:end);
    data.dur = data.dur-1;
    
    % 特征 1，2
    [perf, y1,y2] = get_pRByMDMS(data);
    y = [y y1' y2];
%     plot_y12(perf, y1, y2, MSs, idxMS, data)
    
    % 特征 3
    [perf, y3] = get_pRByNetEvMS(data);
    y = [y y3];
    y3s = [y3s y3'];

%     display_name = sprintf('MS [%s]', string(MSs(idxMS)));
%     plot(y3, '.-', 'DisplayName', display_name)
%     hold on
    
    % 特征 4
    [perf, y4] = get_pR_noSwitch(data);
    y = [y y4'];
    plot_y4(perf, y4, MSs, idxMS, data)

%     display_name = sprintf('MS [%s]', string(MSs(idxMS)));
%     plot(y4, '.-', 'DisplayName', display_name)
%     hold on
%     grid('on')
    
    % 特征 5
    [psyKernel, y5] = getPsyKernelsFeature(data);
    y = [y y5'];
    plot_y5(psyKernel, y5, MSs, idxMS, data)
end

grid("on")
xlabel("MD")
ylabel("pR")

% xticks(1:numel(y3))
% xticklabels(-7:7)
legend()
set(gcf,'color','w');
fig_name = sprintf('feature y3 - SID%d', sIdx);
fil_name = sprintf('feature y3 - SID%d.png', sIdx);

title(fig_name)
saveas(gcf, fil_name)

figure;
h = heatmap(y3s);
netMD = -7:7;

xlabel("MS")
ylabel("netMD")
h.YDisplayLabels = cellstr(string(netMD'));
h.XDisplayLabels = cellstr(string(MSs(1:3)));


fig_name = sprintf('perf y3 pR - SID%d', sIdx);
title(fig_name)
fig_name = sprintf('perf y3 pR - SID%d.png', sIdx);
saveas(gcf, fig_name)

y(isnan(y)) = [];
end

function [perf, y1,y2] = get_pRByMDMS(trial)

    rangeNetMD = -5:5;
    uniqueDur = unique(trial.dur);
    perf = NaN(numel(rangeNetMD),numel(uniqueDur));
    
    cR = trial.cR;
    
    for iMD = 1:numel(rangeNetMD)
        MD = rangeNetMD(iMD);
        
        for iDur = 1:numel(uniqueDur)
            % TODO trial.idxMS > 0  多余吗 （目前都是 > 0）
            % 选择对应的 dur, idxMS, MD， 计算平均 cR， 统计 cR 即 做出右边的选择 的概率
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

function [perf, y3] = get_pRByNetEvMS(trial)

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

function [perf, y4] = get_pR_noSwitch(trial)
    
rangeNetMD = -4:4; rangeNetMD(rangeNetMD == 0) = [];
uniqueDur = unique(trial.dur);
perf = NaN(numel(rangeNetMD),max(trial.idxMS),numel(uniqueDur));

for iMD = 1:numel(rangeNetMD)
    MD = rangeNetMD(iMD);

    cR = trial.cR;
    
    for iMS = 1:max(trial.idxMS)
        for iDur = 1:numel(uniqueDur)
            % iMS = 2 时， iMS=1 -> NaN, 可以替代
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

function [psyKernel, y5] = getPsyKernelsFeature(trial)

    uniqueDur = unique(trial.dur);
    maxDur = max(uniqueDur);
    nDur = numel(uniqueDur);
    
    data1 = trial;
    
    pRpulseR  = NaN(nDur,maxDur);
    pRpulseL  = NaN(nDur,maxDur);
    
    for iDur = 1:nDur
        jDur = data1.dur == uniqueDur(iDur); % idx of dur=2
        for iPulse = 1:uniqueDur(iDur)
            jR = data1.pulses(:,iPulse) > 0; % right movements of pulse
            jL = data1.pulses(:,iPulse) < 0; % left  movements of pulse
            pRpulseR(iDur,iPulse) = nanmean(data1.cR(jDur&jR));
            pRpulseL(iDur,iPulse) = nanmean(data1.cR(jDur&jL));
        end
    end
    
    psyKernel = pRpulseR - pRpulseL;
    
    y5 = psyKernel(:);
end

function plot_y12(perf, y1, y2, MSs, idxMS, data)
    figure;
    h = heatmap(perf);
    % Set x-axis labels
    uniqueDur = unique(data.dur);
    h.XDisplayLabels = cellstr(string(uniqueDur'));

    % Set y-axis labels
    uniqueMD = -5:5;
    h.YDisplayLabels = cellstr(string(uniqueMD'));

    % Set x and y-axis label
    xlabel('Dur');
    ylabel('MD');

    set(gcf,'color','w');
    fig_name = sprintf('perf y1 & y2 - MS[%s]', string(MSs(idxMS)));
    title(fig_name)
    fig_name = sprintf('perf y1 & y2 - MS[%s].png', string(MSs(idxMS)));
    saveas(gcf, fig_name)

    figure;
    plot(y1, '.-')
    hold on
    plot(y2, '.-')
    legend('y1', 'y2')
    grid("on")
    set(gcf,'color','w');
    fig_name = sprintf('feature y1 & y2 - MS[%s]', string(MSs(idxMS)));
    title(fig_name)
    fig_name = sprintf('feature y1 & y2 - MS[%s].png', string(MSs(idxMS)));
    saveas(gcf, fig_name)
end

function plot_y4(perf, y4, MSs, idxMS, data)
    figure;
    h = heatmap(squeeze(perf(:,end,:)));
    % Set x-axis labels
    uniqueDur = unique(data.dur);
    h.XDisplayLabels = cellstr(string(uniqueDur'));

    % Set y-axis labels
    uniqueMD = -4:4; uniqueMD(uniqueMD == 0) = [];
    h.YDisplayLabels = cellstr(string(uniqueMD'));

    % Set x and y-axis label
    xlabel('Dur');
    ylabel('MD');

    set(gcf,'color','w');
    fig_name = sprintf('perf y4- MS[%s]', string(MSs(idxMS)));
    title(fig_name)
    fig_name = sprintf('perf y4 - MS[%s].png', string(MSs(idxMS)));
    saveas(gcf, fig_name)

    figure;
    plot(y4, '.-')
    grid("on")
    set(gcf,'color','w');
    fig_name = sprintf('feature y4 - MS[%s]', string(MSs(idxMS)));
    title(fig_name)
    fig_name = sprintf('feature y4 - MS[%s].png', string(MSs(idxMS)));
    saveas(gcf, fig_name)
end

function plot_y5(perf, y5, MSs, idxMS, data)
    figure;
    h = heatmap(squeeze(perf));
    % Set y-axis labels
    uniqueDur = unique(data.dur);
    h.YDisplayLabels = cellstr(string(uniqueDur'));

    % Set y-axis labels
    pulse=1:(max(uniqueDur));
    h.XDisplayLabels = cellstr(string(pulse'));

    % Set x and y-axis label
    xlabel('Pulse Position');
    ylabel('Dur');

    set(gcf,'color','w');
    fig_name = sprintf('perf y5- MS[%s]', string(MSs(idxMS)));
    title(fig_name)
    fig_name = sprintf('perf y5 - MS[%s].png', string(MSs(idxMS)));
    saveas(gcf, fig_name)

    figure;
    plot(y5, '.-')
    grid("on")
    set(gcf,'color','w');
    fig_name = sprintf('feature y5 - MS[%s]', string(MSs(idxMS)));
    title(fig_name)
    fig_name = sprintf('feature y5 - MS[%s].png', string(MSs(idxMS)));
    saveas(gcf, fig_name)
end
