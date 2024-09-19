function [pool] = HFO_Initial_Detector_DemoVersion(file,config,montage,converterConfig_file)
% [pool] = HFO_Initial_Detector(filename,montage,dataType,config)
% This Function is creating initial pool of HFOs using Amplitude threshold
% detction

if nargin ==3
    converterConfig_file = [];
elseif nargin==2
    converterConfig_file = [];
    montage = [];
elseif nargin<2
    error('Please use proper inputs!');
end

% check the derivationType and dataType>>Extract data
if strcmp(config.derivationType,'monopolar') & strcmp(config.DataType,'edf')
    % removed for Demo
    error('Only bipolar and .mat data can be used for demo code');
elseif strcmp(config.derivationType,'monopolar') & strcmp(config.DataType,'edf_MN_OR')
    % removed for Demo 
    error('Only bipolar and .mat data can be used for demo code');
elseif config.fileName
    if strcmp(config.derivationType,'monopolar') & strcmp(config.DataType,'mat')
        % removed for Demo    
    elseif strcmp(config.derivationType,'bipolar') & strcmp(config.DataType,'mat')
        load(file);
    end
else
    error('Please use bipolar and .mat data for demo');
end

% check the signal range
switch config.signalUnit
    case 'uV'
        % Do nothing
    case 'mV'
        data.data = data.data.*1000; % change to uV
end
config.signalUnit = 'uV';

% Change the sample Rate
switch config.resampleState
    case 'Yes'
        error('Not usable for demo code');
%         fs = config.resampleRate;
%         data.data = resample(data.data,fs,round(montage.SampleRate));
    case 'No'
        fs = montage.SampleRate;
end

% Filtering
[b0,a0]=butter(2,config.lowerBand/(fs/2),'high'); % 2nd order butterworth high pass filter

b1 = fir1(64,config.RippleBand/(fs/2)); %64-order FIR filter between 80-250 Hz
a1 = 1;

b2 = fir1(64,config.FastRippleBand/(fs/2)); %64-order FIR filter between 200-600 Hz
a2 = 1;

b3 = fir1(64,config.HFOBand/(fs/2)); %64-order FIR filter between 80-600 Hz
a3 = 1;
           
fprintf('Filtering iEEG!\r');
input_raw = filtfilt(b0,a0,data.data); %raw iEEG (DC offset is removed!)
% input_raw = filtfilt(b4,a4,input_raw); % remove 60 Hz line noise ## remove this
input_filtered_R = filtfilt(b1,a1,input_raw); %R data
input_filtered_FR = filtfilt(b2,a2,input_raw); %FR data
input_filtered_HFO = filtfilt(b3,a3,input_raw); %HFO data

fprintf('Adaptive thresholding!\r');
% compute channel threshold (in Ripple band)
n = size(input_raw,2);
for i = 1:n
    [~,th_R(:,i)] = get_adaptive_threshold(input_filtered_R(:,i),config.frameLength,config.overlapLength,config.numFrames,config.numOverlapFrames,'Std',5,0);%find the adaptive threshold for each channel
    try
        timestamp_R{i}(:,1) = find_adaptive_event(input_filtered_R(:,i),th_R(:,i),2,1,9);%find the timestamp of each event
        timestamp_R{i}(:,2) = i;
    catch
        timestamp_R{i} = [];
        continue;
    end
end
T = cat(1,timestamp_R{:});
[~,I] = sort(T(:,1));
event.timestamp_R = T(I,:); %timestamp of R events
[alligned,allignedIndex,K] = getaligneddata(input_raw,event.timestamp_R(:,1),[-round(config.eventLength/2-1) round(config.eventLength/2)]);%find initial events 
event.timestamp_R=event.timestamp_R(logical(K),:);
ttlN = size(alligned,3);
for i = 1:ttlN
    event.dataR(:,1,i) = alligned(:,event.timestamp_R(i,2),i);%raw segment
    event.dataR(:,2,i) = input_filtered_HFO(allignedIndex(i,:),event.timestamp_R(i,2));%filtered segment
    event.dataR(:,3,i) = allignedIndex(i,:);%index
end
% compute channel threshold (in Fast Ripple band)
for i = 1:n
    [~,th_FR(:,i)] = get_adaptive_threshold(input_filtered_FR(:,i),config.frameLength,config.overlapLength,config.numFrames,config.numOverlapFrames,'Std',5,0);%find the adaptive threshold for each channel
    try
        timestamp_FR{i}(:,1) = find_adaptive_event(input_filtered_FR(:,i),th_FR(:,i),2,1,9);%find the timestamp of each event
        timestamp_FR{i}(:,2) = i;
    catch
        timestamp_FR{i} = [];
        continue;
    end
end
T = cat(1,timestamp_FR{:});
[~,I] = sort(T(:,1));
event.timestamp_FR = T(I,:); %timestamp of FR events
[alligned,allignedIndex,K] = getaligneddata(input_raw,event.timestamp_FR(:,1),[-round(config.eventLength/2-1) round(config.eventLength/2)]);%find initial events 
event.timestamp_FR=event.timestamp_FR(logical(K),:);
ttlN = size(alligned,3);
for i = 1:ttlN
    event.dataFR(:,1,i) = alligned(:,event.timestamp_FR(i,2),i);%raw segment
    event.dataFR(:,2,i) = input_filtered_HFO(allignedIndex(i,:),event.timestamp_FR(i,2));%filtered segment
    event.dataFR(:,3,i) = allignedIndex(i,:);%index
end

% Merge Rs and FRs
for ch=1:n
    try
        dist{ch} = abs(timestamp_R{ch}(:,1)-timestamp_FR{ch}(:,1).');
        timestamp_All{ch} = [];
        [i, j] = find(dist{ch} < round(config.eventLength/2-1));
        pairs{ch} = [i, j]; %i:Ripple events j:FR events (Keep FR)
        timestamp_R{ch}(i,:)=[]; % remove similar events
    catch
        dist{ch} = [];
    end
    timestamp_All{ch} = [timestamp_R{ch};timestamp_FR{ch}];
end
T = cat(1,timestamp_All{:});
[~,I] = sort(T(:,1));
event.timestamp_All = T(I,:); %timestamp of all events
[alligned,allignedIndex,K] = getaligneddata(input_raw,event.timestamp_All(:,1),[-round(config.eventLength/2-1) round(config.eventLength/2)]);%find initial events 
event.timestamp_All=event.timestamp_All(logical(K),:);
ttlN = size(alligned,3);

for i = 1:ttlN
    event.dataAll(:,1,i) = alligned(:,event.timestamp_All(i,2),i); %raw segment
    event.dataAll(:,2,i) = input_filtered_HFO(allignedIndex(i,:),event.timestamp_All(i,2)); %filtered segment HFO Range
    event.dataAll(:,3,i) = input_filtered_R(allignedIndex(i,:),event.timestamp_All(i,2)); %filtered segment R Range
    event.dataAll(:,4,i) = input_filtered_FR(allignedIndex(i,:),event.timestamp_All(i,2)); %filtered segment FR Range
    event.dataAll(:,5,i) = allignedIndex(i,:); %indecis of event
end


fprintf('removing large artifacts!\r');
% remove large artifacts
accepted_FR = zeros(1,size(event.dataAll,3));
for ev_no=1:size(event.dataAll,3)
    maxFR = max(event.dataAll(:,4,ev_no));
    minFR = min(event.dataAll(:,4,ev_no));  
    if (maxFR<config.FRThreshold) & (minFR>-config.FRThreshold) % FR-th = 100 uV
        accepted_FR(ev_no) = 1;
    else
        accepted_FR(ev_no) = 0;
    end
end
event.dataAll(:,:,accepted_FR==0) = [];
event.timestamp_All(accepted_FR==0,:) = [];

% compute envelope thresholds
env_thR = zeros(size(event.dataAll,3),1);
env_thFR = zeros(size(event.dataAll,3),1);  
for ev_no=1:size(event.dataAll,3)
    tempR = squeeze(event.dataAll(:,3,ev_no));
    tempFR = squeeze(event.dataAll(:,4,ev_no));
    envR_side = calEnvelope([tempR(1:round(config.eventLength*1/3));tempR(round(2*config.eventLength/3):end)],fs);
    envFR_side = calEnvelope([tempFR(1:round(config.eventLength*1/3));tempFR(round(2*config.eventLength/3):end)],fs); 
    env_thR(ev_no) = max(config.minThresholdRipple,config.thresholdMultiplier.*median(envR_side));
    env_thFR(ev_no) = max(config.minThresholdFastRipple,config.thresholdMultiplier.*median(envFR_side)); 
end
env_thR = env_thR;
env_thFR = env_thFR;

% Check for number of crossing
try
    switch config.removeSideHFO
        case 'No'
            for ev_no=1:size(event.dataAll,3)
                p_R(ev_no) = hfo_amp_detector(squeeze(event.dataAll(:,3,ev_no)),[],env_thR,fs,config.cutoffRipple,config.numCrossing);
                p_FR(ev_no) = hfo_amp_detector(squeeze(event.dataAll(:,4,ev_no)),[],env_thFR,fs,config.cutoffFastRipple,config.numCrossing);
                if p_FR(ev_no) | p_R(ev_no)
                    accepted_osc(ev_no)=1;
                else
                    accepted_osc(ev_no)=0;
                end
            end
            event.dataAll(:,:,accepted_osc==0) = [];
            event.timestamp_All(accepted_osc==0,:) = [];
            env_thR(accepted_osc==0,:) = [];
            env_thFR(accepted_osc==0,:) = [];  
        case 'Yes'
            for ev_no=1:size(event.dataAll,3)
                p_R(ev_no) = hfo_amp_detector(squeeze(event.dataAll(:,3,ev_no)),[],env_thR(ev_no),fs,config.cutoffRipple,config.numCrossing);
                p_R_L(ev_no) = hfo_amp_detector(squeeze(event.dataAll(1:round(config.eventLength/3),3,ev_no)),[],env_thR(ev_no),fs,config.cutoffRipple,config.numSideCrossing);
                p_R_R(ev_no) = hfo_amp_detector(squeeze(event.dataAll(round(2*config.eventLength/3):config.eventLength,3,ev_no)),[],env_thR(ev_no),fs,config.cutoffRipple,config.numSideCrossing);
                p_FR(ev_no) = hfo_amp_detector(squeeze(event.dataAll(:,4,ev_no)),[],env_thFR(ev_no),fs,config.cutoffFastRipple,config.numCrossing);
                p_FR_L(ev_no) = hfo_amp_detector(squeeze(event.dataAll(1:round(config.eventLength/3),4,ev_no)),[],env_thFR(ev_no),fs,config.cutoffFastRipple,config.numSideCrossing);
                p_FR_R(ev_no) = hfo_amp_detector(squeeze(event.dataAll(round(2*config.eventLength/3):config.eventLength,4,ev_no)),[],env_thFR(ev_no),fs,config.cutoffFastRipple,config.numSideCrossing);
                if (p_FR(ev_no)==1) | (p_R(ev_no)==1)
                    if p_FR_L(ev_no)+p_FR_R(ev_no)+p_R_L(ev_no)+p_R_R(ev_no)==0% nothing in the side
                        accepted_osc(ev_no)=1;
                    else
                        accepted_osc(ev_no)=0;
                    end
                else
                    accepted_osc(ev_no)=0;
                end
            end
            event.dataAll(:,:,accepted_osc==0) = [];
            event.timestamp_All(accepted_osc==0,:) = [];
            env_thR(accepted_osc==0,:) = [];
            env_thFR(accepted_osc==0,:) = [];
            
            
            % check the HFO band is centralized
            for ev_no=1:size(event.dataAll,3)
                [accepted_env(ev_no),~] = Check_centralized_component(event.dataAll(:,2,ev_no));
            end
            event.dataAll(:,:,accepted_env==0) = [];
            event.timestamp_All(accepted_env==0,:) = [];
            env_thR(accepted_env==0,:) = [];
            env_thFR(accepted_env==0,:) = [];
            fprintf('Returning pool of events\r');
            
            % ######################return the pool########################
            % #############################################################
            %1:
            pool.event.raw = squeeze(event.dataAll(:,1,:));
            pool.event.hfo = squeeze(event.dataAll(:,2,:));
            pool.event.R = squeeze(event.dataAll(:,3,:));
            pool.event.FR = squeeze(event.dataAll(:,4,:));
            %2:
            pool.timestamp = event.timestamp_All(:,1); 
            pool.channelinformation = event.timestamp_All(:,2);
            %3:
            pool.info.envelopeSide.R = env_thR; % Ripple band
            pool.info.envelopeSide.FR = env_thFR; % Ripple band
            pool.info.fs = fs; % sampling Frequency
            pool.info.file_name = file; % data file
            pool.info.signal_range = config.signalUnit; % range of signals
            pool.info.Number_of_channels = n; % Number of channels
            pool.info.montage = montage;
            %4:
            pool.parameters.EventSize = config.eventLength; % size of events
            pool.parameters.th_type = 'Adaptive'; % type of stage 0 thresholding
            pool.parameters.Ripple_band = config.RippleBand; % parameters
            pool.parameters.FastRipple_band = config.FastRippleBand;
            pool.parameters.HFO_band = config.HFOBand;
            pool.parameters.Low_band = config.lowerBand;
            pool.parameters.FR_reject = config.FRThreshold;
            pool.parameters.Number_of_Consecutive_peak_reject = config.numCrossing;
            pool.parameters.Number_of_Consecutive_peak_reject_side = config.numSideCrossing;
            pool.parameters.accepted_osc_event_th = config.thresholdMultiplier;% we are using envelope of side
            pool.parameters.min_thR = config.minThresholdRipple; % min Ripple threshold (uV)
            pool.parameters.min_thFR = config.minThresholdFastRipple; % min FastRipple threshold (uV)
            pool.parameters.timeWindow = [-round(config.eventLength/2-1) round(config.eventLength/2)];
    end
catch
    warning('no event detected');
    pool = [];
    % Nothing!
end




end

