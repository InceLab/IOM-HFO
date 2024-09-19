%--------------------------------------------------------------------------
% Title: Multi-Layer Dictionary Learning using residual-based cascaded kSVD
% Author: [Behrang Fazli Besheli]
% Email: FazliBesheli.Behrang@mayo.edu
% Institution: [Mayo Clinic, Department of Neurosurgery]
% Date: [09/07/2024]
%--------------------------------------------------------------------------
% Description:
% This script performs multi-layer dictionary learning using a residual-based 
% cascaded approach with k-SVD and ASLR (Adaptive Sparse Local Representation). 
% The dictionary is learned from a pool of HFO events, and in each layer, 
% the residuals from the previous layer are used to create another set of data, 
% from which a new dictionary is learned.
% 
% Workflow:
% 1. Dictionaries are learned iteratively through multiple layers. The residual 
%    from each previous layer serves as the input for the next layer, allowing 
%    the model to capture finer details of the HFO events. D = {D1, D2, D3,
%    D4}
% 
% Parameters:
% - Two sets of parameters are used in this demo:
%    1. ASLR Parameters: These control the adaptive sparse learning process.
%    2. k-SVD Parameters: These control the dictionary learning process. For 
%       more details on k-SVD parameters, please refer to the k-SVD paper 
%       (The K-SVD: An Algorithm for Designing of Overcomplete Dictionaries for 
%       Sparse Representation by M. Aharon, M. Elad, and A.M. Bruckstein).
% 
% Outputs:
% - Learned dictionaries for each layer based on residual data.
% 
% Requirements:
% - You need to download the k-SVD toolbox and compile it to run this script.
% - For more detailed information on ASLR and k-SVD parameters, refer to the 
%   main manuscript and the k-SVD paper, respectively.
% - For additional instructions, check the README file.
%--------------------------------------------------------------------------

%% Initialization and Setup
clc;clear;close all;

% Add necessary function directory to the path
addpath(fullfile(pwd, 'Functions'));

% Load training data
load(fullfile(pwd, 'Data\TrainData_HFO_Event.mat'));% example of HFO events

%% Parameter Initialization
% Data segment length (number of samples per event)
eventSize = 512;

% Sampling frequency in Hz
samplingFreq = 2000;

% Define frequency bands for Ripple and Fast Ripple components
rippleBand = [80 270];          % Ripple band (80-270 Hz)
fastRippleBand = [230 600];     % Fast Ripple band (230-600 Hz)
hfoBand = [80 600];             % HFO band (80-600 Hz)

% Design FIR filters for the defined frequency bands
% Ripple filter (64th-order FIR)
[b_ripple, a_ripple] = fir1(64, rippleBand/(samplingFreq/2));

% Fast Ripple filter (64th-order FIR)
[b_fastRipple, a_fastRipple] = fir1(64, fastRippleBand/(samplingFreq/2));

% HFO filter (64th-order FIR)
[b_hfo, a_hfo] = fir1(64, hfoBand/(samplingFreq/2));

% Define the envelope threshold multiplier
envelopeThresholdMultiplier = 3;

% Define the Tukey window alpha parameter (window tapering)
tukeyAlpha = 0.3;

% Define the number of crossings for event detection
numCrossings = 6;

% Apply Tukey window to the training data
tukeyWindow = tukeywin(eventSize, tukeyAlpha);
windowedTrainingData = Train_Data .* tukeyWindow; % Suppress edges of training data

%% Find the Envelope Thresholds of Ripple and FR bands
BndR = zeros(size(Train_Data));
BndFR = zeros(size(Train_Data));
BndHFO = zeros(size(Train_Data));
env_thR = zeros(1,size(Train_Data,2));
env_thFR = zeros(1,size(Train_Data,2));
env_thHFO = zeros(1,size(Train_Data,2));
for k=1:size(Train_Data,2)
    BndR(:,k) = filtfilt(b_ripple,a_ripple,Train_Data(:,k));
    BndFR(:,k) = filtfilt(b_fastRipple,a_fastRipple,Train_Data(:,k)); 
    BndHFO(:,k) = filtfilt(b_hfo,a_hfo,Train_Data(:,k));

    envSideHFO = calEnvelope([BndHFO(1:round(eventSize*1/3),k);BndHFO(round(2*eventSize/3):eventSize,k)],samplingFreq); % envelope of HFO events at the sides
    envSideR = calEnvelope([BndR(1:round(eventSize*1/3),k);BndR(round(2*eventSize/3):eventSize,k)],samplingFreq); % envelope of R events at the sides
    envSideFR = calEnvelope([BndFR(1:round(eventSize*1/3),k);BndFR(round(2*eventSize/3):eventSize,k)],samplingFreq); % envelope of FR events at the sides

   env_thR(k) = max(5,envelopeThresholdMultiplier.*median(envSideR));
   env_thFR(k) = max(4,envelopeThresholdMultiplier.*median(envSideFR));
   env_thHFO(k) = max(5,envelopeThresholdMultiplier.*median(envSideHFO));
end

%% ASLR & kSVD Parameters
fprintf('#######--- Dictionary Learning ---#######\r');
param_ASLR = struct(...
    'Layer1', struct('maxShift', 0, 'locShift', 128, 'locOverlap', 120, 'numAtoms', 6, 'overlapParam', 4, 'sdRemoved', 6), ...
    'Layer2', struct('maxShift', 0, 'locShift', 128, 'locOverlap', 124, 'sideRemove', 48, 'numAtoms', 4, 'overlapParam', 4, 'sdRemoved', 5), ...
    'Layer3', struct('max_shift', 0, 'locShift', 64, 'locOverlap', 62, 'sideRemove', 16, 'numAtoms', 3, 'overlapParam', 4, 'sdRemoved', 5), ...
    'Layer4', struct('max_shift', 0, 'locShift', 64, 'locOverlap', 63, 'sideRemove', 16, 'numAtoms', 2, 'overlapParam', 4, 'sdRemoved', 6));

param_kSVD = struct(...
    'Layer1', struct('L', 2, 'K', 48, 'numIteration', 10, 'InitializationMethod', 'GivenMatrix', 'initialDictionary', data_norm(randn(param_ASLR.Layer1.locShift, 48), 2), 'displayProgress', 1, 'errorFlag', 0, 'preserveDCAtom', 0), ...
    'Layer2', struct('L', 2, 'K', 32, 'numIteration', 10, 'InitializationMethod', 'GivenMatrix', 'initialDictionary', data_norm(randn(param_ASLR.Layer2.locShift, 32), 2), 'displayProgress', 1, 'errorFlag', 0, 'preserveDCAtom', 0), ...
    'Layer3', struct('L', 2, 'K', 24, 'numIteration', 10, 'InitializationMethod', 'GivenMatrix', 'initialDictionary', data_norm(randn(param_ASLR.Layer3.locShift, 24), 2), 'displayProgress', 1, 'errorFlag', 0, 'preserveDCAtom', 0), ...
    'Layer4', struct('L', 2, 'K', 16, 'numIteration', 10, 'InitializationMethod', 'GivenMatrix', 'initialDictionary', data_norm(randn(param_ASLR.Layer4.locShift, 16), 2), 'displayProgress', 1, 'errorFlag', 0, 'preserveDCAtom', 0));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                ####### First Layer ########
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% D1- Buffering
fprintf('1st Layer buffering...\r');
[sx ,sy] = size(buffer(windowedTrainingData(:,1),param_ASLR.Layer1.locShift ,param_ASLR.Layer1.locOverlap,'nodelay'));
Event_buff1 = zeros(sx,sy*size(windowedTrainingData,2));
c = 0;
for kk=1:size(windowedTrainingData,2)
    Event_buff1(:,c+1:c+sy) = buffer(windowedTrainingData(:,kk),param_ASLR.Layer1.locShift ,param_ASLR.Layer1.locOverlap,'nodelay');
    c = c+sy;
end

%% D1- kSVD
fprintf('D1 kSVD Dictionary Learning...\r');
Event_buff1 = detrend(Event_buff1,'constant');% DC offset should remove!
Event_buff1 = data_norm(Event_buff1,2); % Normalization
[D1,~]  = KSVD(Event_buff1,param_kSVD.Layer1);
D1(:,param_kSVD.Layer1.K+1)=1/sqrt(param_ASLR.Layer1.locShift ).*ones(param_ASLR.Layer1.locShift ,1); % add DC component

%% D1- ASLR and Residual Calculation
fprintf('L1 ASLR Representation...\r');
Residual{1} = zeros(512,size(Train_Data,2));
for k=1:size(windowedTrainingData,2)
    [~,~,Residual{1}(:,k),~,~] = Snake_kSVD_reconst(Train_Data(:,k),D1,param_ASLR.Layer1.numAtoms ,param_ASLR.Layer1.overlapParam ,param_ASLR.Layer1.sdRemoved,0);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                ####### Second Layer ########
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% D2- Buffering
fprintf('2nd Layer buffering...\r');
Train_Data_D2 = Residual{1}.*tukeyWindow;
Train_Data_D2 = generate_shifted_data(Train_Data_D2, -param_ASLR.Layer2.maxShift:1:param_ASLR.Layer2.maxShift);
[sx, sy] = size(buffer(Train_Data_D2(:,1),param_ASLR.Layer2.locShift ,param_ASLR.Layer2.locOverlap ,'nodelay'));
Event_buff2_n = zeros(sx,sy*size(Train_Data_D2,2));
c = 0;
for kk=1:size(Train_Data_D2,2)
    Event_buff2_n(:,c+1:c+sy) = buffer(Train_Data_D2(:,kk),param_ASLR.Layer2.locShift ,param_ASLR.Layer2.locOverlap ,'nodelay');
    c = c+sy;
end
%% D2- HFO Attention
fprintf('D2 HFO Attention\r');
L = size(Event_buff2_n,2)/size(Train_Data,2);
th2 = repelem(env_thR, L);
acp = zeros(1,size(Event_buff2_n,2));
for kk=1:size(Event_buff2_n,2)
    p=hfo_amp_detector(Event_buff2_n(:,kk),th2(kk),5,samplingFreq,80,numCrossings); 
    if p==1
        env = calEnvelope(Event_buff2_n(:,kk),samplingFreq)'; % envelope of events
        loc = find(env==max(env));
        if (loc>param_ASLR.Layer2.sideRemove)&&(loc<param_ASLR.Layer2.locShift -param_ASLR.Layer2.sideRemove) % max of envelope should sit at the center of locals
            acp(kk) = 1;
        end
    end
end
Event_buff2 = Event_buff2_n(:,find(acp));
%% D2- kSVD
fprintf('D2 kSVD Dictionary Learning...\r');
Event_buff2 = detrend(Event_buff2,'constant');% DC offset should remove!
Event_buff2 = data_norm(Event_buff2,2);
[D2,~]  = KSVD(Event_buff2,param_kSVD.Layer2);
D2(:,param_kSVD.Layer2.K+1)=1/sqrt(param_ASLR.Layer2.locShift ).*ones(param_ASLR.Layer2.locShift ,1);
%% D2- ASLR and Residual Calculation
fprintf('L2 ASLR Representation...\r');
Residual{2} = zeros(512,size(Train_Data,2));
for i=1:size(Train_Data,2)
    [~,~,Residual{2}(:,i),~,~] = Snake_kSVD_reconst(Residual{1}(:,i),D2,param_ASLR.Layer2.numAtoms ,param_ASLR.Layer2.overlapParam,param_ASLR.Layer2.sdRemoved,0);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                ####### Third Layer ########
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% D3- Buffering
fprintf('3rd Layer buffering...\r');
Train_Data_D3 = Residual{2}.*tukeyWindow;
Train_Data_D3 = generate_shifted_data(Train_Data_D3, -param_ASLR.Layer3.max_shift:1:param_ASLR.Layer3.max_shift);
[sx, sy] = size(buffer(Train_Data_D3(:,1),param_ASLR.Layer3.locShift ,param_ASLR.Layer3.locOverlap ,'nodelay'));
Event_buff3_n = zeros(sx,sy*size(Train_Data_D3,2));
c = 0;
for kk=1:size(Train_Data_D3,2)
    Event_buff3_n(:,c+1:c+sy) = buffer(Train_Data_D3(:,kk),param_ASLR.Layer3.locShift ,param_ASLR.Layer3.locOverlap ,'nodelay');
    c = c+sy;
end

%% D3- HFO Attention
fprintf('D3 HFO Attention\r');
L = size(Event_buff3_n,2)/size(Train_Data,2);
th3 = repelem(env_thR, L);
acp = zeros(1,size(Event_buff3_n,2));
for kk=1:size(Event_buff3_n,2)
    p=hfo_amp_detector(Event_buff3_n(:,kk),th3(kk),5,samplingFreq,80,numCrossings); 
    if p==1
        env = calEnvelope(Event_buff3_n(:,kk),samplingFreq)'; % envelope of events
        loc = find(env==max(env));
        if (loc>param_ASLR.Layer3.sideRemove)&&(loc<param_ASLR.Layer3.locShift -param_ASLR.Layer3.sideRemove) % max of envelope should sit at the center of locals
            acp(kk) = 1;
        end
    end
end
Event_buff3 = Event_buff3_n(:,find(acp));

%% D3- kSVD
fprintf('D3 kSVD Dictionary Learning...\r');
Event_buff3 = detrend(Event_buff3,'constant');% DC offset should remove!
Event_buff3 = data_norm(Event_buff3,2);
[D3,~]  = KSVD(Event_buff3,param_kSVD.Layer3);
D3(:,param_kSVD.Layer3.K+1)=1/sqrt(param_ASLR.Layer3.locShift ).*ones(param_ASLR.Layer3.locShift ,1);
%% D3- ASLR and Residual Calculation
fprintf('L3 ASLR Representation...\r');
Residual{3} = zeros(eventSize,size(Train_Data,2));
for i=1:size(Train_Data,2)
    fprintf('3rd step local reconstruction Event:%.f\r',i);
    [~,~,Residual{3}(:,i),~,~] = Snake_kSVD_reconst(Residual{2}(:,i),D3,param_ASLR.Layer3.numAtoms,param_ASLR.Layer3.overlapParam,param_ASLR.Layer3.sdRemoved,0);
end
fprintf('3rd step local reconstruction done!');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                ####### Fourth Layer ########
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% D4- Buffering
fprintf('4th Layer buffering...\r');
Train_Data_D4 = Residual{3}.*tukeyWindow;
Train_Data_D4 = generate_shifted_data(Train_Data_D4, -param_ASLR.Layer4.max_shift:1:param_ASLR.Layer4.max_shift);
[sx, sy] = size(buffer(Train_Data_D4(:,1),param_ASLR.Layer4.locShift,param_ASLR.Layer4.locOverlap,'nodelay'));
Event_buff4_n = zeros(sx,sy*size(Train_Data_D4,2));
c = 0;
for kk=1:size(Train_Data_D4,2)
    Event_buff4_n(:,c+1:c+sy) = buffer(Train_Data_D4(:,kk),param_ASLR.Layer4.locShift,param_ASLR.Layer4.locOverlap,'nodelay');
    c = c+sy;
    fprintf('Fourth Layer Buffer Events %.f\r',kk);
end
fprintf('Fourth Layer Buffering is done!\r');
%% D4- HFO Attention
fprintf('D4 HFO Attention\r');
L = size(Event_buff4_n,2)/size(Train_Data,2);
th4 = repelem(env_thR, L);
acp = zeros(1,size(Event_buff4_n,2));
for kk=1:size(Event_buff4_n,2)
    p=hfo_amp_detector(Event_buff4_n(:,kk),th4(kk),5,samplingFreq,250,numCrossings); 
    if p==1
        env = calEnvelope(Event_buff4_n(:,kk),samplingFreq)'; % envelope of events
        loc = find(env==max(env));
        if (loc>param_ASLR.Layer4.sideRemove)&&(loc<param_ASLR.Layer4.locShift-param_ASLR.Layer4.sideRemove) % max of envelope should sit at the center of locals
            acp(kk) = 1;
        end
    end
end
Event_buff4 = Event_buff4_n(:,find(acp));

%% D4- kSVD
fprintf('D4 kSVD Dictionary Learning...\r');
Event_buff4 = detrend(Event_buff4,'constant');% DC offset should remove!
Event_buff4 = data_norm(Event_buff4,2);
[D4,~]  = KSVD(Event_buff4,param_kSVD.Layer4);
D4(:,param_kSVD.Layer4.K+1)=1/sqrt(param_ASLR.Layer4.locShift).*ones(param_ASLR.Layer4.locShift,1);

%% D4- ASLR and Residual Calculation
fprintf('L4 ASLR Representation...\r');
Residual{4} = zeros(eventSize,size(Train_Data,2));
for i=1:size(Train_Data,2)
    fprintf('4nd step local reconstruction Event:%.f\r',i);
    [~,~,Residual{4}(:,i),~,~] = Snake_kSVD_reconst(Residual{3}(:,i),D4,param_ASLR.Layer4.numAtoms ,param_ASLR.Layer4.overlapParam ,param_ASLR.Layer4.sdRemoved,0);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                                ####### Learned Dictionary ########
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('#######--- D = {D1, D2, D3, D4} ---#######\r');
fprintf('##########################################\r');
Dictionary = {D1, D2, D3, D4}; % 4-stage dictionary
% Define settings for kSVD and dictionary parameters
setting.kSVD.param = param_kSVD;
setting.Snake = param_ASLR;
% Save dictionary and settings to file
save('Dictionary_CRDL', 'Dictionary', 'setting', '-v7.3');


