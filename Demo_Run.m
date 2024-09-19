%--------------------------------------------------------------------------
% Title: Initial Dual-band HFO detection with pseudo-HFO elimination 
% Author: [Behrang Fazli Besheli]
% Email: FazliBesheli.Behrang@mayo.edu
% Institution: [Mayo Clinic, Department of Neurosurgery]
% Date: [09/07/2024]
%--------------------------------------------------------------------------
% Description:
% This script demonstrates the initial dual-band High-Frequency Oscillation 
% (HFO) detection in iEEG data, followed by the elimination of pseudo-HFOs 
% using a pre-learned model based on Cascaded Residual Dictionary Learning 
% (CRDL) and Adaptive Sparse Learning Representations (ASLR).
% 
% Workflow:
% 1. Initial detection: An amplitude threshold is applied to detect candidate 
%    HFO events in raw iEEG data, creating an initial pool of events.
% 
% 2. Feature extraction: Features are extracted from the initial event pool 
%    using CRDL and ASLR, leveraging pre-learned dictionaries for representation.
% 
% 3. Classification: A pre-compiled Random Forest model is applied to classify 
%    events as real HFOs (pred == 2) or pseudo-HFOs (pred == 1).
% 
% Outputs:
% - result.pool.event.raw: Initially detected events (512 x num_events matrix)
% - result.pool.event.raw(:,pred==1): All pseudo-HFO events
% - result.pool.event.raw(:,pred==2): All real-HFO events
% 
% Notes:
% - pred == 1 corresponds to pseudo-HFOs.
% - pred == 2 corresponds to real HFOs.
% 
%--------------------------------------------------------------------------

%% Initialization and Setup
clc;clear;close all;

% Add necessary function directory to the path
addpath(fullfile(pwd, 'Functions'));

% Load demo iEEG data
fileName = fullfile(pwd, 'Data\Demo_rawiEEG.mat');

% Load dictionary
load(fullfile(pwd, 'Data\Dictionary_CRDL_ASLR.mat'));

% Load random forest model
load(fullfile(pwd, 'Data\RF_CRDL_ASLR_Model.mat'));%RF_CRDL_ASLR_Model.mat'));

%% configuration
config;

% Main parameter structure for the detection configuration
param.detection.DataType = 'mat';            % Data type ('mat', 'edf', etc.)
param.detection.blockRange = [];             % Range of blocks to process (empty means all)
param.detection.saveMat = 'Yes';             % Option to save the results in a .mat file (For .edf to .mat conversion)
param.detection.derivationType = 'bipolar';  % Derivation type (e.g., 'bipolar', 'monopolar')
param.detection.signalUnit = 'uV';           % Unit of signal (microvolts)
param.detection.resampleState = 'No';        % Whether to resample the data
param.detection.resampleRate = 2000;         % Resampling rate in Hz (in case of data resampling)
param.detection.saveResults = true;          % Boolean to save results (1 = true, 0 = false)

% High-frequency oscillation (HFO) detection-related parameters
param.detection.removeSideHFO = 'Yes';       % Option to events with side crossing
param.detection.lowerBand = 1;               % Lower bound for frequency band
param.detection.HFOBand = [80 600];          % Frequency band for HFOs (Hz)
param.detection.RippleBand = [80 270];       % Ripple band (Hz)
param.detection.FastRippleBand = [230 600];  % Fast ripple band (Hz)

% Frame and overlap settings for signal segmentation
param.detection.frameLength = 128;           % Length of each frame in samples
param.detection.overlapLength = 0;           % Overlap between frames
param.detection.numFrames = round(60 / (128 / param.detection.resampleRate)); % Total number of frames (calculated for fs=2000)
param.detection.numOverlapFrames = round(30 / (128 / param.detection.resampleRate)); % Total overlap frames (calculated for fs=2000)

% STFT and filter-related parameters
param.detection.eventLength = 512;      % event length
param.detection.FRThreshold = 100;           % Fast Ripple rejection threshold (100 uV)
param.detection.minThresholdRipple = 5;      % Minimum threshold for ripple detection (5 uV)
param.detection.minThresholdFastRipple = 4;  % Minimum threshold for fast ripple detection (4 uV)
param.detection.thresholdMultiplier = 3;      % Threshold Multiplier

% Channel-specific parameters
param.detection.numCrossing = 6;             % Number of HFO crossing
param.detection.numSideCrossing = 4;         % Number of HFO side crossing to reject
param.detection.cutoffRipple = 80;           % Ripple filter cutoff frequency (Hz)
param.detection.cutoffFastRipple = 250;      % Fast ripple filter cutoff frequency (Hz)

% File name
param.detection.fileName = fileName;         % Name of the input file

param.denoising.NoA = [6,4,3];               % Number of atoms in each layer
param.denoising.shifts = [4,4,4];            % Amount of shoft in aech layer
param.denoising.sdRemoved = [6,8,4];         % Name of the input file
param.denoising.methodology = {'OMP','OMP','OMP'}; % Reconstrcution method

%% iEEG Analysis
[result.pool] = HFO_Initial_Detector_DemoVersion(fileName,param.detection); % Initial HFO detector
result.pool.event.raw = result.pool.event.raw(:,1:100);
try
    % Feature Extraction (CRDL+ASLR)
    [result.Features] = ASLR_Feature_extraction_kSVD(result.pool.event.raw,Dictionary.L2kSVD,2000,param.denoising); % Feature Extraction
    [result.pred, result.votes]= eval_RF(result.Features, model{1}, 'oobe', 'y'); % pseudo-HFO eleimination using RF classifier pred=1: pseudo-HFO , pred=2: real-HFO
catch
    % Nothing
end
