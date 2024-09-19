%--------------------------------------------------------------------------
% Title: Multi-Layer feature extraction using sparse representation
% Author: [Behrang Fazli Besheli]
% Email: FazliBesheli.Behrang@mayo.edu
% Institution: [Mayo Clinic, Department of Neurosurgery]
% Date: [09/07/2024]
%--------------------------------------------------------------------------
% Description:
% This script performs feature extraction based on the quality of representation 
% using a multi-layered sparse representation approach. In this demo, features 
% are extracted as described in the README file.
% 
% Workflow:
% 1. Features are extracted from each detected HFO event based on the quality 
%    of its representation using the dictionary.
% 
% 2. A total of 12 features are derived, capturing various aspects of the 
%    representation such as quality of global and local approximation error.
% 
% Outputs:
% - Features: Extracted feature matrix for all initially detected candidate events.
%--------------------------------------------------------------------------

%% Initialization and Setup
clc; clear; close all;
addpath(fullfile(pwd, 'Functions'));

% Load training data and dictionary
load(fullfile(pwd, 'Data\Labeled_Event.mat'));% Events including real and pseudo-HFOs
load(fullfile(pwd, 'Data\Dictionary_CRDL_ASLR.mat'));

%% kSVD Parameters
config.NoA = [6, 3, 3];         % Number of atoms per layer
config.shifts = [4, 4, 4];      % Shifts per layer
config.sdRemoved = [6, 8, 4];   % Shifts to remove per layer
config.methodology = {'OMP', 'OMP', 'OMP'};

% Feature extraction
Features = ASLR_Feature_extraction_kSVD(Train_Data, Dictionary.L2kSVD, SampleRate(1), config);
Features = [Features, SampleRate', Train_Class', Train_Subject'];

%% Save Features
parameters = {config.NoA, config.shifts, config.sdRemoved, config.methodology};
extractedFeatures = {'Range', 'L2Err_1', 'EVS_1', 'Max(C)_1',...
    'Max(D)_1', 'L2Err_2', 'V2_2', 'Max(C)_2',...
    'Max(D)_2', 'L2Err_3', 'V2_3', 'CE'};

save('ExtractedFeatures_CRDL_ASLR.mat', 'Features', 'parameters', 'extractedFeatures');
