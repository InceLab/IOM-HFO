%--------------------------------------------------------------------------
% Title: Random Forest Model Creation using leave-one-subject-out cross
% validation
% Author: [Behrang Fazli Besheli]
% Email: FazliBesheli.Behrang@mayo.edu
% Institution: [Mayo Clinic, Department of Neurosurgery]
% Date: [09/07/2024]
%--------------------------------------------------------------------------
% Description:
% This script creates Random Forest models using features extracted from 
% previously identified HFO events. The process involves applying 
% leave-one-subject-out cross-validation (LOSO-CV), where models are learned 
% for each subject (model{i} corresponds to the model for the ith subject).
% 
% Workflow:
% 1. For each subject, a Random Forest model is trained on the data from 
%    all other subjects, leaving the ith subject out for validation.
% 
% 2. Prediction results (pred) classify events as either pseudo-HFOs 
%    (pred == 1) or real-HFOs (pred == 2).
% 
% 3. Subject-specific confusion matrices (conMat_sub) are generated for each 
%    model, and a final confusion matrix (confMat_tot) summarizes the results 
%    across all subjects.
% 
% Outputs:
% - model{i}: The learned Random Forest model for the ith subject.
% - pred: Prediction labels (1 = pseudo-HFO, 2 = real-HFO).
% - conMat_sub: Confusion matrix for each subject.
% - confMat_tot: Final confusion matrix across all subjects.
% 
% Requirements:
% - To run this script, you need to download and compile the Random Forest 
%   toolbox. For more information, please refer to the README file.
%--------------------------------------------------------------------------

%% Initialization and Setup
clc; clear; close all;
addpath(fullfile(pwd, 'Functions'));

%load(fullfile(pwd, 'Data\ExtractedFeature_CRDL_ASLR.mat'));% Load features
load(fullfile(pwd, 'ExtractedFeatures_CRDL_ASLR.mat'));% Load features
Train_Class = Features(:,14);
Train_Subject = Features(:,15);

selectedFeatures = 1:12;
Features = Features(:,selectedFeatures)';

%% RF Classification Parameters
% Parameters
minparent = 2; % min parent size
minleaf = 1; % min leaf size
NoTree = 100; % number of trees
Novar = ceil(sqrt(size(Features,1))); % number of variables
confMat_tot = zeros(2,2); % overall confussion matrix
NoSubject = max(Train_Subject); % number of subjects and RF models

config; % RF config

%% Leave-One-Subject-Out Process
for i = 1:NoSubject
    fprintf('Sub:%.f Processing...\n', i);
    test_data = Features(:,Train_Subject==i); 
    test_class = Train_Class(Train_Subject==i);

    if ~isempty(test_class)
        train_data = Features(:,Train_Subject~=i); 
        train_class = Train_Class(Train_Subject~=i);

        % Create RF model
        model{i} = train_RF(train_data', train_class, 'minparent', minparent, 'minleaf', minleaf, 'ntrees', NoTree, ...
                         'oobe', 'y', 'method', 'c', 'nvartosample', Novar);

        % Prediction
        [pred, ~] = eval_RF(test_data', model{i}, 'oobe', 'y');
        accuracy = cal_accuracy(test_class, pred);

        % Misclassified events
        misclassified.all{i} = find(test_class ~= pred);
        misclassified.rHFO{i} = find(test_class == 1 & pred == 2);
        misclassified.pHFO{i} = find(test_class == 2 & pred == 1);

        % Selected Features in each node
        for w = 1:NoTree
            predictorindex{i, w} = model{i}(w).nodeCutVar;
            cutthreshold{i, w} = model{i}(w).nodeCutValue;
        end
        confMat_sub{i} = confusionmat(test_class, pred);
        confMat_tot = confMat_tot + confMat_sub{i};
    end
end

% save the results
save('RF_Model_LOSOCV.mat','confMat_sub','confMat_tot','model');

