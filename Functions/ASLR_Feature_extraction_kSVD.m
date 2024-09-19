function [Features] = ASLR_Feature_extraction_kSVD(Train_Data,Dictionary,fs,config)
%  [Features] = ASLR_Feature_extraction_kSVD(pool,Dictionary,config)

if nargin==3
    config.NoA = [6,3,3];
    config.shifts = [4,4,4];
    config.sdRemoved = [6,8,4];
    config.methodology = {'OMP','OMP','OMP'};
elseif nargin<3
    error('Please check the input function');
end

%% Find the threshold 
Ripple_band = [80 270];
FastRipple_band = [230 600];
HFO_band = [80 600];

b1 = fir1(64,Ripple_band/(fs/2)); %64-order FIR filter between 80-250 Hz
a1 = 1;
b2 = fir1(64,FastRipple_band/(fs/2)); %64-order FIR filter between 200-600 Hz
a2 = 1;
b3 = fir1(64,HFO_band/(fs/2)); %64-order FIR filter between 80-600 Hz
a3 = 1;
          
% Find the envelope in each band
N = size(Train_Data,1);
env_p = 3;
for k=1:size(Train_Data,2)
    BndR(:,k) = filtfilt(b1,a1,Train_Data(:,k));
    BndFR(:,k) = filtfilt(b2,a2,Train_Data(:,k));    
    envSideR = calEnvelope([BndR(1:round(N*1/3),k);BndR(round(2*N/3):N,k)],fs); % envelope of R events at the sides
    envSideFR = calEnvelope([BndFR(1:round(N*1/3),k);BndFR(round(2*N/3):N,k)],fs); % envelope of FR events at the sides
    env_thR(k) = max(5,env_p.*median(envSideR));
    env_thFR(k) = max(4,env_p.*median(envSideFR));
end

%% Preprocessing the events
temp = Train_Data;
alpha = 0.3;
Train_Data = detrend(Train_Data,'linear');
w = tukeywin(512,alpha);
Train_Data = Train_Data.*w;

%% Extract the general feature
for i=1:size(Train_Data,2)
    Features0(i,1) = max(temp(:,i)) - min(temp(:,i));%1- rng
end

%% Extract First step Dictionary
Residual1 = zeros(N,size(Train_Data,2));
Reconstruction1 = zeros(N,size(Train_Data,2));

for i=1:size(Train_Data,2)
    [Matrix_Coeff1{i},Reconstruction1(:,i),Residual1(:,i),~,dError1{i},~] = Snake_kSVD_reconst_general(Train_Data(:,i),Dictionary{1},config.NoA(1),config.shifts(1),config.sdRemoved(1),0);
end
Residual1 = Residual1.*w;
for i=1:size(Train_Data,2)
    [Features1(i,:),~] = Level_based_Feature_extraction_kSVD(Train_Data(:,i),Matrix_Coeff1{i},Residual1(:,i),dError1{i});
end
Features1 = Features1(:,[1,5,7,8]);
%% Extract Second step Dictionary
Residual2 = zeros(512,size(Residual1,2));
Reconstruction2 = zeros(512,size(Residual1,2));
for i=1:size(Train_Data,2)
    [Matrix_Coeff2{i},Matrix_Coeff2N{i},Reconstruction2(:,i),Residual2(:,i),~,dError2{i},...
        CoeffM_2(i),LErr_2(i,:)] = Snake_kSVD_reconst_AllMethod(Residual1(:,i),Dictionary{2},config.NoA(2),config.shifts(2),config.sdRemoved(2),config.methodology{2},env_thR(i),0);
end
Residual2 = Residual2.*w;
for i=1:size(Train_Data,2)
    [Features2(i,:),~] = Level_based_Feature_extraction_kSVD(Residual1(:,i),Matrix_Coeff2{i},...
        Residual2(:,i),dError2{i},32);
end
% adding LE to feature space
max(max(abs(Matrix_Coeff2N{i}(1:end-1,:))))
Features2 = Features2(:,[1,4,9,10]);

%% Extract Third step Dictionary
Dictionary{3}(:,end) = [];
Residual3 = zeros(512,size(Residual2,2));
Reconstruction3 = zeros(512,size(Residual2,2));

for i=1:size(Train_Data,2)
    [Matrix_Coeff3{i},Matrix_Coeff3N{i},Reconstruction3(:,i),Residual3(:,i),~,dError3{i},...
        CoeffM_3(i),LErr_3(i,:)] = Snake_kSVD_reconst_AllMethod(Residual2(:,i),[Dictionary{3}, Dictionary{4}],config.NoA(3),config.shifts(3),config.sdRemoved(3),config.methodology{3},env_thFR(i),0);
end

Residual3 = Residual3.*w;
for i=1:size(Train_Data,2)
    [Features3(i,:),~] = Level_based_Feature_extraction_kSVD(Residual2(:,i),Matrix_Coeff3{i},...
        Residual3(:,i),dError3{i},16);
end
% adding LE to feature space
Features3 = Features3(:,[1,4]);

%% Extra feature-1: adding the Centeral Error (CE)
for i=1:size(Train_Data,2)
    CE(i) = max(abs(Residual3(N/2-32:N/2+31,i)));
end

%% Features
Features = horzcat(Features0, Features1, Features2, Features3, CE');
end