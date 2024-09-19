function [Features,info] = Level_based_Feature_extraction_kSVD(Data,Matrix_Coeff,Residual,dError,dx)
    if nargin==3
        dError = [];
        dx = [];
    elseif nargin==4 
        dx = [];
        dEV = sum(abs(dError),1);
        p = dEV / sum(dEV);
        Ent = -sum(p.*log2(p));
        Ent(isnan(Ent))=0;
    else 
        dEV = sum(abs(dError),1);
        p = dEV / sum(dEV);
        Ent = -sum(p.*log2(p));
        Ent(isnan(Ent))=0;
    end
    %1: first feature (Approximation Error)
    Er = sqrt(sum(Residual.^2))/sqrt(sum(Data.^2))*100;
    L1Er = sum(abs(Residual))/sum(abs(Data))*100;
    %2: Second faeture (QFactor)
    [V2,V3] = VFactor_Local(Residual,128,4);
    
    %3: Local Error
    if ~isempty(dx)
        sp = length(Data);% center of data and dx: length of lI
        LE = sqrt(sum(Residual(sp/2-dx:sp/2+dx).^2))/sqrt(sum(Data(sp/2-dx:sp/2+dx).^2))*100;
        SK = skewness(Residual(sp/2-dx:sp/2+dx));
    end
    
    %4: Line noise (Repetitive Patterns)
    temp_cov = cov(sign(Matrix_Coeff(1:end-1,:)));
    sptl_cov = cov(sign(Matrix_Coeff(1:end-1,:))');
    [~,Sptl_D] = eig(sptl_cov);
    [~,temp_D] = eig(temp_cov);
    EVS = max(diag(Sptl_D));
    EVT = max(diag(temp_D));
    
    
    if ~isempty(dx)
        Features = [Er;L1Er;LE;V2;V3;abs(SK);EVS;EVT;max(max(abs(Matrix_Coeff(1:end-1,:))))./norm(Data);max(max(abs(dError)));Ent];
        info = {'Approximation Error','L1Approximation Error','VFactor1','VFactor2','VFactor3','DeltaQ','maximum Spatial Eigen Value',...
            'maximum temporal Eigen Value','max max','max mean'};
    else
        Features = [Er;L1Er;V2;V3;EVS;EVT;max(max(abs(Matrix_Coeff(1:end-1,:))))./norm(Data);max(max(abs(dError)));Ent];
        info = {'Approximation Error','L1Approximation Error','VFactor1','VFactor2','VFactor3','DeltaQ','maximum Spatial Eigen Value',...
            'maximum temporal Eigen Value','max max','max mean','max change in Error','Entropy of max change in Error'};    
    end
    
end

