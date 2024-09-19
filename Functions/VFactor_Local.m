function [V2,V3] = VFactor_Local(Data,Wnd,Overlap,position)
    if nargin==3
        position = [1 length(Data)];
    end
    if nargin==2
        Overlap = round(numel(Data)./2);
        position = [1 length(Data)];
    end
    if nargin==1
        Overlap = round(numel(Data)./2);
        Wnd = numel(Data);
        position = [1 length(Data)];
    end
    Data = Data(position(1):position(2));
    M = Data(bsxfun(@plus,(1:Wnd),(0:Overlap:length(Data)-Wnd)'));
    S = median(std(M'));
    
    %Q1 = max((max(M,[],2)-min(M,[],2))./std(M')');% % QFactor on window
    V2 = (max(Data)-min(Data))./S;%Qfactor with windowed std
    V3 = (max(Data)-min(Data))./std(Data);% Qfactor old
    %Vmax = max(abs(Data));
    %Qmin = min(Data);
    %Vrng = max(Data) - min(Data);
    %Qsmax = max(Data)./std(Data);
    %Qsmin = min(Data)./std(Data);
    %Qssum = (max(Data) + min(Data))./std(Data);
end

