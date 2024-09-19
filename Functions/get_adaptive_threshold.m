function [v,th]=get_adaptive_threshold(data,frame,overlap,h_frame,h_overlap,type,param,input)
%       data: single channel raw data.
%      frame: window length for rectangular window, or pre-defined window.
%    overlap: overlaping samples when calculating the variance.
%    h_frame: Number of adaptive frames when recalculating threshold
%  h_overlap: Number of adaptive overlap frames when recalculating threshold
%       type: choose which type of operator to use.
%      param: parameter used for the threshold.

    if nargin<6
        input=[];
    elseif ~strcmp(type,'Manual') 
        input=[];
    elseif input==0
        input=[];
    end
    if strcmp(type,'Std')==1
        [v,~] = buffered_stats(data,frame,overlap,'std');
    else
        [v,~] = buffered_stats(data,frame,overlap,'var');
    end

    [h_th,bs] = buffered_stats(v,h_frame,h_overlap,'median');
    N = length(data);
    Nv = length(v);

    % check the last buffer
    dt = Nv - (bs-1)*(h_frame - h_overlap);

    if dt<0.75*h_frame %if logical amount of data exist in the last buffered data
        h_th(end) = h_th(end-1);
    end

    Y_samples = linspace(1,length(h_th),N);
    th = interp1(h_th, Y_samples,'linear').*param;
    
end