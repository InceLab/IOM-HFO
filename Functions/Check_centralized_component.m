function [accept,th] = Check_centralized_component(data,center_range,side_range,wnd,overlap,param)
% [accept] = Check_centralized_component(data,center_range,side_range,wnd,overlap,param)
%
% input:   data: filtered event (in HFO range)
%          center_range: range of event at the center (default: remove 1/10 samples at the left and right sides)
%          side_range: range of event at the side(default: 1/3 at the left side and 1/3 at the right side of the event)
%          wnd: Window size std
%          overlap: overlap size of std
%          param: threshold's parameter (th = th*param)
%
% output:  accept: check if the event accepted as hfo
%          threshold: threshold of the event

    if nargin<2
        center_range = 1/10;
        side_range = 1/3;
        wnd = 8;
        overlap = 4;
        param = 3;
    end

    sd_center = temp_variance(data(round(length(data)*center_range):round(length(data)*(1-center_range))),wnd,overlap,2);
    sd_side = [temp_variance(data(1:round(length(data)*side_range)),wnd,overlap,2); temp_variance(data(round(2*length(data)*side_range):length(data)),wnd,overlap,2)];
    th = param.*median(sd_side);
    FF_mid = find(sd_center>th, 1);
    if isempty(FF_mid)
        accept = 0;
    else
        accept = 1;
    end   
end

