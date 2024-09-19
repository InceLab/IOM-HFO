function [m] = mean_c(data,coeff)
    len = size(data,1);
    no = size(data,2);
    m = (data*coeff')./sum(coeff);
end