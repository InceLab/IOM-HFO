function [y,coeff,loc,residual,Error]=OMP_Visualize(D,X,L,S1,S2,S3) 
%=============================================
% Sparse representation (OMP) of a group of signals based on a given dictionary and specified number of atoms to use. 
% Input arguments: 
%   D   - The dictionary (its columns MUST be normalized).
%   X   - The signals to represent.
%   L   - The max number of coefficients for each signal.
%   S1  - The minimum error to stop OMP.
%   S2  - The minimum error difference to stop OMP.
%   S3  - Minimum iteration.
% Output arguments: 
%   y        - Approximated signals.
%   coeff    - Sparse coefficient matrix.
%   loc      - Indices of selected dictionary atoms.
%   residual - Residual errors.
%   Error    - Error values.
%=============================================


[~, P]=size(X);
[~, K]=size(D);

for k=1:1:P
    a=[];
    x=X(:,k);
    residual(:,1)=x;

    indx = [];
    for j=1:1:L % iteration
        proj=D'*residual(:,j);
        [~, pos]=max(abs(proj));
        pos=pos(1);
        indx = [indx pos];
        a=pinv(D(:,indx(1:j)))*x;
        residual(:,j+1)=x-D(:,indx(1:j))*a;
        y(:,j) = D(:,indx(1:j))*a;
        
        % Calculate Error
        Error(j) = sqrt(sum(residual(:,j+1).^2))/sqrt(sum(X.^2));
        temp=zeros(K,1);
        temp(indx(1:j))=a;
        coeff(:,k) = temp;
        loc = indx;
        
        if j>1
            Error_Diff(j-1) = Error(j-1) - Error(j) ;
        end

        % Error Breakpoint
        if j>S3
            if (Error(j) < S1) | (Error_Diff(j-1)<S2)
                break;
            end
        end

        %plot_OMP(X,y,D,Error,residual,coeff,loc,j,anim)
    end
end
return;