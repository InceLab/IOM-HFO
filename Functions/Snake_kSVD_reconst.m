function [Matrix_Coeff,Reconstruction,Residual,Error,dError,Max_Coeff] = Snake_kSVD_reconst(Data,Dictionary,NoAtoms,Overlap,sdRemoved,draw)
    if nargin==5
        draw = 0;
    elseif nargin==4
        draw = 0;
        sdRemoved = 4;
    end
    
    % Initialize Parameters    
    counter = 0;
    counter2 = 0;
    counter3 = 0;
    No_block = length(Data)/(Overlap);
    No_Seg = size(Dictionary,1)/(Overlap);
    Reconstruction = [];
    
    % Buffer Data
    loc_shft = size(Dictionary,1);
    segments = buffer(Data,loc_shft,loc_shft-Overlap,'nodelay');
    
    % Recontruct the Segments using OMP
    for no=1:size(segments,2)
        [rec{no},coeff{no},~,~,E(:,no)] = OMP_Visualize(Dictionary,segments(:,no),NoAtoms,0,0,1);
        coeffM(no) = max(abs(coeff{no}))./norm(segments(:,no));% normalized maximum coefficient
        if NoAtoms ==1
            dError(:,no) = E(:,no);
        else
            dError(:,no) = diff(E(:,no));
        end
        rec1(:,no) = rec{no}(:,end);
        rec1_buff{no} = buffer(rec1(:,no),Overlap,0,'nodelay');
    end
    
    Matrix_Coeff = cell2mat(coeff);
    Max_Coeff = max(coeffM);% Normalized maximum coefficient
    
    
    % Reconstruction
    for k=1:No_block
        rec_t{k} = [];
        if (k<size(Dictionary,1)/Overlap)
            for k2=1+counter2:k+counter2
                rec_t{k}(:,k2-counter2) =  rec1_buff{k2-counter2}(:,k+1-k2+counter2);
            end
            counter2 = counter2 + 1;
            Reconstruction =  [Reconstruction;mean(rec_t{k},2)];
        elseif (k>No_block-size(Dictionary,1)/Overlap+1)
            m = mod(No_block+1,k);
            for k2=1+counter3:m+counter3
                rec_t{k}(:,k2-counter3) =  rec1_buff{No_block-size(Dictionary,1)/Overlap+1-k2+1+counter3}(:,k2+1);
            end
            counter3 = counter3 + 1;
            Reconstruction =  [Reconstruction;mean(rec_t{k},2)];
        else
            for k2=1+counter:No_Seg+counter
                rec_t{k}(:,k2-counter) =  rec1_buff{k2}(:,No_Seg+1-k2+counter);
            end
            Reconstruction =  [Reconstruction;mean(rec_t{k}(:,1+sdRemoved:No_Seg-sdRemoved),2)];
            counter = counter + 1;
        end
    end

    % Calculate The Residual and Error
    Residual = Data-Reconstruction;
    Error = sqrt(sum(Residual.^2))/sqrt(sum(Data.^2))*100;
    
    % plot the snake in case draw=1
    if draw==1
        figure;
        sgtitle('Local kSVD Recontruction')
        subplot(1,2,1);
        plot(Data,'k');
        hold on;
        plot(Reconstruction,'r');
        hold on;
        plot(Residual,'b');
        hold on;
        plot(zeros(length(Data),1),'--k');
        legend('Origial','Reconstructed','Residual');
        xlim([1 length(Data)]);
        txt = sprintf('Appoximation Error:%.3f',Error);
        title(txt);
        
        subplot(1,2,2);
        title('Snake Reconstruction!');
        for j=1:size(rec1,2)
            plot(Data,'b');
            hold on;
            plot(1+Overlap*(j-1):loc_shft+Overlap*(j-1),segments(:,j),'g');
            hold on;
            plot(1+Overlap*(j-1):loc_shft+Overlap*(j-1),rec1(:,j),'r');
            hold off;
            xlim([1 length(Data)]);
            pause();
        end   
    end
    
    
    
end

