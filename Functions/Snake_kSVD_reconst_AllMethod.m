function [Matrix_Coeff,Matrix_CoeffN,Reconstruction,Residual,Error,dError,Max_Coeff,LE] = Snake_kSVD_reconst_AllMethod(Data,Dictionary,NoAtoms,Overlap,sdRemoved,type,th,draw,filename)
% [Matrix_Coeff,Reconstruction,Residual,Error,dError,Max_Coeff] = Snake_kSVD_reconst2(Data,Dictionary,NoAtoms,Overlap,sdRemoved,type,draw,filename)
% types: -'OMP': Orthogonal Matching Pursuit
%        -'WLS': Weighted Least Square
%        -'HUB_rr': Hubber robust regression
%        -'CCH_rr': Chauchy Robust regression
%        -'L1_OMP': calculating the L1 error using OMP Projection
    if nargin==8
        filename ='ASLR Example';
    elseif nargin==7
        filename ='ASLR Example';
        draw = 0;
    elseif nargin==6
        filename ='ASLR Example';
        draw = 0;
        th = 5;
    elseif nargin==5
        filename ='ASLR Example';
        draw = 0;
        th = 5;
        type = 'OMP';
    elseif nargin==4
        filename ='ASLR Example';
        sdRemoved = 4;
        draw = 0;
        th = 5;
        type = 'OMP';
    elseif nargin < 4
        error('Check the inputs!');
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
    err_seg = zeros(1,size(segments,2));
    % Recontruct the Segments using OMP/weighted least square/robust
    % regression
    for no=1:size(segments,2)
        switch type
            case 'OMP'
                [rec{no},coeff{no},~,~,E(:,no)] = OMP_Visualize(Dictionary,segments(:,no),NoAtoms,0,0,1);
                Matrix_Coeff = cell2mat(coeff);
                Matrix_CoeffN = cell2mat(coeff)./norm(segments(:,no));    
            case 'OMP_Smooth'
                len = length(segments(:,no));
                ws = tukeywin(len,0.1);
                segments_s = segments_s(:,no).*ws';
                [rec{no},coeff{no},~,~,E(:,no)] = OMP_Visualize(Dictionary,segments_s,NoAtoms,0,0,1);
                Matrix_Coeff = cell2mat(coeff);
                Matrix_CoeffN = cell2mat(coeff)./norm(segments(:,no));   
            case 'WLS'
                [rec{no},res{no},coeff{no},~] = wls_reg(segments(:,no),Dictionary,NoAtoms,10);
                for k=1:NoAtoms
                    E(k,no) = sqrt(sum(res{no}(:,k+1).^2))/sqrt(sum(segments(:,no).^2));
                end
                Matrix_Coeff = cell2mat(coeff')';
                Matrix_CoeffN = cell2mat(coeff')'./norm(segments(:,no));
            case 'Hub_rr'
               [rec{no},res{no},coeff{no},~] = rob_reg(segments(:,no),Dictionary,5,.5,50,0.001,'Huber',NoAtoms);
               for k=1:NoAtoms
                    E(k,no) = sqrt(sum(res{no}(:,k+1).^2))/sqrt(sum(segments(:,no).^2));
               end
                Matrix_Coeff = cell2mat(coeff')';
                Matrix_CoeffN = cell2mat(coeff')'./norm(segments(:,no));
            case 'CCH_rr'
               [rec{no},res{no},coeff{no},~] = rob_reg(segments(:,no),Dictionary,5,.5,50,0.001,'Cauchy',NoAtoms);
               for k=1:NoAtoms
                    E(k,no) = sqrt(sum(res{no}(:,k+1).^2))/sqrt(sum(segments(:,no).^2));
               end
               Matrix_Coeff = cell2mat(coeff')';
               Matrix_CoeffN = cell2mat(coeff')'./norm(rec{no}(:,end));
            case 'L1_OMP'
                [rec{no},res{no},coeff{no},~] = L1OMP_reg(segments(:,no),Dictionary,NoAtoms);
               for k=1:NoAtoms
                    E(k,no) = sqrt(sum(res{no}(:,k+1).^2))/sqrt(sum(segments(:,no).^2));
               end
               Matrix_Coeff = cell2mat(coeff')';
               Matrix_CoeffN = cell2mat(coeff')'./norm(segments(:,no));
            otherwise
                error('Undefined reconstruction algorithm');
        end

        coeffM(no) = max(abs(coeff{no}))./norm(segments(:,no));% normalized maximum coefficient
        if NoAtoms ==1
            dError(:,no) = E(:,no);
        else
            dError(:,no) = diff(E(:,no));
        end
        rec1(:,no) = rec{no}(:,end);
        rec1_buff{no} = buffer(rec1(:,no),Overlap,0,'nodelay');
    end
    Max_Coeff = max(coeffM);% Normalized maximum coefficient
    
    % Reconstruction
    for k=1:No_block
        rec_t{k} = [];
        if (k<size(Dictionary,1)/Overlap)
            for k2=1+counter2:k+counter2
                rec_t{k}(:,k2-counter2) =  rec1_buff{k2-counter2}(:,k+1-k2+counter2);
            end
            counter2 = counter2 + 1;
            switch type
                case 'OMP_Smooth'
                    Reconstruction =  [Reconstruction;mean_c(rec_t{k},fliplr(rec1_coeff(1:size(rec_t{k},2))))];
                otherwise
                    Reconstruction =  [Reconstruction;mean(rec_t{k},2)];
            end
        elseif (k>No_block-size(Dictionary,1)/Overlap+1)
            m = mod(No_block+1,k);
            for k2=1+counter3:m+counter3
                rec_t{k}(:,k2-counter3) =  rec1_buff{No_block-size(Dictionary,1)/Overlap+1-k2+1+counter3}(:,k2+1);
            end
            counter3 = counter3 + 1;
            switch type
                case 'OMP_Smooth'
                    Reconstruction =  [Reconstruction;mean_c(rec_t{k},fliplr(rec1_coeff(1:size(rec_t{k},2))))];
                otherwise
                    Reconstruction =  [Reconstruction;mean(rec_t{k},2)];
            end
        else
            for k2=1+counter:No_Seg+counter
                rec_t{k}(:,k2-counter) =  rec1_buff{k2}(:,No_Seg+1-k2+counter);
            end
            switch type
                case 'OMP_Smooth'
                    Reconstruction =  [Reconstruction;mean_c(rec_t{k},(rec1_coeff(1:size(rec_t{k},2))))];
                otherwise
                    Reconstruction =  [Reconstruction;mean(rec_t{k}(:,1+sdRemoved:No_Seg-sdRemoved),2)];
            end
            counter = counter + 1;
        end
    end
    
    
    % Calculate The Residual and Error
    Residual = Data-Reconstruction;
    Error = sqrt(sum(Residual.^2))/sqrt(sum(Data.^2))*100;
    
    % Calculate the side Local Error (LE)
    sz = size(Dictionary,1)./2;
    res2_side = [Residual(1:256-sz); Residual(257+sz:end)];
    seg2_side = [Data(1:256-sz); Data(257+sz:end)];
    LE = zeros(5,1);   
    if max(abs(res2_side)) > th       
        LE(1) = max(abs(res2_side));
        LE(2) = max((res2_side).^2);
        LE(3) = sum(abs(res2_side(abs(res2_side)>th)));
        LE(4) = sum(res2_side(abs(res2_side)>th).^2);
        [LE(5),~] = zerocross_count(abs(res2_side') - th);
    end
    
    
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
            pause;
        end
        
        elseif draw==2
        % Visualization with Video
        fg = figure();
        set(gcf,'color','w');
        writerObj = VideoWriter(filename,'Uncompressed AVI');
        writerObj.FrameRate = 5;
        %writerObj.Quality = 75;
        open(writerObj);
        set(fg, 'Units', 'inches', 'Position', [1 1 5 4]);
        set(gcf,'Renderer','zbuffer'); 


        txt = 'ASLR Layer-2!';%2nd Level ASLR Reconstrcution of Events!
        text(0.28,0.5,txt,'Color','black','FontSize',14)
        axis off;
        box off;

        for frmV = 1:1*writerObj.FrameRate
            frame = getframe(gcf);
            writeVideo(writerObj,frame);
            pause(1/writerObj.FrameRate);
        end
        clf(fg);
        

        t = tiledlayout(5,4);
        t.TileSpacing = 'compact';
        t.Padding = 'compact';
        YL_org = 1.2.*max(abs(Data));
        for j=1:size(rec1,2)
            nexttile(1,[4 4])
            h0 = plot(Data,'b','DisplayName','Raw Event');
            hold on;
            h1 = plot(zeros(length(Data),1),'--k');
            h2 = plot(1+Overlap*(j-1):loc_shft+Overlap*(j-1),segments(:,j),'g','DisplayName','Raw Segment');
            h3 = plot(1+Overlap*(j-1):loc_shft+Overlap*(j-1),rec1(:,j),'r','Linewidth',1.5,'DisplayName','Reconstrcuted Segment');
            hold off;
            xlim([1 length(Data)]);
            ylim([-YL_org YL_org]);
            ylabel('Amplitude (uV)');
            xticks([1 128 256 384 512]);
            xticklabels([0 64 128 192 256]);
            xlabel('Time (ms)');
            set(gca, 'Color','w', 'XColor',[0.5 0.5 0.5], 'YColor',[0.5 0.5 0.5]);
            %axis off;
            box off;
            XL = get(gca, 'XLim');
            YL = get(gca, 'YLim');
            rectangle('Position',[XL(1) YL(1) XL(2)-XL(1) YL(2)-YL(1)],'EdgeColor',[0.5 0.5 0.5],'LineWidth',0.5);
            
            XLNew = XL(1) + (j-1)*Overlap; 
            rectangle('Position',[XLNew YL(1) length(segments(:,j)) YL(2)-YL(1)],'FaceColor',[0.9290 0.6940 0.1250,.1],'LineStyle','none');
            
            legend([h0 h2 h3],'Location','northeast','NumColumns',1,'Fontsize',8);%,'Location','best'
            legend boxoff 
            title('ASLR representation');
            tmpcoeff = find(coeff{1, j});
            for j2 = 1:NoAtoms
                nexttile(16+j2)
                plot(Dictionary(:,tmpcoeff(j2)),'k');axis tight;
                txt2 = sprintf('c: %.2f',coeff{1, j}(tmpcoeff(j2)));
                title(txt2);
                box off;axis off;
            end

            % pause for one frame

            pause(1/writerObj.FrameRate);
            frame = getframe(gcf);
            writeVideo(writerObj,frame);
        end
        
        cla(fg);
        t = tiledlayout(3,1);
        t.TileSpacing = 'compact';
        t.Padding = 'compact';
        %1-: raw event
        nexttile;
        plot(Data,'k');axis off;box off;title('Residual Layer-1');%'Residual Layer-1'
        YL_fix = 1.1.*max(abs(Data));
        axis tight;
        ylim([-YL_fix YL_fix]);
        %2- reconstructed
        nexttile;
        plot(Reconstruction,'k');axis off;box off;title('Reconstructed Layer-2');%'Reconstructed Layer-2'
        axis tight;
        ylim([-YL_fix YL_fix]);
        %3- residual
        nexttile;
        plot(Residual,'k');axis off;box off;title('Residual Layer-2');%'Residual Layer-2'
        axis tight;
        ylim([-YL_fix YL_fix]);
        for j3=1:3
            frame = getframe(gcf);
            writeVideo(writerObj,frame);
            pause(1/writerObj.FrameRate);
        end
        close(writerObj);
    else
        % Do nothing
    end
    
    
    
end

