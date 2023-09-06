clear all;
close all;
load('save.mat');
order = 1;
for i = 2:51
    
    ind = find(mask(:,i-1)>0);
    amount = mask(ind, i-1);
    for j = 1:amount
        switch ind
            case 1
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[1 0 0],'linewidth',1.0);hold on;
            case 2
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[0 1 0],'linewidth',1.0);hold on;
            case 3
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[0 0 1],'linewidth',1.0);hold on;
            case 4
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[1 1 0],'linewidth',1.0);hold on;
            case 5
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[1 0 1],'linewidth',1.0);hold on;
            case 6
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[0 1 1],'linewidth',1.0);hold on;
            case 7
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[.5 0.5 0.5],'linewidth',1.0);hold on;
            case 8
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[0 0.5 .5],'linewidth',1.0);hold on;
            case 9
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[.5 .8 0],'linewidth',1.0);hold on;
            otherwise
                scatter(Q(i,1),Q(i,2),j*200,'MarkerEdgecolor',[1 .5 0],'linewidth',1.0);hold on;
        end
    end
    if size(find(arrayY==i-2))>0   
        
        %type = find(mask(:,i-1)==1);
        txt = num2str(find(arrayY==i-2));
        %txt = num2str(i);
        text(Q(i,1),Q(i,2),txt,'fontsize',25)
        order = order + 1;
    end
end

figure;
plot(bestrms);

