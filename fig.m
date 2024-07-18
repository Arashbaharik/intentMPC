clc;
clear all;
close all;
import casadi.*; 
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0,'defaulttextinterpreter','Latex')
set(gca,'FontSize',20)


figure(1)
load(strcat('real',num2str(10)))
j=K;
xlim([y0(1) y_f(1)+50])
ylim([0 X_T(2)+60])
plot(XX(1,1:j),XX(2,1:j),'b')
xlim([y0(1) y_f(1)+50])
ylim([0 X_T(2)+60])
hold on
plot(YY(1,1:j),YY(2,1:j),'r')
xlim([y0(1) y_f(1)+50])
ylim([0 X_T(2)+60])
load('determ')
j=K;
xlim([y0(1) y_f(1)+50])
ylim([0 X_T(2)+60])
plot(XX(1,1:j),XX(2,1:j),'k')
load('nominal')
j=K;
xlim([y0(1) y_f(1)+50])
ylim([0 X_T(2)+60])
plot(XX(1,1:j),XX(2,1:j),'g')
load(strcat('real',num2str(10)))
marker =imread('airb.png');
marker = imrotate(marker,-XX(3,j)*180/pi,'bicubic','loose');
newmarker = marker;
newmarker(marker == 0) = 255;
markersize = [25,25];
x_low = XX(1,j) - markersize(1)/2; 
x_high = XX(1,j) + markersize(1)/2;
y_low = XX(2,j) - markersize(2)/2;
y_high = XX(2,j) + markersize(2)/2;
hold on
imagesc([x_low x_high], [y_low y_high], newmarker)
xlim([y0(1) y_f(1)+50])
ylim([0 X_T(2)+60])
marker =imread('airr.png');
marker = imrotate(marker,-YY(3,j)*180/pi,'bicubic','loose');
newmarker = marker;
newmarker(marker == 0) = 255;
markersize = [25,25]; 
x_low = YY(1,j) - markersize(1)/2; 
x_high = YY(1,j) + markersize(1)/2;
y_low = YY(2,j) - markersize(2)/2; 
y_high = YY(2,j) + markersize(2)/2;
hold on
imagesc([x_low x_high], [y_low y_high], newmarker)
xlim([y0(1) y_f(1)+50])
ylim([0 X_T(2)+60])
hold on
patch(X_T(1)+10*cos(phi),X_T(2)+10*sin(phi),'g');
hold on
plot(YY1(1,:),YY1(2,:),'--r')
hold on
plot(YY2(1,:),YY2(2,:),'--r')
legend('Scenario-tree MPC','Intruder trajectory','Classic MPC','Nominal trajectory','Ownship target','','','Location','southeast','Interpreter','latex')  
ylabel('$y$[m]')
xlabel('$x$[m]')



figure(2)
load(strcat('real',num2str(10)))
plot(d,'LineWidth',1,'Color',[0 0 1])
hold on
load('determ')
plot(d,'LineWidth',1,'Color',[0 0 0])
hold on
load('nominal')
plot(d,'LineWidth',1,'Color',[0 1 0])
hold on
yline(sqrt(2000),'--r')
ylim([0 220])
xlim([1 82])
legend('Scenario-tree MPC','Classic MPC','Nominal paths distance','Minimum distance','Location','northwest','Interpreter','latex')  
ylabel('$\rho$[m]')
xlabel('$t$[s]')




figure(3)
for mm=[1,2]
    if mm==1
        load('intent.mat')
        j=K;
        plot(XX(1,1:j),XX(2,1:j),'b')
        hold on
        j=7;
        j1=11;
        marker =imread('airb.png');
        marker = imrotate(marker,-XX(3,j)*180/pi,'bicubic','loose');
        newmarker = marker;
        newmarker(marker == 0) = 255;
        markersize = [8,8];
        x_low = XX(1,j) - markersize(1)/2; 
        x_high = XX(1,j) + markersize(1)/2;
        y_low = XX(2,j) - markersize(2)/2;
        y_high = XX(2,j) + markersize(2)/2;
        imagesc([x_low x_high], [y_low y_high], newmarker)
        hold on
    elseif mm==2
        load('nonintent.mat')
        j=K;
        plot(XX(1,1:j),XX(2,1:j),'Color',[0 0.6 0])
        j=11;
        j1=11;
        marker =imread('airg.png');
        marker = imrotate(marker,-XX(3,j)*180/pi,'bicubic','loose');
        newmarker = marker;
        newmarker(marker == 0) = 255;
        markersize = [10,10];
        x_low = XX(1,j) - markersize(1)/2; 
        x_high = XX(1,j) + markersize(1)/2;
        y_low = XX(2,j) - markersize(2)/2;
        y_high = XX(2,j) + markersize(2)/2;
        hold on
        imagesc([x_low x_high], [y_low y_high], newmarker)
        hold on
    end
    figure(3)
    j=K;
    plot(YY(1,1:j),YY(2,1:j),'r')
    hold on
    marker =imread('airr.png');
    marker = imrotate(marker,-YY(3,j1)*180/pi,'bicubic','loose');
    newmarker = marker;
    newmarker(marker == 0) = 255;
    markersize = [10,10]; 
    x_low = YY(1,j1) - markersize(1)/2; 
    x_high = YY(1,j1) + markersize(1)/2;
    y_low = YY(2,j1) - markersize(2)/2; 
    y_high = YY(2,j1) + markersize(2)/2;
    hold on
    imagesc([x_low x_high], [y_low y_high], newmarker)
    xlim([-50 100])
    ylim([0  230])
    xlabel('$x$[m]')
    ylabel('$y$[m]')
    legend('Intent-aware ownship','Intruder','Intent-unaware ownship','Location','southeast','Interpreter','latex')  
    hold on
end



figure(4)
iii=1;
for jj=[1,2]
    if jj==1
        load(strcat('real',num2str(10)))
    else
        load('determ')
    end  
    for j=[35,57]
        figure(4)
        subplot(2,2,iii)
        Y_predk=reshape(Y_pred(j,:,:,:),n_x,N+1,M_r^N_r);
        for ii=1:M_r^N_r
            figure(4)
            subplot(2,2,iii)
            plot(Y_predk(1,:,ii),Y_predk(2,:,ii),'k')
            hold on
        end
        figure(4)
        subplot(2,2,iii)
        plot(reshape(X_pred(j,1,:),1,N+1),reshape(X_pred(j,2,:),1,N+1),'k')
        hold on
        subplot(2,2,iii)
        xlim([y0(1) y_f(1)+50])
        ylim([0 X_T(2)+60])
        plot(XX(1,1:j),XX(2,1:j),'b')
        xlim([y0(1) y_f(1)+50])
        ylim([0 X_T(2)+60])
        hold on
        plot(YY(1,1:j),YY(2,1:j),'r')
        xlim([y0(1) y_f(1)+50])
        ylim([0 X_T(2)+60])
        marker =imread('airb.png');
        marker = imrotate(marker,-XX(3,j)*180/pi,'bicubic','loose');
        newmarker = marker;
        newmarker(marker == 0) = 255;
        markersize = [25,25];
        x_low = XX(1,j) - markersize(1)/2; 
        x_high = XX(1,j) + markersize(1)/2;
        y_low = XX(2,j) - markersize(2)/2;
        y_high = XX(2,j) + markersize(2)/2;
        hold on
        imagesc([x_low x_high], [y_low y_high], newmarker)
        xlim([y0(1) y_f(1)+50])
        ylim([0 X_T(2)+60])
        marker =imread('airr.png');
        marker = imrotate(marker,-YY(3,j)*180/pi,'bicubic','loose');
        newmarker = marker;
        newmarker(marker == 0) = 255;
        markersize = [25,25]; 
        x_low = YY(1,j) - markersize(1)/2; 
        x_high = YY(1,j) + markersize(1)/2;
        y_low = YY(2,j) - markersize(2)/2; 
        y_high = YY(2,j) + markersize(2)/2;
        hold on
        imagesc([x_low x_high], [y_low y_high], newmarker)
        xlim([y0(1) y_f(1)+50])
        ylim([0 X_T(2)+60])
        hold on
        patch(X_T(1)+10*cos(phi),X_T(2)+10*sin(phi),'g');
        hold on
        plot(YY1(1,:),YY1(2,:),'--r')
        hold on
        plot(YY2(1,:),YY2(2,:),'--r')
        hold on
        iii=iii+1;
        xlabel('$x$[m]')
        ylabel('$y$[m]')
    end
end



 L=20;
 for jj=1:L
     if mod(jj,2)==1
         l=(jj+1)/2+10;
     else
         l=jj/2;
     end
    load(strcat('real',num2str(l)))
    figure(5)
    xlim([y0(1) y_f(1)+50])
    ylim([0 X_T(2)+60])
    if l==10
        plot(XX(1,1:j),XX(2,1:j),'LineWidth',2,'Color',[0 0 0])
        xlim([y0(1) y_f(1)+50])
        ylim([0 X_T(2)+60])
        hold on
        plot(YY(1,1:j),YY(2,1:j),'LineWidth',2,'Color',[0 0 0])
        hold on
        plot(YY1(1,:),YY1(2,:),'--k')
        hold on
        plot(YY2(1,:),YY2(2,:),'--k')
        hold on
    else
        plot(XX(1,1:j),XX(2,1:j),'b')
        xlim([y0(1) y_f(1)+50])
        ylim([0 X_T(2)+60])
        hold on
        plot(YY(1,1:j),YY(2,1:j),'r')
    end
    xlim([y0(1) y_f(1)+50])
    ylim([0 X_T(2)+60])
    hold on
    patch(X_T(1)+10*cos(phi),X_T(2)+10*sin(phi),'g');
    hold on;
    h = zeros(3, 1);
    ylabel('$y$[m]')
    xlabel('$x$[m]')
    h(1) = plot(NaN,NaN,'r');
    h(2) = plot(NaN,NaN,'b');
    h(3) = plot(NaN,NaN,'LineWidth',2,'Color',[0 0 0]);
    legend(h, 'Intruder trajectories','Ownship trajectories','Trajectories without disturbances','Location','southeast','Interpreter','latex');
    hold on
    figure(7)
    if l==10
        subplot(2,1,1)
        plot(linearV,'LineWidth',2,'Color',[0 0 0])
        hold on
        yline(9,'--r')
        yline(6,'--r')
        xlim([0 length(linearV)])
        ylim([0 10])
        xlabel('$t$[s]')
        ylabel('$v^1\,$[m/s]')
        subplot(2,1,2)
        plot(angularV,'LineWidth',2,'Color',[0 0 0])
        hold on
        yline(-0.1,'--r')
        yline(0.1,'--r')
        xlim([0 length(angularV)])
        ylim([-0.11 0.11])
        xlabel('$t$[s]')
        ylabel('$u^1\,$[rad/{s}]')
        hold on
    else
        subplot(2,1,1)
        plot(linearV,'LineWidth',1,'Color',[0,0,1])
        hold on
        yline(9,'--r')
        yline(6,'--r')
        xlim([0 length(linearV)])
        ylim([0 10])
        xlabel('$t$[s]')
        ylabel('$v^1\,$[m/s]')
        subplot(2,1,2)
        plot(angularV,'LineWidth',1,'Color',[0,0,1])
        hold on
        yline(-0.1,'--r')
        yline(0.1,'--r')
        xlim([0 length(angularV)])
        ylim([-0.11 0.11])
        xlabel('$t$[s]')
        ylabel('$u^1\,$[rad/{s}]')
        hold on
    end
    if l==10
        figure(6)
        plot(d,'LineWidth',2,'Color',[0 0 0])
        hold on
        yline(sqrt(2000),'--r')
        ylim([0 220])
        xlim([1 length(d)])
        ylabel('$\rho$[m]')
        xlabel('$t$[s]')
        h = zeros(3, 1);
        h(1) = plot(NaN,NaN,'g');
        h(2) = plot(NaN,NaN,'LineWidth',2,'Color',[0 0 0]);
        h(3) = plot(NaN,NaN,'-r');
        legend(h, 'Distance with disturbances','Distance without disturbances','Minimum distance','','','Location','northwest','Interpreter','latex');
        hold on
    else
        figure(6)
        plot(d,'color',[0 0.5 0])
        hold on
        yline(sqrt(2000),'--r')
        ylim([0 220])
        xlim([1 length(d)])
        ylabel('$\rho$[m]')
        xlabel('$t$[s]')
        hold on
    end
 end

set(figure(1),'PaperSize',[14 11])
print(figure(1), '-dpdf', 'f1.pdf','-fillpage','-r300'); 
set(figure(5),'PaperSize',[14 11])
print(figure(5), '-dpdf', 'f5.pdf','-fillpage','-r300'); 
set(figure(2),'PaperSize',[12 8])
print(figure(2), '-dpdf', 'f2.pdf','-fillpage','-r300'); 
set(figure(6),'PaperSize',[12 8])
print(figure(6), '-dpdf', 'f6.pdf','-fillpage','-r300'); 
set(figure(3),'PaperSize',[14 11])
print(figure(3), '-dpdf', 'f3.pdf','-fillpage','-r300'); 
set(figure(4),'PaperSize',[12 12])
print(figure(4), '-dpdf', 'f4.pdf','-fillpage','-r300'); 
set(figure(7),'PaperSize',[12 8])
print(figure(7), '-dpdf', 'f7.pdf','-fillpage','-r300'); 