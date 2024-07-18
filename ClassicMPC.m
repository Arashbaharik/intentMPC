clc;
clear all;
close all;
import casadi.*;
%the basics are from the scenarioMPC code, some lines are commented for the
%classic MPC

% problem definition
n_x     = 3;            %number of state 
n_u     = 2;            %number of control 
N       = 30;           %horizon
phi     = 0:0.01:2*pi;  %generating circle 
t_e     = 1;            %disceret time
X_T     = [420;560;pi/2];%ownship-target state
v       = 10;           %intruder linear velocity
dphi    = 0.07;         %intruder max angular velocity in rad/s
max_sep = sqrt(2000);   % mimumum allowed distance
x0      = [0;0;0];      %ownship initial state
y0      = [-120;90;0];  %intruder initial state
y_f     = [600;450;0];  %intuder waypoint
% MPC
M_r   = 3;  % number of braches (uncertainties) in the scenario-tree MPC
N_r   = 3;    %robust horizon of the scenario-tree MPC
X     = SX .sym('X',n_x,N+1);  %ownship state
Y     = SX .sym('Y',n_x,N+1);   %intruder state
U     = SX .sym('U',n_u,N);    %ownship input
U1    = SX .sym('U1',n_u,N);  %intruder inputs
X0    = SX .sym('X0',n_x,1);  %initial state of the ownship
Y0    = SX .sym('Y0',n_x,1);   %initial state of the intruder
Yr    = SX .sym('Yr',n_x,N+1,M_r^N_r);   %Sate of different scenarios for the intruder   Yr{i}: n_x*N+1 SX with 1<=i<=M_r^N_r
U1r   = SX .sym('U1r',n_u,N,M_r^N_r); %control of different scenarios for the intruder
Cost  = 0;   %cost 
ineq  = [];  %inequality constraints
eq    = [X(:,1)-X0;Y(:,1)-Y0];  %equality constraints
Yrc   = [];
Urc   = [];
for i = 1:N   
    if i==1
        Cost = Cost+0.1*(X(:,i)-X_T)'*(X(:,i)-X_T);
    elseif i>1
        Cost = Cost+0.1*(X(:,i)-X_T)'*(X(:,i)-X_T)+5500000*(U(2,i)-U(2,i-1))'*(U(2,i)-U(2,i-1));
    end
    ineq = [ineq;-(X(:,i)-Y(:,i))'*diag([1,1,0])*(X(:,i)-Y(:,i))+max_sep^2;U(1,i)-0.9*v;-U(1,i)+0.6*v;U(2,i)-0.1;-U(2,i)-0.1]; 
    eq   = [eq;X(:,i+1)-[X(1,i)+t_e*U(1,i)*cos(X(3,i));X(2,i)+t_e*U(1,i)*sin(X(3,i));X(3,i)+U(2,i)];
            Y(:,i+1)-[Y(1,i)+t_e*U1(1,i)*cos(Y(3,i));Y(2,i)+t_e*U1(1,i)*sin(Y(3,i));Y(3,i)+U1(2,i)]];    
end
ineq   = [ineq;-(X(:,N+1)-Y(:,N+1))'*diag([1,1,0])*(X(:,N+1)-Y(:,N+1))+max_sep^2];
for j  = 1:M_r^N_r %all scenatios
    eq   = [eq;Yr{j}(:,1)-Y0];
    for i=1:N
        ineq   = [ineq;-(X(:,i)-Yr{j}(:,i))'*diag([1,1,0])*(X(:,i)-Yr{j}(:,i))+max_sep^2]; 
        eq     = [eq;Yr{j}(:,i+1)-[Yr{j}(1,i)+t_e*U1r{j}(1,i)*cos(Yr{j}(3,i));Yr{j}(2,i)+t_e*U1r{j}(1,i)*sin(Yr{j}(3,i));Yr{j}(3,i)+U1r{j}(2,i)]];    
    end
    Yrc   = [Yrc;Yr{j}(:)];
    Urc   = [Urc;U1r{j}(:)];
end
Cost   = Cost+1*(X(:,N+1)-X_T)'*(X(:,N+1)-X_T);
% NLP
Pr    = [U(:);X(:);Y(:);Yrc]; %decision variables
nlp   = struct('x',Pr, 'f',Cost, 'g',[ineq;eq],'p',[X0;Y0;U1(:);Urc]);
opts  = struct;
opts.ipopt.mu_target   = 1e-4;
opts.ipopt.mu_init     = 1e-4;
opts.ipopt.print_level = 0;
opts.print_time        = false;
S     = nlpsol('S', 'ipopt', nlp,opts);
% Dubin path LSR
    r       = v/dphi;
    ix      = y_f(1)-y0(1);
    iy      = y_f(2)-y0(2);
    ll      = sqrt(ix^2+(iy-2*r)^2-4*r^2);
    alpha   = asin(2*r/(sqrt((iy-2*r)^2+ix^2)))+atan((iy-2*r)/ix); %angle of L
    k1      = round(alpha/dphi); %time for L
    k2      = k1+round(ll/v);  %time for S
    K       = k2+k1+1; %time til end
    UU1     = [v*ones(1,K);zeros(1,K)];
    for k   = 1:k1  
    UU1(2,k) = dphi;   %angle: L
    end
    UU1(2,k1+1)= 0.24*dphi; %interpolation
    for k    = k1+2:k2 
    UU1(2,k) = 0;   %angle: S
    end
    for k=k2+1:K
    UU1(2,k) = -dphi;  %angle: R
    end
    exUU1=[UU1 [v*ones(1,K);zeros(1,K)]]; %extend of control
    YY1(:,1) = y0;  %circles for Dubin path
    YY2(:,1) = y_f; %circles for Dubin path
% Run MPC
L     = 20; %number of realization
for l  = 10  %L=10 for the nominal realization
    load('initialRMPC.mat') %intial condition
    XX(:,1) = x0;
    YY(:,1) = y0;
    for k=1:K    %run MPC for k=1:K
        UU1k  = exUU1(:,k:k+N-1); %control inputs of the intruder
      % UU1kp = exUU1(:,k+N_r:k+N-1);    %control inputs of the intruder after robust horizon
        UU1kp = exUU1(:,k:k+N-1);    %only classic MPC
        %only classic MPC
        UUrc  = []; 
        for j1   = 1:M_r^N_r   
                    UUrc  = [UUrc;UU1kp(:)];
        end
        %commented for the classic MPC
      %  for j1   = [-dphi 0 dphi]   %generating control of intruder for different scenarios
      %      for j2  = [-dphi 0 dphi]
      %          for j3  = [-dphi 0 dphi]
      %              UUrc  = [UUrc;[v;j1;v;j2;v;j3;UU1kp(:)]];
      %          end
      %      end
      %  end
        R          = S('x0',xx0,'lbg',[-inf*ones(length(ineq),1);zeros(length(eq),1)],'ubg',[zeros(length(ineq),1);zeros(length(eq),1)],'p',[XX(:,k);YY(:,k);UU1k(:);UUrc]);
        S.stats.return_status  %print status of solvers
        pri         = full(R.x);  %optimal decision variables
        xx0         = [pri(n_u+1:N*n_u);pri((N-1)*n_u+1:N*n_u); %initial guess of NLP for the next step (shift approach)
                       pri(N*n_u+n_x+1:N*n_u+(N+1)*n_x);pri(N*n_u+N*n_x+1:N*n_u+(N+1)*n_x)];
        for j    = 2:M_r^N_r+2
            xx0  = [xx0;pri(N*n_u+(j-1)*(N+1)*n_x+n_x+1:N*n_u+j*(N+1)*n_x);pri(N*n_u+j*(N+1)*n_x-n_x+1:N*n_u+j*(N+1)*n_x)];
        end
        UU          = pri(1:n_u*N); %MPC policy
        UUc         = reshape(UU,2,N);
        XX(:,k+1)   = [XX(1,k)+t_e*UUc(1,1)*cos(XX(3,k));XX(2,k)+t_e*UUc(1,1)*sin(XX(3,k));XX(3,k)+UUc(2,1)]; %run the ownship
        YY(:,k+1)   = [YY(1,k)+t_e*UU1(1,k)*cos(YY(3,k));YY(2,k)+t_e*UU1(1,k)*sin(YY(3,k));YY(3,k)+UU1(2,k)+0.0001*(l-L/2)]; %run the intruder with a possible uncertainty
        YY1(:,k+1)  = [YY1(1,k)+t_e*1.1*v*cos(YY1(3,k));YY1(2,k)+t_e*1.1*v*sin(YY1(3,k));YY1(3,k)+1.1*dphi];  %circles for Dubin path
        YY2(:,k+1)  = [YY2(1,k)+t_e*1.1*v*cos(YY2(3,k));YY2(2,k)+t_e*1.1*v*sin(YY2(3,k));YY2(3,k)-1.1*dphi];  %circles for Dubin path
        d(k)        = norm([XX(1,k)-YY(1,k);XX(2,k)-YY(2,k)]);   %distance between intruder-ownship
        linearV(k)  = UUc(1,1); %linear velocity of the ownship
        angularV(k) = UUc(2,1); %angular velocity of the ownship
        X_pred(k,:,:)  = reshape(pri(N*n_u+1:N*n_u+(N+1)*n_x),n_x,N+1); %ownship prediction
        Y_pred(k,:,:,:)= reshape(pri(N*n_u+2*(N+1)*n_x+1:end),n_x,N+1,M_r^N_r); %intruder prediction
    end
    j        = K; %specidy the plot of specific time
    Y_predk  = reshape(Y_pred(j,:,:,:),n_x,N+1,M_r^N_r); 
    for ii   = 1:M_r^N_r
        figure(j)
        plot(Y_predk(1,:,ii),Y_predk(2,:,ii),'k') %plot all predicted scenarios
        hold on
    end
    figure(j)
    plot(reshape(X_pred(j,1,:),1,N+1),reshape(X_pred(j,2,:),1,N+1),'k')
    hold on
    figure(j)
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
    plot(YY(1,:),YY(2,:),'--r')
    figure(1)
    plot(d)
    hold on
    figure(2)
    subplot(2,1,1)
    plot(linearV)
    hold on
    subplot(2,1,2)
    plot(angularV)
    hold on
    save('determ')
end