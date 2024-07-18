clc;
clear all;
close all;
import casadi.*; 
n_x  = 3;
n_u  = 2;
N    = 30;
phi  = 0:0.01:2*pi;
t_e  = 1;
X_T  = [0;0;-pi/2];
v    = 10;
dphi = 0.07;
max_sep = sqrt(2000); 
x0   = [0;230;-pi/2];
y0   =[0;0;pi/2];
X    = SX .sym('X',n_x,N+1);
Y    = SX .sym('Y',n_x,N+1);
U    = SX .sym('U',n_u,N);
U1    = SX .sym('U1',n_u,N);
X0    = SX .sym('X0',n_x,1);
Y0    = SX .sym('Y0',n_x,1);
Cost  = 0;
ineq  = [];
eq    = [X(:,1)-X0;Y(:,1)-Y0];
for i = 1:N
    if i  ==1
        Cost = Cost+0.1*(X(:,i)-X_T)'*(X(:,i)-X_T);
    elseif i>1
        Cost = Cost+0.1*(X(:,i)-X_T)'*(X(:,i)-X_T)+5500000*(U(2,i)-U(2,i-1))'*(U(2,i)-U(2,i-1));
    end
    ineq    =  [ineq;-1*norm(X(:,i)-Y(:,i))^2+1*max_sep^2;U(1,i)-0.9*v;-U(1,i);U(2,i)-0.1;-U(2,i)-0.1]; 
    eq      =  [eq;X(:,i+1)-[X(1,i)+t_e*U(1,i)*cos(X(3,i));X(2,i)+t_e*U(1,i)*sin(X(3,i));X(3,i)+U(2,i)];
    Y(:,i+1)-[Y(1,i)+t_e*U1(1,i)*cos(Y(3,i));Y(2,i)+t_e*U1(1,i)*sin(Y(3,i));Y(3,i)+U1(2,i)]];    
end
Cost   = Cost+1*(X(:,N+1)-X_T)'*(X(:,N+1)-X_T);
nlp   = struct('x',[U(:);X(:);Y(:)], 'f',Cost, 'g',[ineq;eq],'p',[X0;Y0;U1(:)]);
opts  = struct;
opts.ipopt.mu_target   = 1e-4;
opts.ipopt.mu_init     = 1e-4;
opts.ipopt.print_level = 0;
opts.print_time        = false;
S     = nlpsol('S', 'ipopt', nlp,opts);
UU0   =  [v*ones(1,N);0*dphi*ones(1,N)]; %intent
XX(:,1) = x0;
YY(:,1) = y0;
xx0     = 0.1*ones((n_u+2*n_x)*N+2*n_x,1);
K       = 30;
UU1     = [v*ones(1,K);-dphi*ones(1,K)];
for k=1:K
    R          = S('x0',xx0,'lbg',[-inf*ones(length(ineq),1);zeros(length(eq),1)],'ubg',[zeros(length(ineq),1);zeros(length(eq),1)],'p',[XX(:,k);YY(:,k);UU0(:)]);
    S.stats.return_status
    pri         = full(R.x);
    xx0=[pri(n_u+1:N*n_u);pri((N-1)*n_u+1:N*n_u);
        pri(N*n_u+n_x+1:N*n_u+(N+1)*n_x);pri(N*n_u+N*n_x+1:N*n_u+(N+1)*n_x);
        pri(N*n_u+(N+1)*n_x+n_x+1:N*n_u+2*(N+1)*n_x);pri(N*n_u+(2*N+1)*n_x+1:N*n_u+2*(N+1)*n_x)];
    UU          = pri(1:n_u*N);
    UUc         = reshape(UU,2,N);
    XX(:,k+1) = [XX(1,k)+t_e*UUc(1,1)*cos(XX(3,k));XX(2,k)+t_e*UUc(1,1)*sin(XX(3,k));XX(3,k)+UUc(2,1)];
    YY(:,k+1) = [YY(1,k)+t_e*UU1(1,k)*cos(YY(3,k));YY(2,k)+t_e*UU1(1,k)*sin(YY(3,k));YY(3,k)+UU1(2,k)];
    d(k)      =  norm([XX(1,k)-YY(1,k);XX(2,k)-YY(2,k)]);
    linearV(k)  = UUc(1,1);
    angularV(k) = UUc(2,1);
end
plot(XX(1,1:end),XX(2,1:end),'b')
hold on
plot(YY(1,1:end),YY(2,1:end),'r')
save('nonintent')