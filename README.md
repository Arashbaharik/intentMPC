# intentMPC

*You need to have Casadi on Matlab to run the code*
1) First run "iniRMPC.m" to generate "initialRMPC.mat" as a feasible initial guess for the MPC.

2) Classic MPC.m ,  ScenarioMPC.m and nominaMPC.m provide Matlab codes for the classic MPC, scenario-tree MPC, and nominal path (without safety constraint), respectively. These codes generate "determ.mat", "real{1:20}.mat" and "nominal.mat" data, respectively. Note that "real{1:20}.mat" corresponds to different uncertainties and "real10.mat" corresponds to the case without uncertainty. 

3)"intentMPC.m" and "nonIntentMPC.m" are for showing intent-based MPC and without intent MPC. These codes are for comparing the intent-awareness value and generating "intent.mat" and "nonintent.mat" data files.


4) All these data (with ".mat") and ".png" files (aircraft images) are used in "fig.m". This Matlab code generates all figures, used in the paper, called "f{1:7}.pdf". 
