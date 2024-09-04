### Simulation Setup

A shorter horizon results in faster simulations. In this case, the simulation uses `N=10`, which leads to iterations taking less than 1 second each, making it suitable for real-time implementation.

1. **Run `iniRMPC.m`** to generate the `initialRMPC.mat`. This provides a feasible first guess for the solution.
2. **Run `ScenarioMPC.m`**.

> **Note:** Ensure that the horizon used in `iniRMPC.m` is the same as the one used in `ScenarioMPC.m`.
