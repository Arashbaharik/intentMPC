### Simulation Setup

A shorter horizon results in faster simulations. In this case, the simulation uses `N=10`, which leads to iterations taking less than 1 second each, making it suitable for real-time implementation.

1. **Run `iniRMPC`** to generate the initial RMPC. This provides a feasible first guess for the solution.
2. **Run `ScenarioMPC`**.

> **Note:** Ensure that the horizon used in `iniRMPC` is the same as the one used in `ScenarioMPC`.
