import numpy as np
import os
from dataclasses import dataclass
from casadi import SX, nlpsol, vertcat, Function
import matplotlib.pyplot as plt
from symaware.base import (
    Agent,
    AgentCoordinator,
    AwarenessVector,
    Controller,
    Identifier,
    KnowledgeDatabase,
    PerceptionSystem,
    StateObservation,
    TimeIntervalAsyncLoopLock,
    TimeSeries,
    get_logger,
    initialize_logger,
    log,
    MultiAgentAwarenessVector,
    MultiAgentKnowledgeDatabase,
)
import scipy.io
import math
import random

try:
    import pybullet as p
    from symaware.simulators.pybullet import (
        Environment,
        Velocity2DModel,
        CubeEntity,
        Plane100Entity,
        AirplaneEntity,
    )

except ImportError as e:
    raise ImportError(
        "symaware-pybullet non found. "
        "Try running `pip install symaware-pybullet` or `pip install symaware[simulators]`"
    ) from e

ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
EGO_AGENT_ID = 0
INTRUDER_AGENT_ID = 1
EGO_AGENT_COLOR = (11 / 255, 143 / 255, 48 / 255, 1)
INTRUDER_AGENT_COLOR = (0.9, 0.2, 0.2, 1)


@dataclass(frozen=True)
class SkyEntity(Plane100Entity):
    def initialise(self):
        super().initialise()
        sky_tex = p.loadTexture(os.path.join(ASSETS_PATH, "sky.png"))
        p.changeVisualShape(self.entity_id, -1, textureUniqueId=sky_tex)

    def __hash__(self) -> Identifier:
        return super().__hash__()


@dataclass(frozen=True)
class StateAndTimeObservation(StateObservation):
    elapsed_time: float


class TreeMPCKnowledgeDatabase(KnowledgeDatabase):
    n_x: int  # state dimension
    n_u: int  # control dimension
    N: int  # prediction horizon
    phi: np.ndarray  # circle
    time_interval: int  # time interval
    X_T: np.ndarray  # target state
    v: float  # linear velocity
    dphi: float  # angular velocity
    max_sep = float  # maximum separation
    y_f: np.ndarray  # intruder waypoint
    M_r: int  # number of branches (uncertainties) in the scenario-tree MPC
    N_r: int  # robust horizon of the scenario-tree MPC
    xx0: np.ndarray  # initial decision variables values
    elapsed_time: float  # elapsed time from the beginning of the simulation


class AircraftPerceptionSystem(PerceptionSystem):
    """
    This perceptions system captures the state of both the agent and the intruder.
    The state if a tuple of three elements:

    - x position
    - y position
    - yaw angle

    Furthermore, the elapsed time of the simulation is also stored in the agent's self knowledge.
    """

    def __init__(
        self,
        agent_id: Identifier,
        env: Environment,
        async_loop_lock: TimeIntervalAsyncLoopLock | None = None,
    ):
        super().__init__(agent_id, env, async_loop_lock)
        self._last_position = np.zeros(2)

    def initialise_component(
        self,
        agent: Agent,
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase[TreeMPCKnowledgeDatabase],
    ):
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        self._last_position = initial_awareness_database.get(self._agent_id).state[:2]

    def _compute(self) -> dict[Identifier, StateAndTimeObservation]:
        observations: dict[Identifier, StateAndTimeObservation] = {}
        for agent_id in self._env.agent_states:
            state = self._env.get_agent_state(agent_id)
            orientation = p.getEulerFromQuaternion(state[3:])
            observations[agent_id] = StateAndTimeObservation(
                agent_id, np.append(state[:2], orientation[2]), self._env.elapsed_time
            )
        return observations

    def _update(self, perceived_information: dict[Identifier, StateAndTimeObservation]):
        super()._update(perceived_information)
        current_position = perceived_information.get(self._agent_id).state[:2]
        p.addUserDebugLine(
            [self._last_position[0], self._last_position[1], 2],
            [current_position[0], current_position[1], 2],
            EGO_AGENT_COLOR[:3] if self._agent_id == EGO_AGENT_ID else INTRUDER_AGENT_COLOR[:3],
            3,
        )
        self._last_position = current_position
        if self._agent_id not in perceived_information:
            raise ValueError(f"Agent {self._agent_id} not in perceived information")
        self._agent.self_knowledge["elapsed_time"] = perceived_information.get(self._agent_id).elapsed_time


class IntruderController(Controller):
    """
    Controller for the intruder.
    It will follow a Dublin path from its starting position the next waypoint.

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    l:
        Identifier of realization, used to generate random variations in the control inputs.
        If set to 0, no random variations will be generated.
        If 0 < l <= 20, a deterministic random variation will be generated.
        If l > 20, a random variation will be generated.
    async_loop_lock:
        Async loop lock to use for the controller when running in an async loop.
    """

    __LOGGER = get_logger(__name__, "IntruderController")

    def __init__(
        self,
        agent_id: Identifier,
        l: int = 0,
        async_loop_lock: TimeIntervalAsyncLoopLock | None = None,
    ):
        super().__init__(agent_id, async_loop_lock)
        self._control_inputs: np.ndarray
        self._k1: float
        self._k2: float
        self._l = l
        self._random_variation: float = 0.0

    @property
    def K(self):
        return self._k2 + self._k1 + 1

    def initialise_component(
        self,
        agent: Agent,
        initial_awareness_database: MultiAgentAwarenessVector,
        initial_knowledge_database: MultiAgentKnowledgeDatabase[TreeMPCKnowledgeDatabase],
    ):
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        v = initial_knowledge_database[agent.id]["v"]
        dphi = initial_knowledge_database[agent.id]["dphi"]
        y_f = initial_knowledge_database[agent.id]["y_f"]
        state = agent.self_state

        r = v / dphi
        ix = y_f[0] - state[0]
        iy = y_f[1] - state[1]
        ll = np.sqrt(ix**2 + (iy - 2 * r) ** 2 - 4 * r**2)
        # angle of L
        alpha = np.arcsin(2 * r / (np.sqrt((iy - 2 * r) ** 2 + ix**2))) + np.arctan((iy - 2 * r) / ix)
        self._k1 = round(alpha / dphi)
        # time for L
        self._k2 = self._k1 + round(ll / v)

        if self._l <= 0:
            self._random_variation = 0
        if self._l <= 20:
            self._random_variation = 0.0001 * math.floor(self._l / 2) * (-1 if self._l & 1 == 0 else 1)

        print(
            f"Agent {agent.id} - k1: {self._k1}, k2: {self._k2}. l: {self._l}, random_variation: {self._random_variation}"
        )
        # Only precompute the control inputs if there are subscribers
        if len(self._callbacks.get("intruder_control_inputs", [])) > 0:
            UU = v * np.ones((2, self.K))
            UU[1, : self._k1] = dphi
            UU[1, self._k1 : self._k1 + 1] = 0.24 * dphi
            UU[1, self._k1 + 1 : self._k2] = 0
            UU[1, self._k2 : self.K] = -dphi

            # Send the matrix of intruder control inputs to the ego controller
            self._notify("intruder_control_inputs", UU, v, self.K)

    @log(__LOGGER)
    def _compute(self) -> tuple[np.ndarray, TimeSeries]:
        """
        Compute the control input for the intruder.
        """
        dphi = self._agent.self_knowledge["dphi"]
        v = self._agent.self_knowledge["v"]
        # elapsed_time = self._agent.self_knowledge["elapsed_time"]
        elapsed_time = self._iteration_count

        if self._l > 20:
            self._random_variation = 0.0001 * (random.random() - 0.5) * (20)

        # from 0 to k1: circle, curving up
        if elapsed_time < self._k1:
            return np.array((v, dphi + self._random_variation)), TimeSeries()
        # from k1 to k1+1: circle, easing up
        if elapsed_time < self._k1 + 1:
            return np.array((v, 0.24 * dphi + self._random_variation)), TimeSeries()
        # from k1+1 to k2: straight line
        if elapsed_time < self._k2:
            return np.array((v, self._random_variation)), TimeSeries()
        # from k2 to k2+k1+1: circle, curving down
        if elapsed_time < self.K:
            return np.array((v, -dphi + self._random_variation)), TimeSeries()
        # no other control input, continue straight
        return np.array((v, self._random_variation)), TimeSeries()


class ScenarioTreeMPController(Controller):
    """
    Scenario Tree MPC.
    Based on paper: "Robust Model Predictive Control for Aircraft Intent-Aware Collision Avoidance".
    Available at: https://www.arxiv.org/abs/2408

    Args
    ----
    agent_id:
        Identifier of the agent this component belongs to
    L:
        number of realization.
        Corresponds to the number of scenarios to consider in the scenario-tree MPC.
        A higher number of realizations will increase the robustness of the MPC at the cost of computational time.
    async_loop_lock:
        Async loop lock to use for the controller when running in an async loop.
    """

    __LOGGER = get_logger(__name__, "TreeMPController")

    def __init__(
        self,
        agent_id: Identifier,
        async_loop_lock: TimeIntervalAsyncLoopLock | None = None,
    ):
        super().__init__(agent_id, async_loop_lock)
        self._xx0: np.ndarray  # non-linear optimization inputs
        self._lbg = np.ndarray  # lower bounds
        self._ubg = np.ndarray  # upper bounds
        self._S: Function  # optimization function
        self._UU: np.ndarray  # control inputs of the intruder
        self._extUU: np.ndarray  # control inputs of the intruder after robust horizon
        self._opts = {
            "ipopt": {
                "mu_target": 1e-4,
                "print_level": 0,
                "mu_init": 1e-4,
            },
            "print_time": False,
        }

    def set_intruder_control_inputs(self, UU: np.ndarray, v: float, K: int):
        self._UU = UU
        self._extUU = np.hstack((UU, np.array([v * np.ones(K), np.zeros(K)])))  # extend of control

    def initialise_component(
        self,
        agent: Agent,
        initial_awareness_database: dict[Identifier, AwarenessVector],
        initial_knowledge_database: dict[Identifier, TreeMPCKnowledgeDatabase],
    ):
        super().initialise_component(agent, initial_awareness_database, initial_knowledge_database)
        n_x = initial_knowledge_database[agent.id]["n_x"]
        n_u = initial_knowledge_database[agent.id]["n_u"]
        N = initial_knowledge_database[agent.id]["N"]
        M_r = initial_knowledge_database[agent.id]["M_r"]
        N_r = initial_knowledge_database[agent.id]["N_r"]
        max_sep = initial_knowledge_database[agent.id]["max_sep"]
        X_T = initial_knowledge_database[agent.id]["X_T"]
        v = initial_knowledge_database[agent.id]["v"]
        t_e = initial_knowledge_database[agent.id]["time_interval"]

        self._xx0 = initial_knowledge_database[agent.id]["xx0"]

        X = SX.sym("X", n_x, N + 1)  # state of the ownship
        Y = SX.sym("Y", n_x, N + 1)  # state of the intruder
        U = SX.sym("U", n_u, N)  # control of the ownship
        U1 = SX.sym("U1", n_u, N)  # control of the intruder
        X0 = SX.sym("X0", n_x, 1)  # initial state of the ownship
        Y0 = SX.sym("Y0", n_x, 1)  # initial state of the intruder
        # state of different scenarios for the intruder
        Yr = SX.sym("Yr", n_x, N + 1, M_r**N_r)
        # control of different scenarios for the intruder
        U1r = SX.sym("U1r", n_u, N, M_r**N_r)

        cost = 0  # cost function
        ineq: list[SX] = []  # inequality constraints
        eq: list[SX] = []  # equality constraints

        Yrc: list[SX] = []
        Urc: list[SX] = []

        for i in range(n_x):
            eq.append(X[i, 0] - X0[i])
        for i in range(n_x):
            eq.append(Y[i, 0] - Y0[i])

        for i in range(N):
            if i == 0:
                cost += 0.1 * (X[:, i] - X_T).T @ (X[:, i] - X_T)
            else:
                cost += +0.1 * (X[:, i] - X_T).T @ (X[:, i] - X_T) + 5500000 * (U[1, i] - U[1, i - 1]).T @ (
                    U[1, i] - U[1, i - 1]
                )

            ineq.append(-((X[:, i] - Y[:, i]).T @ np.diag([1, 1, 0])) @ (X[:, i] - Y[:, i]) + max_sep**2)
            ineq.append(U[0, i] - 0.9 * v)
            ineq.append(-U[0, i] + 0.6 * v)
            ineq.append(U[1, i] - 0.1)
            ineq.append(-U[1, i] - 0.1)

            eq.append(X[0, i + 1] - (X[0, i] + t_e * U[0, i] * np.cos(X[2, i])))
            eq.append(X[1, i + 1] - (X[1, i] + t_e * U[0, i] * np.sin(X[2, i])))
            eq.append(X[2, i + 1] - (X[2, i] + U[1, i]))

            eq.append(Y[0, i + 1] - (Y[0, i] + t_e * U1[0, i] * np.cos(Y[2, i])))
            eq.append(Y[1, i + 1] - (Y[1, i] + t_e * U1[0, i] * np.sin(Y[2, i])))
            eq.append(Y[2, i + 1] - (Y[2, i] + U1[1, i]))

        ineq.append(-((X[:, N] - Y[:, N]).T @ np.diag([1, 1, 0])) @ (X[:, N] - Y[:, N]) + max_sep**2)

        for j in range(M_r**N_r):  # all scenarios
            eq.append(Yr[j][0, 0] - Y0[0])
            eq.append(Yr[j][1, 0] - Y0[1])
            eq.append(Yr[j][2, 0] - Y0[2])
            for i in range(N):
                ineq.append(-((X[:, i] - Yr[j][:, i]).T @ np.diag([1, 1, 0])) @ (X[:, i] - Yr[j][:, i]) + max_sep**2)
                eq.append(Yr[j][0, i + 1] - (Yr[j][0, i] + t_e * U1r[j][0, i] * np.cos(Yr[j][2, i])))
                eq.append(Yr[j][1, i + 1] - (Yr[j][1, i] + t_e * U1r[j][0, i] * np.sin(Yr[j][2, i])))
                eq.append(Yr[j][2, i + 1] - (Yr[j][2, i] + U1r[j][1, i]))

            Yrc.append(Yr[j][:])
            Urc.append(U1r[j][:])

        cost += (X[:, N] - X_T).T @ (X[:, N] - X_T)

        nlp = {
            "x": vertcat(U[:], X[:], Y[:], *Yrc),
            "f": cost,
            "g": vertcat(*(ineq + eq)),
            "p": vertcat(X0, Y0, U1[:], *Urc),
        }

        self._lbg = np.concatenate((-np.inf * np.ones(len(ineq)), np.zeros(len(eq))))
        self._ubg = np.zeros(len(ineq) + len(eq))

        self._S = nlpsol("nplsolver", "ipopt", nlp, self._opts)

    @log(__LOGGER)
    def _compute(self) -> tuple[np.ndarray, TimeSeries]:
        dphi = self._agent.self_knowledge["dphi"]
        v = self._agent.self_knowledge["v"]
        n_u = self._agent.self_knowledge["n_u"]
        n_x = self._agent.self_knowledge["n_x"]
        N = self._agent.self_knowledge["N"]
        M_r = self._agent.self_knowledge["M_r"]
        N_r = self._agent.self_knowledge["N_r"]

        state = self._agent.self_state
        intruder_state = self._agent.awareness_database[self._agent.id + 1].state
        # elapsed_time = self._agent.self_knowledge["elapsed_time"]
        # k = int(elapsed_time + 0.9999 / self._agent.self_knowledge["time_interval"])
        k = self._iteration_count

        # control inputs of the intruder
        UU1k = self._extUU[:, k : k + N]
        # control inputs of the intruder after robust horizon
        UU1kp = self._extUU[:, k + N_r : k + N]
        UUrc = []
        # generating control of intruder for different scenarios
        for j1 in [-dphi, 0, dphi]:
            for j2 in [-dphi, 0, dphi]:
                for j3 in [-dphi, 0, dphi]:
                    UUrc.extend([v, j1, v, j2, v, j3] + UU1kp.flatten(order="F").tolist())
        UUrc = np.array(UUrc)
        P = np.concatenate((state, intruder_state, UU1k.flatten(order="F"), UUrc))

        R = self._S(x0=self._xx0, lbg=self._lbg, ubg=self._ubg, p=P)

        pri = np.array(R["x"]).flatten()  # optimal decision variables

        self._xx0 = np.concatenate(
            [
                pri[n_u : N * n_u],
                pri[(N - 1) * n_u : N * n_u],
                pri[N * n_u + n_x : N * n_u + (N + 1) * n_x],
                pri[N * n_u + N * n_x : N * n_u + (N + 1) * n_x],
            ]
        )
        for j in range(2, M_r**N_r + 3):
            self._xx0 = np.concatenate(
                [
                    self._xx0,
                    pri[N * n_u + (j - 1) * (N + 1) * n_x + n_x : N * n_u + j * (N + 1) * n_x],
                    pri[N * n_u + j * (N + 1) * n_x - n_x : N * n_u + j * (N + 1) * n_x],
                ]
            )

        return pri[: n_u * N].reshape(2, N, order="F")[:, 0], TimeSeries()


class Snapshot:

    counter: int = 0

    def save_snapshot(self, env: Environment):
        return # Comment this line to save the images
        img = env.take_screenshot(1440 * 3, 1120 * 3)
        plt.imsave(f"img/frame-{self.counter}.jpeg", img)
        self.counter += 1


def main():
    ###########################################################
    # 0. Parameters                                           #
    ###########################################################
    TIME_INTERVAL = 1
    LOG_LEVEL = "INFO"
    N = 10
    L = 0

    initialize_logger(LOG_LEVEL)

    # Load the initial decision variables values
    mat = scipy.io.loadmat(os.path.join(ASSETS_PATH, f"{N}_initialRMPC.mat"))
    snapshot = Snapshot()

    ###########################################################
    # 1. Create the environment and add the obstacles         #
    ###########################################################
    env = Environment(plane_scaling=-1, gravity=0)
    env.add_on_stepped(snapshot.save_snapshot)

    agent_coordinator = AgentCoordinator[TreeMPCKnowledgeDatabase](env)

    shared_knowledge = TreeMPCKnowledgeDatabase(
        n_x=3,
        n_u=2,
        N=N,
        phi=np.arange(0, 2 * np.pi, 0.01),
        time_interval=1,
        X_T=np.array([420, 560, np.pi / 2]),
        v=10,
        dphi=0.07,
        max_sep=np.sqrt(2000),
        y_f=np.array([600, 450, 0]),
        M_r=3,
        N_r=3,
        xx0=np.array(mat["xx0"]).squeeze(),
    )

    ###########################################################
    # 2. Create the ego and assign it an entity and model     #
    ###########################################################
    ego = Agent[TreeMPCKnowledgeDatabase](
        EGO_AGENT_ID,
        AirplaneEntity(
            EGO_AGENT_ID,
            model=Velocity2DModel(EGO_AGENT_ID, time_step=1.0),
            position=np.array([0, 0, 2]),
            global_scaling=2,
            rgb_colors=EGO_AGENT_COLOR,
        ),
    )

    ############################################################
    # 3. Create the intruder and assign it an entity and model #
    ############################################################
    intruder = Agent[TreeMPCKnowledgeDatabase](
        INTRUDER_AGENT_ID,
        AirplaneEntity(
            INTRUDER_AGENT_ID,
            model=Velocity2DModel(INTRUDER_AGENT_ID, time_step=1.0),
            position=np.array([-120, 90, 2]),
            global_scaling=2,
            rgb_colors=INTRUDER_AGENT_COLOR,
        ),
    )

    ###########################################################
    # 4. Add the agent to the environment                     #
    ###########################################################
    env.add_agents((ego, intruder))

    ###########################################################
    # 5. Create and set the component of the agents           #
    ###########################################################
    ego.add_components(
        AircraftPerceptionSystem(ego.id, env),
        ScenarioTreeMPController(ego.id),
    )
    intruder.add_components(
        AircraftPerceptionSystem(intruder.id, env),
        IntruderController(intruder.id, l=L),
    )

    intruder.controller.add(
        "intruder_control_inputs",
        ego.controller.set_intruder_control_inputs,
    )

    ############################################################
    # 6. Initialise both agent with some starting information  #
    ############################################################
    ego.initialise_agent(
        AwarenessVector(ego.id, np.array((0, 0, 0))),
        {ego.id: shared_knowledge},
    )
    intruder.initialise_agent(
        AwarenessVector(intruder.id, np.array((intruder.entity.position[0], intruder.entity.position[1], 0))),
        {intruder.id: shared_knowledge},
    )

    ###########################################################
    # 7. Add the agents to the coordinator                    #
    ###########################################################
    agent_coordinator.add_agents((ego, intruder))
    # agent_coordinator.add_agents((intruder))

    ###########################################################
    # 8. Run the simulation                                   #
    ###########################################################
    env.set_debug_camera_position(distance=1, yaw=0, pitch=-89.999, position=(250, 270, 300))
    env.disable_debug_visualizer()

    # Add the sky background
    sky_entity = SkyEntity(position=[-160, -100, -5], global_scaling=420)
    env.add_entities((sky_entity,))

    # Add the targets for demonstration purposes
    ego_target = CubeEntity(
        position=[420, 560, -4],
        global_scaling=4,
        rgb_colors=(11 / 255, 143 / 255, 48 / 255, 1),
    )
    intruder_target = CubeEntity(position=[600, 450, -4], global_scaling=4, rgb_colors=(0.9, 0.2, 0.2, 1))
    env.add_entities((ego_target, intruder_target))

    agent_coordinator.run(TIME_INTERVAL, steps_limit=82)


if __name__ == "__main__":
    main()
