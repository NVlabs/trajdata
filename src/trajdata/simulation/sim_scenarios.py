"""
Simulation scenario runners and baseline policies.

A *policy* is any callable with signature::

    policy(obs: AgentBatch) -> Dict[str, StateArray]

where the returned dict maps ``agent_name → new_xyzh_state`` for the next
timestep.

Built-in policies
-----------------
* :class:`ConstantVelocityPolicy` – each agent continues at its last observed
  velocity (dead-reckoning).
* :class:`RandomWalkPolicy` – adds Gaussian noise to the last position.

High-level runner
-----------------
:class:`SimRunner` wraps a :class:`~trajdata.simulation.sim_scene.SimulationScene`
and runs it for a fixed number of steps, collecting observations and metrics.

Usage::

    from trajdata.simulation import SimulationScene
    from trajdata.simulation.sim_scenarios import SimRunner, ConstantVelocityPolicy
    from trajdata.simulation.sim_metrics import ADE, FDE

    policy = ConstantVelocityPolicy()
    runner = SimRunner(sim_scene, policy, max_steps=20)
    results = runner.run(metrics=[ADE(), FDE()])
    print(results)  # {"ade": {"agent_0": 1.2, ...}, "fde": {...}}
"""
from typing import Callable, Dict, List, Optional

import numpy as np

from trajdata.data_structures.state import StateArray
from trajdata.simulation.sim_metrics import SimMetric
from trajdata.simulation.sim_stats import SimStatistic


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------

class ConstantVelocityPolicy:
    """Propagate each agent at its last observed velocity (linear extrapolation).

    The policy reads the current-agent observation from the batch and steps
    each agent forward by ``dt * velocity``.
    """

    def __call__(self, obs, dt: float) -> Dict[str, "StateArray"]:
        """
        Args:
            obs: :class:`~trajdata.AgentBatch` from ``SimulationScene.get_obs()``.
            dt: Simulation time-step in seconds.

        Returns:
            Dict mapping ``agent_name → next StateArray`` in ``"x,y,z,h"`` format.
        """
        result = {}
        names = obs.agent_name if not isinstance(obs, dict) else obs["agent_name"]
        states = obs.curr_agent_state if not isinstance(obs, dict) else obs["curr_agent_state"]

        for i, name in enumerate(names):
            s = states[i].numpy()  # shape [D]; internal format x,y,z,xd,yd,xdd,ydd,h
            x   = float(s[0]) if s.shape[0] > 0 else 0.0
            y   = float(s[1]) if s.shape[0] > 1 else 0.0
            z   = float(s[2]) if s.shape[0] > 2 else 0.0
            vx  = float(s[3]) if s.shape[0] > 3 else 0.0
            vy  = float(s[4]) if s.shape[0] > 4 else 0.0
            h   = float(s[7]) if s.shape[0] > 7 else float(np.arctan2(vy, vx))

            next_state = np.array([x + vx * dt, y + vy * dt, z, h], dtype=np.float64)
            result[name] = StateArray.from_array(next_state, "x,y,z,h")
        return result


class RandomWalkPolicy:
    """Each agent takes a random step drawn from a Gaussian.

    Args:
        stddev: Standard deviation of position perturbation in metres (default 0.1).
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, stddev: float = 0.1, seed: Optional[int] = None) -> None:
        self.stddev = stddev
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, dt: float) -> Dict[str, "StateArray"]:
        result = {}
        names = obs.agent_name if not isinstance(obs, dict) else obs["agent_name"]
        states = obs.curr_agent_state if not isinstance(obs, dict) else obs["curr_agent_state"]

        for i, name in enumerate(names):
            s = states[i].numpy()
            x = float(s[0]) if s.shape[0] > 0 else 0.0
            y = float(s[1]) if s.shape[0] > 1 else 0.0
            z = float(s[2]) if s.shape[0] > 2 else 0.0
            h = float(s[7]) if s.shape[0] > 7 else 0.0
            noise = self.rng.normal(0.0, self.stddev, size=2)
            next_state = np.array([x + noise[0], y + noise[1], z, h], dtype=np.float64)
            result[name] = StateArray.from_array(next_state, "x,y,z,h")
        return result


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

PolicyFn = Callable  # (obs, dt) → Dict[str, StateArray]


class SimRunner:
    """Run a :class:`~trajdata.simulation.sim_scene.SimulationScene` with a policy.

    Args:
        sim_scene: An initialised ``SimulationScene`` instance.
        policy: A callable ``(obs, dt) → Dict[agent_name, StateArray]``.
        max_steps: Maximum number of simulation steps to run.
        dt: Override the scene time-step; if ``None`` the scene's own dt is used.
    """

    def __init__(
        self,
        sim_scene,
        policy: PolicyFn,
        max_steps: int = 50,
        dt: Optional[float] = None,
    ) -> None:
        self.sim_scene = sim_scene
        self.policy = policy
        self.max_steps = max_steps
        self._dt = dt

    @property
    def dt(self) -> float:
        if self._dt is not None:
            return self._dt
        return self.sim_scene.scene.dt

    def run(
        self,
        metrics: Optional[List[SimMetric]] = None,
        stats: Optional[List[SimStatistic]] = None,
        verbose: bool = False,
    ) -> dict:
        """Execute the simulation loop.

        Returns:
            A dict with keys ``"metrics"`` and/or ``"stats"`` containing the
            computed values, plus ``"steps"`` (number of steps executed).
        """
        obs = self.sim_scene.reset()
        steps = 0

        for step in range(self.max_steps):
            action = self.policy(obs, self.dt)
            obs = self.sim_scene.step(action, return_obs=True)
            steps += 1
            if verbose:
                print(f"Step {step + 1}/{self.max_steps}")

        self.sim_scene.finalize()

        results: dict = {"steps": steps}
        if metrics:
            results["metrics"] = self.sim_scene.get_metrics(metrics)
        if stats:
            results["stats"] = self.sim_scene.get_stats(stats)

        return results
