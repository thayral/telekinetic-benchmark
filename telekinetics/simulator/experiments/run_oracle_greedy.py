import statistics
from telekinetics.simulator.agents.oracle_greedy import OracleGreedyAgent
from telekinetics.simulator.control.telekinesis import TelekinesisActionInterface
from telekinetics.simulator.core.env import EnvConfig, TelekinesisEnv
from telekinetics.simulator.observations.oracle import OracleSceneObservation
from telekinetics.simulator.scenes.tabletop_obstacles import TabletopObstacleScene, TabletopObstacleSceneConfig
from telekinetics.simulator.tasks.place_in_region import PlaceObjectInRegionTask
from telekinetics.simulator.tasks.samplers import PlaceInRegionSampler

import mujoco.viewer

import time

def build_env(seed: int = 42) -> TelekinesisEnv:
    scene = TabletopObstacleScene(TabletopObstacleSceneConfig(n_objects=4, seed=seed, gravity_on=False, plane_z=0.53))
    action_interface = TelekinesisActionInterface(plane_z=0.53, max_mocap_offset=0.04)
    observation_provider = OracleSceneObservation(include_obstacles=True)
    task = PlaceObjectInRegionTask(PlaceInRegionSampler())
    return TelekinesisEnv(
        scene=scene,
        action_interface=action_interface,
        observation_provider=observation_provider,
        task=task,
        cfg=EnvConfig(seed=seed, settle_steps=20),
    )

def final_goal_distance(obs: dict) -> float:
    target_idx = int(obs["task"]["target_object"])
    goal_x, goal_y = obs["task"]["goal_center_xy"]
    for obj in obs["objects"]:
        if int(obj["index"]) == target_idx:
            dx = float(obj["pos"][0]) - float(goal_x)
            dy = float(obj["pos"][1]) - float(goal_y)
            return (dx * dx + dy * dy) ** 0.5
    raise RuntimeError("Target object missing from observation.")

def run_episode(env: TelekinesisEnv, agent: OracleGreedyAgent, max_steps: int, seed: int):
    obs = env.reset(seed=seed)
    agent.reset()
    reward = 0.0
    for t in range(max_steps):
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            return {"success": True, "steps": t + 1, "final_distance": final_goal_distance(obs), "reward": reward, "info": info}
    return {"success": False, "steps": max_steps, "final_distance": final_goal_distance(obs), "reward": reward, "info": info}

def main():
    num_episodes = 20
    max_steps = 200
    env = build_env(seed=42)
    agent = OracleGreedyAgent(step_size=0.01, steps_per_action=4, goal_tolerance=0.01)

    # ## DEBUG MODE
    # with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    #     obs = env.reset()
    #     for t in range(max_steps):
    #         time.sleep(0.1)
    #         action = agent.act(obs)
    #         obs, reward, done, info = env.step(action)
    #         viewer.sync()
    #         results = {"success": done, "steps": t + 1, "final_distance": final_goal_distance(obs), "reward": reward, "info": info}
    # results = [results]


    results = [run_episode(env, agent, max_steps=max_steps, seed=1000 + ep) for ep in range(num_episodes)]

    success_rate = sum(r["success"] for r in results) / len(results)
    mean_steps = statistics.mean(r["steps"] for r in results)
    mean_final_distance = statistics.mean(r["final_distance"] for r in results)

    print("Oracle greedy baseline")
    print(f"Episodes: {num_episodes}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Mean episode steps: {mean_steps:.2f}")
    print(f"Mean final distance: {mean_final_distance:.4f}")
    print("\nPer-episode summary:")
    for i, r in enumerate(results):
        print(f"  ep={i:02d}  success={int(r['success'])}  steps={r['steps']:3d}  final_dist={r['final_distance']:.4f}")



    env.close()

if __name__ == "__main__":
    main()
