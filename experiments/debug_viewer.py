import mujoco.viewer
from telekinetics.control.telekinesis import TelekinesisActionInterface
from telekinetics.control.teleop import Teleop
from telekinetics.core.env import EnvConfig, TelekinesisEnv
from telekinetics.observations.oracle import OracleSceneObservation
from telekinetics.scenes.tabletop_obstacles import TabletopObstacleScene, TabletopObstacleSceneConfig
from telekinetics.tasks.place_in_region import PlaceObjectInRegionTask
from telekinetics.tasks.samplers import PlaceInRegionSampler

def build_env():
    scene = TabletopObstacleScene(TabletopObstacleSceneConfig(n_objects=4, seed=42, gravity_on=False, plane_z=0.53))
    action_interface = TelekinesisActionInterface(plane_z=0.53)
    observation_provider = OracleSceneObservation(include_obstacles=True)
    task = PlaceObjectInRegionTask(PlaceInRegionSampler())
    return TelekinesisEnv(
        scene=scene,
        action_interface=action_interface,
        observation_provider=observation_provider,
        task=task,
        cfg=EnvConfig(seed=42, settle_steps=20),
    )

def main():
    env = build_env()
    obs = env.reset()
    teleop = Teleop(n_objects=len(env.meta.object_names))
    print("Controls: 1..9 select object, arrows move selected object")
    print("Task:", obs["task"])
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        try:

            print(env.scene.metadata())
            print(env.get_object_states())

            while viewer.is_running():

                action = teleop.action(speed=0.0001, steps=4)
                obs, reward, done, info = env.step(action)
                if done:
                    print("Success:", info)
                    env.reset()
                viewer.sync()
        finally:
            teleop.close()

if __name__ == "__main__":
    main()
