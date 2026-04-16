import sys
from pathlib import Path
import numpy as np

# REPO_ROOT = Path("/path/to/telekinetic-benchmark-main")
# sys.path.insert(0, str(REPO_ROOT.parent))

from telekinetics.simulator.scenes.tabletop_obstacles import (
    TabletopObstacleScene,
    TabletopObstacleSceneConfig,
)
from telekinetics.simulator.control.telekinesis import TelekinesisActionInterface
from telekinetics.simulator.core.env import TelekinesisEnv
from telekinetics.simulator.observations.oracle import OracleSceneObservation

from telekinetics.benchmark.action_library import (
    build_translation_action_library,
    sample_action_instance,
)
from telekinetics.benchmark.transition_generator import generate_single_transition


def main():
    rng = np.random.default_rng(0)

    # 1) Build a simple tabletop scene
    scene = TabletopObstacleScene(
        TabletopObstacleSceneConfig(
            n_objects=4,
            seed=0,
        )
    )

    # 2) Build the simulator environment
    env = TelekinesisEnv(
        scene=scene,
        action_interface=TelekinesisActionInterface(),
        observation_provider=OracleSceneObservation(),
    )

    # 3) Reset once before reading object states
    env.reset(seed=0)

    # 4) Build a small symbolic action library
    library = build_translation_action_library(magnitudes=(0.06,))

    # 5) Sample one action on one object
    object_states = env.get_object_states()
    symbolic_action = sample_action_instance(
        rng=rng,
        object_states=object_states,
        action_library=library,
        rollout_steps=80,
        settle_steps=10,
    )

    print("Sampled action:", symbolic_action)

    # 6) Generate and save one transition
    result = generate_single_transition(
        env=env,
        symbolic_action=symbolic_action,
        dataset_root="debug_dataset",
        seed=0,
    )

    print("Triplet id:", result["triplet_id"])
    print("Triplet json:", result["triplet"])

    env.close()


if __name__ == "__main__":
    main()