# benchmark

Benchmark package layered on top of `telekinetics`.

## Scope

This package owns:
- symbolic benchmark actions
- action sampling
- state capture packaging and hashing
- transition / triplet generation
- dataset storage

It does **not** own the low-level simulator control API. That remains in
`telekinetics.core.action` and the telekinesis backend.

## Intended boundary

- `telekinetics`: execution/runtime layer
- `benchmark`: semantics/data layer

## Minimal usage

```python
from telekinetics.scenes.tabletop_obstacles import TabletopObstacleScene, TabletopObstacleSceneConfig
from telekinetics.control.telekinesis import TelekinesisActionInterface
from telekinetics.core.env import TelekinesisEnv
from telekinetics.observations.oracle import OracleObservationProvider

from telekinetics.benchmark import (
    build_translation_action_library,
    sample_action_instance,
    generate_single_transition,
)

scene = TabletopObstacleScene(TabletopObstacleSceneConfig())
env = TelekinesisEnv(
    scene=scene,
    action_interface=TelekinesisActionInterface(),
    observation_provider=OracleObservationProvider(),
)
env.reset(seed=0)

action_library = build_translation_action_library(magnitudes=(0.06, 0.10))
action = sample_action_instance(env.rng, env.get_object_states(), action_library)
result = generate_single_transition(env, action, dataset_root="dataset", seed=0)
```
