from simulator.env import TelekineticEnv
from benchmark.transition_generator import generate_one_transition
from benchmark.action_library import build_translation_action_library

env = TelekineticEnv(...)
library = build_translation_action_library(magnitudes=(0.06,))

result = generate_one_transition(
    env=env,
    output_dir="debug_dataset",
    action_library=library,
    seed=0,
)
print(result)