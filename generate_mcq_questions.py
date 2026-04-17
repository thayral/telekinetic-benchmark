from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np

from telekinetics.simulator.scenes.tabletop_obstacles import (
    TabletopObstacleScene,
    TabletopObstacleSceneConfig,
)
from telekinetics.simulator.control.telekinesis import TelekinesisActionInterface
from telekinetics.simulator.core.env import TelekinesisEnv
from telekinetics.simulator.observations.oracle import OracleSceneObservation

from telekinetics.benchmark.action_library import build_translation_action_library
from telekinetics.benchmark.symbolic_actions import ActionInstance, instantiate_action
from telekinetics.benchmark.action_adapter import symbolic_to_telekinetic_action
from telekinetics.benchmark.state_record import capture_state, restore_state
from telekinetics.benchmark.storage import DatasetPaths, save_state_record


CHOICES = ("A", "B", "C", "D")
MIRROR_DIRECTION = {
    "left": "right",
    "right": "left",
    "forward": "back",
    "back": "forward",
}


def _question_id(seed: int, start_hash: str, action: ActionInstance) -> str:
    raw = f"{seed}|{start_hash}|{action.spec.action_id}|{action.target_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _render_and_save(env, paths: DatasetPaths, prefix: str, camera_name: str, seed: int | None):
    rgb = env.render_rgb(camera_name=camera_name)
    provisional = capture_state(env, seed=seed)
    render_name = f"{prefix}_{provisional.state_hash}.png"
    rel_render_path = f"renders/{render_name}"
    imageio.imwrite(paths.renders_dir / render_name, rgb)
    state = capture_state(env, render_path=rel_render_path, seed=seed)
    save_state_record(paths, state)
    return state


def _run_action_from_state(env, start_state, action: ActionInstance, paths: DatasetPaths, prefix: str, camera_name: str, seed: int):
    restore_state(env, start_state)
    env.step(symbolic_to_telekinetic_action(action))
    end_state = _render_and_save(env, paths, prefix=prefix, camera_name=camera_name, seed=seed)
    return {
        "action": action,
        "state": end_state,
    }


def _clone_action(base: ActionInstance, *, target_index: int | None = None, target_name: str | None = None, direction: str | None = None) -> ActionInstance:
    spec = base.spec
    if direction is not None:
        spec = type(spec)(
            action_id=f"translate_{spec.frame}_{direction}_{str(spec.magnitude).replace('.', 'p')}",
            action_type=spec.action_type,
            frame=spec.frame,
            direction=direction,
            magnitude=spec.magnitude,
            label=f"move {direction}",
        )
    return instantiate_action(
        spec=spec,
        target_index=base.target_index if target_index is None else target_index,
        target_name=base.target_name if target_name is None else target_name,
        rollout_steps=base.rollout_steps,
        settle_steps=base.settle_steps,
    )


def _pick_wrong_object(rng, object_states: list[dict], correct_action: ActionInstance):
    candidates = [o for o in object_states if int(o["index"]) != int(correct_action.target_index)]
    if not candidates:
        return None
    chosen = candidates[int(rng.integers(len(candidates)))]
    return _clone_action(
        correct_action,
        target_index=int(chosen["index"]),
        target_name=chosen.get("label") or chosen.get("name") or f"object_{chosen['index']}",
    )


def _build_candidate_actions(rng, object_states: list[dict], correct_action: ActionInstance) -> list[tuple[str, ActionInstance]]:
    candidates: list[tuple[str, ActionInstance]] = [("correct", correct_action)]

    mirrored_dir = MIRROR_DIRECTION[correct_action.spec.direction]
    candidates.append(("mirrored_direction", _clone_action(correct_action, direction=mirrored_dir)))

    wrong_object = _pick_wrong_object(rng, object_states, correct_action)
    if wrong_object is not None:
        candidates.append(("wrong_object", wrong_object))

    orthogonal_options = {
        "left": ["forward", "back"],
        "right": ["forward", "back"],
        "forward": ["left", "right"],
        "back": ["left", "right"],
    }
    alt_dir = orthogonal_options[correct_action.spec.direction][int(rng.integers(2))]
    candidates.append(("wrong_direction", _clone_action(correct_action, direction=alt_dir)))

    return candidates


def generate_single_mcq_question(
    dataset_root: str | Path = "mcq_dataset",
    seed: int = 0,
    n_objects: int = 4,
    camera_name: str = "cam_oblique",
):
    rng = np.random.default_rng(seed)
    paths = DatasetPaths(Path(dataset_root))
    paths.ensure()

    scene = TabletopObstacleScene(TabletopObstacleSceneConfig(n_objects=n_objects, seed=seed))
    env = TelekinesisEnv(
        scene=scene,
        action_interface=TelekinesisActionInterface(),
        observation_provider=OracleSceneObservation(),
    )

    try:
        env.reset(seed=seed)
        start_state = _render_and_save(env, paths, prefix="start", camera_name=camera_name, seed=seed)
        object_states = env.get_object_states()

        library = build_translation_action_library(magnitudes=(0.5,))
        target_obj = object_states[int(rng.integers(len(object_states)))]
        base_spec = library[int(rng.integers(len(library)))]
        correct_action = instantiate_action(
            spec=base_spec,
            target_index=int(target_obj["index"]),
            target_name=target_obj.get("label") or target_obj.get("name") or f"object_{target_obj['index']}",
            rollout_steps=500,
            settle_steps=15,
        )

        candidate_actions = _build_candidate_actions(rng, object_states, correct_action)
        rollouts = []
        seen_end_hashes = set()
        for foil_type, action in candidate_actions:
            result = _run_action_from_state(
                env,
                start_state,
                action,
                paths,
                prefix=foil_type,
                camera_name=camera_name,
                seed=seed,
            )
            end_hash = result["state"].state_hash
            if foil_type != "correct" and end_hash in seen_end_hashes:
                continue
            seen_end_hashes.add(end_hash)
            rollouts.append((foil_type, result))

        correct_index = next(i for i, (foil_type, _) in enumerate(rollouts) if foil_type == "correct")
        perm = rng.permutation(len(rollouts))
        shuffled = [rollouts[int(i)] for i in perm]
        correct_choice = CHOICES[next(i for i, old_i in enumerate(perm) if int(old_i) == correct_index)]

        qid = _question_id(seed, start_state.state_hash, correct_action)
        question = {
            "question_id": qid,
            "initial_image": start_state.render_path,
            "prompt": f"What happens if I {correct_action.text_label}?",
            "action": correct_action.to_dict(),
            "correct_choice": correct_choice,
            "subset": "basic_mcq",
            "seed": seed,
            "camera_name": camera_name,
            "choices": {},
            "metadata": {
                "start_state_hash": start_state.state_hash,
                "scene_name": type(env.scene).__name__,
            },
        }

        for idx, (foil_type, result) in enumerate(shuffled):
            choice = CHOICES[idx]
            end_state = result["state"]
            action = result["action"]
            question["choices"][choice] = {
                "image": end_state.render_path,
                "resulting_state_hash": end_state.state_hash,
                "foil_type": foil_type,
                "action_used_to_generate": action.to_dict(),
            }
            question[f"option_{choice.lower()}"] = end_state.render_path

        dst = paths.root / "questions"
        dst.mkdir(parents=True, exist_ok=True)
        out_path = dst / f"{qid}.json"
        out_path.write_text(json.dumps(question, indent=2))
        return question, out_path
    finally:
        env.close()


if __name__ == "__main__":
    question, path = generate_single_mcq_question(dataset_root="debug_mcq_dataset", seed=0)
    print(f"Saved question to: {path}")
    print(json.dumps(question, indent=2))
