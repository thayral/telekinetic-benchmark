from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np

from telekinetics.simulator.scenes.tabletop_obstacles import (
    TabletopObstacleSceneConfig,
    TabletopObstacleSceneFactory,
    build_env_from_scene,
    build_env_from_state_record,
)

from telekinetics.benchmark.action_library import build_translation_action_library
from telekinetics.benchmark.symbolic_actions import ActionInstance, ActionSpec, instantiate_action
from telekinetics.benchmark.action_adapter import symbolic_to_telekinetic_action
from telekinetics.benchmark.state_record import capture_state
from telekinetics.benchmark.storage import DatasetPaths, save_state_record


CHOICES = ("A", "B", "C")
FOIL_CATEGORIES = ("wrong_object", "wrong_direction", "wrong_scene")
ALL_DIRECTIONS = ("forward", "back", "left", "right")

DEFAULT_MAGNITUDES = (0.5,)
DEFAULT_ROLLOUT_STEPS = 500
DEFAULT_SETTLE_STEPS = 20


def _question_id(category: str, seed: int, start_hash: str, action: ActionInstance) -> str:
    raw = f"{category}|{seed}|{start_hash}|{action.spec.action_id}|{action.target_name}".encode("utf-8")
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


def _clone_spec_with_direction(spec: ActionSpec, direction: str) -> ActionSpec:
    mag_tag = str(float(spec.magnitude)).replace(".", "p")
    return ActionSpec(
        action_id=f"translate_{spec.frame}_{direction}_{mag_tag}",
        action_type=spec.action_type,
        frame=spec.frame,
        direction=direction,
        magnitude=float(spec.magnitude),
        label=f"move {direction}",
    )


def _clone_action(
    base: ActionInstance,
    *,
    target_index: int | None = None,
    target_name: str | None = None,
    direction: str | None = None,
) -> ActionInstance:
    spec = base.spec if direction is None else _clone_spec_with_direction(base.spec, direction)
    return instantiate_action(
        spec=spec,
        target_index=base.target_index if target_index is None else target_index,
        target_name=base.target_name if target_name is None else target_name,
        rollout_steps=base.rollout_steps,
        settle_steps=base.settle_steps,
    )


def _sample_correct_action(
    rng,
    object_states: list[dict],
    *,
    magnitudes: Iterable[float],
    rollout_steps: int,
    settle_steps: int,
) -> ActionInstance:
    library = build_translation_action_library(magnitudes=magnitudes)
    obj = object_states[int(rng.integers(len(object_states)))]
    spec = library[int(rng.integers(len(library)))]
    target_name = obj.get("label") or obj.get("name") or f"object_{obj['index']}"
    return instantiate_action(
        spec=spec,
        target_index=int(obj["index"]),
        target_name=target_name,
        rollout_steps=rollout_steps,
        settle_steps=settle_steps,
    )


def _run_action_from_state_record(
    start_state,
    action: ActionInstance,
    *,
    cfg: TabletopObstacleSceneConfig,
    paths: DatasetPaths,
    prefix: str,
    camera_name: str,
    seed: int,
):
    """Canonical replay path: rebuild env from saved state, then run the action."""
    env = build_env_from_state_record(start_state, config=cfg)
    try:
        env.step(symbolic_to_telekinetic_action(action))
        end_state = _render_and_save(env, paths, prefix=prefix, camera_name=camera_name, seed=seed)
    finally:
        env.close()
    return {
        "foil_type": prefix.split("_", 1)[0],
        "state": end_state,
        "generation_action": action.to_dict(),
    }


def _build_wrong_object_foils(rng, object_states: list[dict], correct_action: ActionInstance) -> list[ActionInstance]:
    candidates = [o for o in object_states if int(o["index"]) != int(correct_action.target_index)]
    if len(candidates) < 2:
        raise ValueError("Need at least two non-target objects for wrong_object foils")
    chosen_indices = rng.choice(len(candidates), size=2, replace=False)
    foils: list[ActionInstance] = []
    for choice_idx in np.atleast_1d(chosen_indices):
        obj = candidates[int(choice_idx)]
        foils.append(
            _clone_action(
                correct_action,
                target_index=int(obj["index"]),
                target_name=obj.get("label") or obj.get("name") or f"object_{obj['index']}",
            )
        )
    return foils


def _build_wrong_direction_foils(rng, correct_action: ActionInstance) -> list[ActionInstance]:
    alt_dirs = [d for d in ALL_DIRECTIONS if d != correct_action.spec.direction]
    chosen_indices = rng.choice(len(alt_dirs), size=2, replace=False)
    return [_clone_action(correct_action, direction=alt_dirs[int(i)]) for i in np.atleast_1d(chosen_indices)]


def _build_wrong_scene_foils(
    rng,
    correct_action: ActionInstance,
    *,
    cfg: TabletopObstacleSceneConfig,
    factory: TabletopObstacleSceneFactory,
    paths: DatasetPaths,
    camera_name: str,
    seed: int,
) -> list[dict]:
    """Generate unrelated answer images from genuinely fresh scenes.

    This now rebuilds new scene/env pairs instead of using env.reset(...), so the
    foil is structurally disconnected from the question state.
    """
    foils: list[dict] = []
    seen_hashes: set[str] = set()
    attempts = 0
    while len(foils) < 2 and attempts < 20:
        attempts += 1
        fresh_seed = int(seed + 1000 + attempts)
        fresh_scene = factory.create(seed=fresh_seed)
        fresh_env = build_env_from_scene(fresh_scene)
        try:
            fresh_env.reset(seed=fresh_seed)
            object_states = fresh_env.get_object_states()
            if not object_states:
                raise RuntimeError("No objects found in fresh_env during wrong_scene foil generation")
            fresh_action = _sample_correct_action(
                rng,
                object_states,
                magnitudes=(correct_action.spec.magnitude,),
                rollout_steps=correct_action.rollout_steps,
                settle_steps=correct_action.settle_steps,
            )
            fresh_env.step(symbolic_to_telekinetic_action(fresh_action))
            end_state = _render_and_save(
                fresh_env,
                paths,
                prefix=f"wrong_scene_{attempts}",
                camera_name=camera_name,
                seed=fresh_seed,
            )
        finally:
            fresh_env.close()
        if end_state.state_hash in seen_hashes:
            continue
        seen_hashes.add(end_state.state_hash)
        foils.append(
            {
                "foil_type": "wrong_scene",
                "state": end_state,
                "generation_action": fresh_action.to_dict(),
                "fresh_scene_seed": fresh_seed,
            }
        )
    if len(foils) < 2:
        raise RuntimeError("Could not generate two unique wrong_scene foils")
    return foils


def _make_prompt(action: ActionInstance) -> str:
    return (
        f"Given the initial scene, what is the resulting scene after moving the "
        f"{action.target_name} {action.spec.direction}?"
    )


def generate_single_question(
    *,
    # env,
    cfg: TabletopObstacleSceneConfig,
    factory: TabletopObstacleSceneFactory,
    paths: DatasetPaths,
    rng,
    category: str,
    question_seed: int,
    n_objects: int,
    camera_name: str,
    magnitudes: Iterable[float],
    rollout_steps: int,
    settle_steps: int,
) -> tuple[dict, Path]:
    if category not in FOIL_CATEGORIES:
        raise ValueError(f"Unsupported category: {category}")


    scene = factory.create(seed=question_seed)
    env = build_env_from_scene(scene)
    env.reset(seed=question_seed)
    start_state = _render_and_save(env, paths, prefix="start", camera_name=camera_name, seed=question_seed)
    object_states = env.get_object_states()
    if not object_states:
        raise RuntimeError("No objects found after env.reset(); scene/env wiring is inconsistent")

    correct_action = _sample_correct_action(
        rng,
        object_states,
        magnitudes=magnitudes,
        rollout_steps=rollout_steps,
        settle_steps=settle_steps,
    )

    correct_rollout = _run_action_from_state_record(
        start_state,
        correct_action,
        cfg=cfg,
        paths=paths,
        prefix="correct",
        camera_name=camera_name,
        seed=question_seed,
    )

    if category == "wrong_object":
        foil_actions = _build_wrong_object_foils(rng, object_states, correct_action)
        foil_rollouts = [
            _run_action_from_state_record(
                start_state,
                action,
                cfg=cfg,
                paths=paths,
                prefix=f"wrong_object_{i}",
                camera_name=camera_name,
                seed=question_seed,
            )
            for i, action in enumerate(foil_actions)
        ]
    elif category == "wrong_direction":
        foil_actions = _build_wrong_direction_foils(rng, correct_action)
        foil_rollouts = [
            _run_action_from_state_record(
                start_state,
                action,
                cfg=cfg,
                paths=paths,
                prefix=f"wrong_direction_{i}",
                camera_name=camera_name,
                seed=question_seed,
            )
            for i, action in enumerate(foil_actions)
        ]
    else:
        foil_rollouts = _build_wrong_scene_foils(
            rng,
            correct_action,
            cfg=cfg,
            factory=factory,
            paths=paths,
            camera_name=camera_name,
            seed=question_seed,
        )

    candidates = [correct_rollout, *foil_rollouts]
    unique_hashes = {c["state"].state_hash for c in candidates}
    if len(unique_hashes) != 3:
        raise RuntimeError(
            f"Question seed {question_seed} in category {category} did not produce 3 unique answer images"
        )

    perm = rng.permutation(len(candidates))
    shuffled = [candidates[int(i)] for i in perm]
    correct_choice = CHOICES[next(i for i, old_i in enumerate(perm) if int(old_i) == 0)]

    qid = _question_id(category, question_seed, start_state.state_hash, correct_action)
    question = {
        "question_id": qid,
        "subset": category,
        "foil_type": category,
        "seed": int(question_seed),
        "camera_name": camera_name,
        "n_objects": int(n_objects),
        "initial_image": start_state.render_path,
        "prompt": _make_prompt(correct_action),
        "action": correct_action.to_dict(),
        "correct_choice": correct_choice,
        "choices": {},
        "metadata": {
            "start_state_hash": start_state.state_hash,
            "scene_name": type(env.scene).__name__,
            "magnitudes": [float(m) for m in magnitudes],
            "rollout_steps": int(rollout_steps),
            "settle_steps": int(settle_steps),
        },
    }

    for idx, candidate in enumerate(shuffled):
        choice = CHOICES[idx]
        state = candidate["state"]
        question["choices"][choice] = {
            "image": state.render_path,
            "resulting_state_hash": state.state_hash,
            "foil_type": candidate["foil_type"],
            "generation_action": candidate.get("generation_action"),
            "fresh_scene_seed": candidate.get("fresh_scene_seed"),
        }
        question[f"option_{choice.lower()}"] = state.render_path

    questions_dir = paths.root / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    out_path = questions_dir / f"{qid}.json"
    out_path.write_text(json.dumps(question, indent=2), encoding="utf-8")
    return question, out_path


def generate_mcq_dataset(
    dataset_root: str | Path = "mcq_dataset_v2",
    *,
    n_questions_per_category: int = 4,
    categories: Iterable[str] = FOIL_CATEGORIES,
    dataset_seed: int = 0,
    n_objects: int = 4,
    camera_name: str = "cam_oblique",
    magnitudes: Iterable[float] = DEFAULT_MAGNITUDES,
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
) -> list[dict]:
    paths = DatasetPaths(Path(dataset_root))
    paths.ensure()
    (paths.root / "questions").mkdir(parents=True, exist_ok=True)

    master_rng = np.random.RandomState(dataset_seed)

    categories = tuple(categories)
    invalid = [c for c in categories if c not in FOIL_CATEGORIES]
    if invalid:
        raise ValueError(f"Unsupported categories requested: {invalid}")


    cfg = TabletopObstacleSceneConfig(n_objects=n_objects)
    factory = TabletopObstacleSceneFactory(cfg)


    # scene_seed = int(master_rng.randint(0, 2**31 - 1))
    # scene = factory.create(seed=scene_seed)
    # env = build_env_from_scene(scene)

    rng = np.random.default_rng(dataset_seed)
    questions: list[dict] = []
    manifest_rows: list[dict] = []


    try:
        for category_index, category in enumerate(categories):
            made = 0
            attempt = 0
            while made < n_questions_per_category:
                attempt += 1
                # question_seed = int(seed + category_index * 10_000 + attempt)
                
                question_seed = int(master_rng.randint(0, 2**31 - 1))


                try:
                    question, out_path = generate_single_question(
                        # env=env,
                        cfg=cfg,
                        factory=factory,
                        paths=paths,
                        rng=rng,
                        category=category,
                        question_seed=question_seed,
                        n_objects=n_objects,
                        camera_name=camera_name,
                        magnitudes=magnitudes,
                        rollout_steps=rollout_steps,
                        settle_steps=settle_steps,
                    )
                except RuntimeError:
                    continue
                questions.append(question)
                manifest_rows.append(
                    {
                        "question_id": question["question_id"],
                        "subset": question["subset"],
                        "initial_image": question["initial_image"],
                        "prompt": question["prompt"],
                        "option_a": question["option_a"],
                        "option_b": question["option_b"],
                        "option_c": question["option_c"],
                        "correct_choice": question["correct_choice"],
                        "question_json": str(out_path.relative_to(paths.root)),
                    }
                )
                made += 1
    finally:
        # env.close()
        pass

    manifest_path = paths.root / "questions_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8")
    return questions


if __name__ == "__main__":
    questions = generate_mcq_dataset(
        dataset_root="debug_mcq_dataset_v2",
        n_questions_per_category=4,
        n_objects=4,
        dataset_seed=0,
        magnitudes=(0.5,),
        rollout_steps=500,
    )
    print(f"Generated {len(questions)} questions")
    counts = {}
    for q in questions:
        counts[q["subset"]] = counts.get(q["subset"], 0) + 1
    print(json.dumps(counts, indent=2))
