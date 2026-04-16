from __future__ import annotations

from pathlib import Path
import hashlib

import imageio.v2 as imageio

from telekinetics.benchmark.action_adapter import symbolic_to_telekinetic_action
from telekinetics.benchmark.state_record import capture_state
from telekinetics.benchmark.storage import DatasetPaths, save_state_record, save_triplet_record



def _triplet_id(start_hash: str, end_hash: str, action_id: str, target_name: str) -> str:
    raw = f"{start_hash}|{end_hash}|{action_id}|{target_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]



def generate_single_transition(
    env,
    symbolic_action,
    dataset_root: str | Path,
    camera_name: str = "cam_oblique",
    seed: int | None = None,
) -> dict:
    """Generate and persist one causal triplet from the current env state."""
    paths = DatasetPaths(Path(dataset_root))
    paths.ensure()

    start_rgb = env.render_rgb(camera_name=camera_name)
    provisional_start = capture_state(env, seed=seed)
    start_render_name = f"{provisional_start.state_hash}.png"
    imageio.imwrite(paths.renders_dir / start_render_name, start_rgb)
    start_state = capture_state(env, render_path=f"renders/{start_render_name}", seed=seed)
    save_state_record(paths, start_state)

    env.step(symbolic_to_telekinetic_action(symbolic_action))

    end_rgb = env.render_rgb(camera_name=camera_name)
    provisional_end = capture_state(env, seed=seed)
    end_render_name = f"{provisional_end.state_hash}.png"
    imageio.imwrite(paths.renders_dir / end_render_name, end_rgb)
    end_state = capture_state(env, render_path=f"renders/{end_render_name}", seed=seed)
    save_state_record(paths, end_state)

    triplet = {
        "initial_state_hash": start_state.state_hash,
        "resulting_state_hash": end_state.state_hash,
        "input_img": start_state.render_path,
        "output_img": end_state.render_path,
        "action": symbolic_action.to_dict(),
        "meta": {
            "scene_name": type(env.scene).__name__,
            "camera_name": camera_name,
            "seed": seed,
        },
    }
    triplet_id = _triplet_id(
        start_hash=start_state.state_hash,
        end_hash=end_state.state_hash,
        action_id=symbolic_action.spec.action_id,
        target_name=symbolic_action.target_name,
    )
    save_triplet_record(paths, triplet_id, triplet)
    return {
        "triplet_id": triplet_id,
        "triplet": triplet,
        "start_state": start_state,
        "end_state": end_state,
    }
