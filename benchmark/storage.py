from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from telekinetics.benchmark.state_record import StateRecord


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def states_dir(self) -> Path:
        return self.root / "states"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def triplets_dir(self) -> Path:
        return self.root / "triplets"

    def ensure(self) -> None:
        self.states_dir.mkdir(parents=True, exist_ok=True)
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.triplets_dir.mkdir(parents=True, exist_ok=True)



def save_state_record(paths: DatasetPaths, state: StateRecord) -> Path:
    paths.ensure()
    dst = paths.states_dir / f"{state.state_hash}.json"
    dst.write_text(json.dumps(state.to_dict(), indent=2))
    return dst



def save_triplet_record(paths: DatasetPaths, triplet_id: str, payload: dict) -> Path:
    paths.ensure()
    dst = paths.triplets_dir / f"{triplet_id}.json"
    dst.write_text(json.dumps(payload, indent=2))
    return dst
