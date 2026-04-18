from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
    dst.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    return dst


def load_state_record(path: str | Path) -> StateRecord:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return StateRecord.from_dict(data)


def load_state_record_by_hash(paths: DatasetPaths, state_hash: str) -> StateRecord:
    return load_state_record(paths.states_dir / f"{state_hash}.json")


def save_triplet_record(paths: DatasetPaths, triplet_id: str, payload: dict[str, Any]) -> Path:
    paths.ensure()
    dst = paths.triplets_dir / f"{triplet_id}.json"
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dst


def load_triplet_record(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
