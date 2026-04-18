from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import streamlit as st


st.set_page_config(page_title="Telekinetic Dataset Visualizer", layout="wide")


@st.cache_data(show_spinner=False)
def load_manifest(dataset_root: str) -> list[dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / "questions_manifest.jsonl"
    rows: list[dict[str, Any]] = []

    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    else:
        questions_dir = root / "questions"
        if not questions_dir.exists():
            raise FileNotFoundError(
                f"Could not find {manifest_path} or a questions/ directory under {root}"
            )
        for path in sorted(questions_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            row = {
                "question_id": payload.get("question_id", path.stem),
                "subset": payload.get("subset", "unknown"),
                "initial_image": payload.get("initial_image"),
                "prompt": payload.get("prompt", ""),
                "option_a": payload.get("option_a"),
                "option_b": payload.get("option_b"),
                "option_c": payload.get("option_c"),
                "option_d": payload.get("option_d"),
                "correct_choice": payload.get("correct_choice"),
                "question_json": str(path.relative_to(root)),
            }
            rows.append(row)

    enriched: list[dict[str, Any]] = []
    for row in rows:
        payload = load_question_payload(root, row)
        item = dict(row)
        item["_payload"] = payload
        item["_question_path"] = _resolve_question_path(root, row)
        item["_available_choices"] = available_choices(item, payload)
        item["_foil_types"] = foil_types(payload)
        enriched.append(item)

    return enriched


@st.cache_data(show_spinner=False)
def load_question_payload(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    question_path = _resolve_question_path(root, row)
    if question_path and question_path.exists():
        return json.loads(question_path.read_text())

    payload = {
        "question_id": row.get("question_id"),
        "subset": row.get("subset"),
        "prompt": row.get("prompt"),
        "initial_image": row.get("initial_image"),
        "correct_choice": row.get("correct_choice"),
        "choices": {},
    }
    for letter in ["A", "B", "C", "D"]:
        image_key = f"option_{letter.lower()}"
        if row.get(image_key):
            payload["choices"][letter] = {
                "image": row[image_key],
                "foil_type": "unknown",
            }
    return payload


def _resolve_question_path(root: Path, row: dict[str, Any]) -> Path | None:
    rel = row.get("question_json")
    if rel:
        return (root / rel).resolve()
    qid = row.get("question_id")
    if qid:
        candidate = root / "questions" / f"{qid}.json"
        if candidate.exists():
            return candidate.resolve()
    return None


def available_choices(row: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    if isinstance(payload.get("choices"), dict) and payload["choices"]:
        return sorted(payload["choices"].keys())
    choices: list[str] = []
    for letter in ["A", "B", "C", "D"]:
        if row.get(f"option_{letter.lower()}"):
            choices.append(letter)
    return choices


def foil_types(payload: dict[str, Any]) -> list[str]:
    result: set[str] = set()
    choices = payload.get("choices", {})
    if isinstance(choices, dict):
        for choice_payload in choices.values():
            foil = choice_payload.get("foil_type")
            if foil:
                result.add(str(foil))
    return sorted(result)


def resolve_image(root: Path, relative_or_absolute: str | None) -> Path | None:
    if not relative_or_absolute:
        return None
    p = Path(relative_or_absolute)
    if p.is_absolute() and p.exists():
        return p
    candidate = (root / relative_or_absolute).resolve()
    if candidate.exists():
        return candidate
    return None


def render_metadata_block(data: dict[str, Any]) -> None:
    if not data:
        st.caption("No metadata available.")
        return
    st.json(data, expanded=False)


st.title("Telekinetic Dataset Visualizer")
st.caption("Browse generated MCQ questions, inspect foils, and sanity-check dataset quality.")

with st.sidebar:
    st.header("Dataset")
    dataset_root_input = st.text_input(
        "Dataset root",
        value="debug_mcq_dataset_v2",
        help="Folder containing questions_manifest.jsonl and/or questions/ plus image assets.",
    )

    refresh = st.button("Reload dataset")
    if refresh:
        load_manifest.clear()
        load_question_payload.clear()

try:
    records = load_manifest(dataset_root_input)
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

root = Path(dataset_root_input).expanduser().resolve()

subset_values = sorted({str(r.get("subset", "unknown")) for r in records})
foil_values = sorted({foil for r in records for foil in r.get("_foil_types", [])})
choice_count_values = sorted({len(r.get("_available_choices", [])) for r in records})

with st.sidebar:
    st.header("Filters")
    selected_subsets = st.multiselect("Subset", subset_values, default=subset_values)
    selected_foil_types = st.multiselect("Foil type", foil_values, default=foil_values)
    selected_choice_counts = st.multiselect(
        "Number of answer options",
        choice_count_values,
        default=choice_count_values,
    )
    search_text = st.text_input("Search prompt / id / object text", value="")
    only_show_answer = st.checkbox("Reveal correct answer by default", value=False)
    randomize_button = st.button("Jump to random example")

filtered = []
needle = search_text.strip().lower()
for record in records:
    if selected_subsets and str(record.get("subset")) not in selected_subsets:
        continue
    if selected_choice_counts and len(record.get("_available_choices", [])) not in selected_choice_counts:
        continue
    if selected_foil_types:
        existing_foils = set(record.get("_foil_types", []))
        if existing_foils and existing_foils.isdisjoint(selected_foil_types):
            continue
    if needle:
        haystack = " ".join(
            [
                str(record.get("question_id", "")),
                str(record.get("prompt", "")),
                json.dumps(record.get("_payload", {}).get("action", {})),
            ]
        ).lower()
        if needle not in haystack:
            continue
    filtered.append(record)

if not filtered:
    st.warning("No questions match the current filters.")
    st.stop()

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if randomize_button:
    st.session_state.current_index = random.randrange(len(filtered))
else:
    st.session_state.current_index = max(0, min(st.session_state.current_index, len(filtered) - 1))

with st.sidebar:
    st.header("Navigation")
    chosen_index = st.number_input(
        "Question index",
        min_value=0,
        max_value=len(filtered) - 1,
        value=st.session_state.current_index,
        step=1,
    )
    st.session_state.current_index = int(chosen_index)
    st.caption(f"Showing {len(filtered)} / {len(records)} questions")

record = filtered[st.session_state.current_index]
payload = record["_payload"]
correct_choice = payload.get("correct_choice") or record.get("correct_choice")

nav1, nav2, nav3 = st.columns([1, 1, 4])
with nav1:
    if st.button("⬅ Prev", use_container_width=True) and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.rerun()
with nav2:
    if st.button("Next ➡", use_container_width=True) and st.session_state.current_index < len(filtered) - 1:
        st.session_state.current_index += 1
        st.rerun()
with nav3:
    st.markdown(
        f"**Question** `{record.get('question_id', 'unknown')}` &nbsp;&nbsp; "
        f"**Subset** `{record.get('subset', 'unknown')}`"
    )

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Prompt")
    st.write(payload.get("prompt") or record.get("prompt") or "")

    initial_image_path = resolve_image(root, payload.get("initial_image") or record.get("initial_image"))
    st.subheader("Initial scene")
    if initial_image_path:
        st.image(str(initial_image_path), use_container_width=True)
        st.caption(str(initial_image_path.relative_to(root)) if initial_image_path.is_relative_to(root) else str(initial_image_path))
    else:
        st.error("Initial image not found.")

with right:
    st.subheader("Summary")
    st.markdown(f"**Correct choice:** `{correct_choice}`")
    st.markdown(f"**Available choices:** `{', '.join(record['_available_choices'])}`")
    question_path = record.get("_question_path")
    if question_path:
        display_path = question_path.relative_to(root) if question_path.is_relative_to(root) else question_path
        st.markdown(f"**Question JSON:** `{display_path}`")

    meta_summary = {
        "action": payload.get("action"),
        "metadata": payload.get("metadata"),
    }
    render_metadata_block(meta_summary)

st.subheader("Answer options")
choice_letters = record["_available_choices"]
cols = st.columns(len(choice_letters))
for col, letter in zip(cols, choice_letters):
    choice_payload = payload.get("choices", {}).get(letter, {})
    image_rel = choice_payload.get("image") or record.get(f"option_{letter.lower()}")
    image_path = resolve_image(root, image_rel)
    with col:
        header = f"Option {letter}"
        if only_show_answer and letter == correct_choice:
            header += " ✅"
        st.markdown(f"**{header}**")
        if image_path:
            st.image(str(image_path), use_container_width=True)
        else:
            st.error("Image not found")
        foil_type = choice_payload.get("foil_type")
        if foil_type:
            st.caption(f"foil_type: {foil_type}")
        if st.checkbox(f"Show details {letter}", key=f"details_{record.get('question_id')}_{letter}"):
            st.json(choice_payload, expanded=False)

reveal = st.toggle("Reveal full answer and provenance", value=only_show_answer)
if reveal:
    st.subheader("Answer")
    st.success(f"Correct choice: {correct_choice}")
    if isinstance(payload.get("choices"), dict) and correct_choice in payload["choices"]:
        st.json(payload["choices"][correct_choice], expanded=True)

    st.subheader("Full question JSON")
    st.json(payload, expanded=False)

st.divider()

st.subheader("Dataset diagnostics")
diag1, diag2, diag3 = st.columns(3)
with diag1:
    st.metric("Questions loaded", len(records))
with diag2:
    st.metric("Questions after filters", len(filtered))
with diag3:
    st.metric("Subsets", len(subset_values))

with st.expander("Subset distribution", expanded=False):
    counts: dict[str, int] = {}
    for r in records:
        subset = str(r.get("subset", "unknown"))
        counts[subset] = counts.get(subset, 0) + 1
    st.json(counts)

with st.expander("Foil-type distribution", expanded=False):
    foil_counts: dict[str, int] = {}
    for r in records:
        for foil in r.get("_foil_types", []):
            foil_counts[foil] = foil_counts.get(foil, 0) + 1
    st.json(foil_counts)
