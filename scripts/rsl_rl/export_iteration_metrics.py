"""Export all TensorBoard scalar events in one RSL-RL run to CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def export_tensorboard_scalars(log_dir: str | Path) -> tuple[Path, Path]:
    """Export scalar events to wide and long CSV files.

    The wide file contains one row per RSL-RL learning iteration and one column
    per TensorBoard tag. Some RSL-RL tags use elapsed-time values as their
    TensorBoard ``step`` instead of the learning-iteration index, so records are
    aligned by chronological ordinal within each tag. The long file preserves
    both the aligned iteration index and the original TensorBoard source step.

    When multiple event files contain the same ``tag, source_step`` pair, the
    record with the latest wall time is retained. This makes the export robust
    to resumed writers and event-file rotations.
    """

    run_dir = Path(log_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"RSL-RL log directory does not exist: {run_dir}")

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {run_dir}")

    scalar_records: dict[tuple[str, int], tuple[float, float]] = {}
    for event_file in event_files:
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                key = (tag, int(event.step))
                existing = scalar_records.get(key)
                if existing is None or float(event.wall_time) >= existing[1]:
                    scalar_records[key] = (float(event.value), float(event.wall_time))

    if not scalar_records:
        raise RuntimeError(f"No scalar summaries found in TensorBoard events under: {run_dir}")

    records_by_tag: dict[str, list[tuple[int, float, float]]] = {}
    for (tag, source_step), (value, wall_time) in scalar_records.items():
        records_by_tag.setdefault(tag, []).append((source_step, value, wall_time))
    for records in records_by_tag.values():
        records.sort(key=lambda record: (record[2], record[0]))

    long_path = run_dir / "iteration_metrics_long.csv"
    long_rows = []
    for tag, records in records_by_tag.items():
        for iteration, (source_step, value, wall_time) in enumerate(records):
            long_rows.append(
                {
                    "iteration": iteration,
                    "source_step": source_step,
                    "tag": tag,
                    "value": value,
                    "wall_time": wall_time,
                }
            )
    long_rows.sort(key=lambda row: (int(row["iteration"]), str(row["tag"])))
    with long_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("iteration", "source_step", "tag", "value", "wall_time"),
        )
        writer.writeheader()
        writer.writerows(long_rows)

    tags = sorted(records_by_tag)
    iteration_count = max(len(records) for records in records_by_tag.values())
    wide_path = run_dir / "iteration_metrics.csv"
    with wide_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("iteration", *tags))
        writer.writeheader()
        for iteration in range(iteration_count):
            row: dict[str, int | float] = {"iteration": iteration}
            for tag in tags:
                records = records_by_tag[tag]
                if iteration < len(records):
                    row[tag] = records[iteration][1]
            writer.writerow(row)

    print(
        "[IterationMetrics] Exported "
        f"{iteration_count} aligned iterations and {len(tags)} tags to:\n"
        f"  {wide_path}\n"
        f"  {long_path}"
    )
    return wide_path, long_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="Path to one RSL-RL run directory.")
    args = parser.parse_args()
    export_tensorboard_scalars(args.log_dir)


if __name__ == "__main__":
    main()
