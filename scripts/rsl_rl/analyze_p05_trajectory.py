from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def scenario_name(env_id: int) -> str:
    slot = env_id % 20
    if slot < 10:
        return "full"
    if slot < 15:
        return "exit"
    return "stop"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fixed-partition P0.5 trajectory traces.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    episodes: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    with args.csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            episodes[(int(row["env_id"]), int(row["episode_id"]))].append(row)

    summaries: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (env_id, _), rows in episodes.items():
        distances = [float(row["goal_distance_m"]) for row in rows]
        min_index = min(range(len(rows)), key=distances.__getitem__)
        min_distance = distances[min_index]
        rebound = max(distances[min_index:]) - min_distance
        near_rows = [row for row in rows if float(row["goal_distance_m"]) <= 0.5]
        near_speed = (
            sum(abs(float(row["velocity_toward_goal_mps"])) for row in near_rows) / len(near_rows)
            if near_rows
            else 0.0
        )
        summaries[scenario_name(env_id)].append(
            {
                "success": float(any(int(row["stopped_goal_reached_after_step"]) for row in rows)),
                "collision": float(any("obstacle_collision" in row["termination_terms_after_step"] for row in rows)),
                "min_distance": min_distance,
                "rebound": rebound,
                "heading_at_min": abs(float(rows[min_index]["heading_error_rad"])),
                "near_speed": near_speed,
            }
        )

    fields = ("success", "collision", "min_distance", "rebound", "heading_at_min", "near_speed")
    print("scenario\tepisodes\t" + "\t".join(fields))
    for scenario in ("full", "exit", "stop"):
        values = summaries[scenario]
        means = [sum(item[field] for item in values) / len(values) if values else 0.0 for field in fields]
        print(f"{scenario}\t{len(values)}\t" + "\t".join(f"{value:.6f}" for value in means))


if __name__ == "__main__":
    main()
