#!/usr/bin/env python3
"""Batch deterministic/diagnostic characterization helper for Ranger checkpoints.

This is an experiment utility only. It invokes the existing play.py/train.py entry points
without modifying environment, reward, policy, PPO, or checkpoint contents.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/tqm/miniconda3/envs/isaaclab/bin/python"


def run_and_log(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}); see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--load-run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=("eval", "diagnostic"), required=True)
    parser.add_argument("--lam", type=float, default=None, help="Optional Teacher-PPO GAE lambda override for diagnostics.")
    args = parser.parse_args()

    out_dir = ROOT / "logs" / "analysis" / "ppo_teacher_characterization" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        if args.mode == "eval":
            command = [
                PYTHON,
                "scripts/rsl_rl/play.py",
                "--task",
                "Template-Ranger-ShortGoalFlat-C-Recurrent",
                "--num_envs",
                "64",
                "--seed",
                str(seed),
                "--max_steps",
                "1400",
                "--evaluation_summary",
                "--evaluation_metrics_csv",
                str(out_dir / f"eval_seed{seed}.csv"),
                "--checkpoint",
                args.checkpoint,
                "--headless",
            ]
            run_and_log(command, out_dir / f"eval_seed{seed}.log")
        else:
            command = [
                PYTHON,
                "scripts/rsl_rl/train.py",
                "--task",
                "Template-Ranger-ShortGoalFlat-C-Recurrent-TeacherPPO",
                "--num_envs",
                "64",
                "--seed",
                str(seed),
                "--max_iterations",
                "1",
                "--resume",
                "--load_run",
                args.load_run,
                "--checkpoint",
                Path(args.checkpoint).name,
                "--teacher_loss_coef",
                "0",
                "--teacher_ppo_diagnostic_only",
                "--teacher_ppo_diagnostic_rollout_steps",
                "192",
            ]
            if args.lam is not None:
                command.extend(("--teacher_ppo_lam", str(args.lam)))
            command.append("--headless")
            run_and_log(command, out_dir / f"diag_seed{seed}.log")


if __name__ == "__main__":
    main()
