#!/usr/bin/env python3
"""Convert the project-local RANGER URDF to USD with explicit importer settings."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Convert RANGER/urdf/RANGER.urdf to RANGER/usd/RANGER.usd.")
parser.add_argument(
    "--input",
    type=str,
    default="/home/tqm/Isaaclab_projects/Ranger/RANGER/urdf/RANGER.urdf",
    help="Input URDF path.",
)
parser.add_argument(
    "--usd_dir",
    type=str,
    default="/home/tqm/Isaaclab_projects/Ranger/RANGER/usd",
    help="Output USD directory.",
)
parser.add_argument("--usd_file_name", type=str, default="RANGER.usd", help="Output USD file name.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.dict import print_dict


def main() -> None:
    asset_path = Path(args_cli.input).expanduser().resolve()
    usd_dir = Path(args_cli.usd_dir).expanduser().resolve()
    usd_dir.mkdir(parents=True, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=str(asset_path),
        usd_dir=str(usd_dir),
        usd_file_name=args_cli.usd_file_name,
        force_usd_conversion=True,
        make_instanceable=True,
        fix_base=False,
        root_link_name=None,
        link_density=0.0,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="none",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=1.0,
            ),
        ),
        collider_type="convex_hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
    )

    print("[RangerUrdfConvert] config:", flush=True)
    print_dict(cfg.to_dict(), nesting=0)
    converter = UrdfConverter(cfg)
    print(f"[RangerUrdfConvert] generated: {converter.usd_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
