from __future__ import annotations

import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "RANGER" / "usd" / "configuration"
BASE_USD = CFG_DIR / "RANGER_base.usd"
PHYSICS_USD = CFG_DIR / "RANGER_physics.usd"
SOURCE_PRIM_PATH_STR = "/RANGER/d435i_depth_optical_frame/visuals"
TARGET_PRIM_PATH_STR = "/visuals/d435i_depth_optical_frame"
BACKUP_USD = PHYSICS_USD.with_name("RANGER_physics.before_d435i_visual_fix.usd")


def main() -> None:
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        # The USD Python bindings are available only after Isaac Sim/Kit starts.
        from pxr import Sdf, Usd

        source_prim_path = Sdf.Path(SOURCE_PRIM_PATH_STR)
        target_prim_path = Sdf.Path(TARGET_PRIM_PATH_STR)

        base_layer = Sdf.Layer.FindOrOpen(str(BASE_USD))
        physics_layer = Sdf.Layer.FindOrOpen(str(PHYSICS_USD))
        if base_layer is None:
            raise RuntimeError(f"Unable to open base USD layer: {BASE_USD}")
        if physics_layer is None:
            raise RuntimeError(f"Unable to open physics USD layer: {PHYSICS_USD}")

        source_spec = base_layer.GetPrimAtPath(source_prim_path)
        if source_spec is None:
            raise RuntimeError(f"Expected source prim is missing from base layer: {source_prim_path}")

        existing_target = physics_layer.GetPrimAtPath(target_prim_path)
        if existing_target is not None:
            print(f"[USD fix] Target prim already exists: {target_prim_path}")
        else:
            if not BACKUP_USD.exists():
                shutil.copy2(PHYSICS_USD, BACKUP_USD)
                print(f"[USD fix] Backup created: {BACKUP_USD}")

            target_spec = Sdf.CreatePrimInLayer(physics_layer, target_prim_path)
            if target_spec is None:
                raise RuntimeError(f"Failed to create target prim: {target_prim_path}")
            target_spec.specifier = Sdf.SpecifierDef
            target_spec.typeName = "Xform"
            physics_layer.Save()
            print(f"[USD fix] Created empty Xform target: {target_prim_path}")

        # Reload both layers after saving so validation does not rely on cached specs.
        base_layer.Reload()
        physics_layer.Reload()
        physics_layer = Sdf.Layer.FindOrOpen(str(PHYSICS_USD))
        if physics_layer is None or physics_layer.GetPrimAtPath(target_prim_path) is None:
            raise RuntimeError(f"USD fix validation failed: target is still missing: {target_prim_path}")

        stage = Usd.Stage.Open(str(BASE_USD))
        if stage is None:
            raise RuntimeError(f"Unable to compose base USD after repair: {BASE_USD}")
        composed_source = stage.GetPrimAtPath(source_prim_path)
        if not composed_source.IsValid():
            raise RuntimeError(f"Composed source prim is invalid after repair: {source_prim_path}")

        print("[USD fix] Validation passed.")
        print(f"[USD fix] Source prim: {source_prim_path}")
        print(f"[USD fix] Reference target: {PHYSICS_USD}@{target_prim_path}")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
