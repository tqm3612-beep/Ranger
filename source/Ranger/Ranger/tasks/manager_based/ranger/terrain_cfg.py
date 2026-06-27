import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainImporterCfg


SIMPLE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=terrain_gen.TerrainGeneratorCfg(
        seed=23,
        curriculum=True,
        size=(8.0, 8.0),
        border_width=0.0,
        num_rows=4,
        num_cols=4,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=None,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={
            "gentle_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.34,
                slope_range=(0.052, 0.192),
                platform_width=1.4,
                border_width=0.0,
            ),
            "mild_undulation": terrain_gen.HfWaveTerrainCfg(
                proportion=0.33,
                amplitude_range=(0.0075, 0.0275),
                num_waves=3,
                border_width=0.0,
            ),
            "low_steps": terrain_gen.HfPyramidStairsTerrainCfg(
                proportion=0.33,
                step_height_range=(0.03, 0.11),
                step_width=0.3,
                platform_width=1.4,
                border_width=0.0,
            ),
        },
    ),
    max_init_terrain_level=0,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
    debug_vis=False,
)
