from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import Ranger.tasks  # noqa: F401
TASK = "Template-Ranger-Debug-v0"
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1, use_fabric=True)
env = gym.make(TASK, cfg=env_cfg)
unwrapped = env.unwrapped
wheel_term = unwrapped.action_manager.get_term("wheel_motor_csv")
robot = unwrapped.scene["robot"]
wheel_ids, _ = robot.find_joints(["w_lb", "w_lf", "w_rf", "w_rb"], preserve_order=True)
sign = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=unwrapped.device)
start = 0
leg_slice = wheel_slice = None
for name, dim in zip(unwrapped.action_manager.active_terms, unwrapped.action_manager.action_term_dim):
    sl = slice(start, start + dim)
    if name == "leg_hydraulic": leg_slice = sl
    if name == "wheel_motor_csv": wheel_slice = sl
    start += dim
assert leg_slice is not None and wheel_slice is not None
cases = {
    "policy_all_plus": torch.tensor([1.0, 1.0, 1.0, 1.0], device=unwrapped.device),
    "policy_all_minus": torch.tensor([-1.0, -1.0, -1.0, -1.0], device=unwrapped.device),
    "policy_phys_forward_sign": torch.tensor([-1.0, -1.0, 1.0, 1.0], device=unwrapped.device),
    "policy_opposite_phys_forward_sign": torch.tensor([1.0, 1.0, -1.0, -1.0], device=unwrapped.device),
}
with torch.inference_mode():
    for name, pattern in cases.items():
        env.reset()
        for _ in range(30):
            a = torch.zeros(env.action_space.shape, device=unwrapped.device); a[:, leg_slice] = -0.34; env.step(a)
        rows = []
        for _ in range(90):
            a = torch.zeros(env.action_space.shape, device=unwrapped.device); a[:, leg_slice] = -0.34; a[:, wheel_slice] = pattern; env.step(a)
            pt = wheel_term.velocity_target[0].clone(); st = pt * sign
            pv = robot.data.joint_vel[0, wheel_ids].clone(); sv = pv * sign
            rows.append(torch.cat((pt, st, pv, sv, robot.data.root_lin_vel_b[0, :1], robot.data.root_ang_vel_b[0, 2:3])))
        m = torch.stack(rows[30:]).mean(dim=0)
        print(f"CASE={name}")
        print("policy", pattern.tolist())
        print("physical_target", m[0:4].tolist())
        print("semantic_target", m[4:8].tolist())
        print("semantic_vel", m[12:16].tolist())
        print("mean_vx", float(m[16].item()), "mean_wz", float(m[17].item()))
env.close(); simulation_app.close()
