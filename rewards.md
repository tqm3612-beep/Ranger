Stand_Input:
   base_lin_vel，3 维，base frame 线速度，按 2.0 归一化。
   base_ang_vel，3 维，base frame 角速度，按 3.0 归一化。
   projected_gravity，3 维。
   leg_joint_pos_rel，4 维，g_* 相对默认角度。
   leg_joint_vel_rel，4 维，g_* 相对默认角速度。
   wheel_joint_vel_rel，4 维，w_* 相对默认角速度。
   hydraulic_stroke_state，4 维，当前是 stroke_command 的归一化值。
   hydraulic_effort_state，4 维。
   当前代码里 leg_hydraulic 是 position-target 模式，effort_actual 只是兼容占位，所以这 4 维现在实际上是 0 附近的兼容量
   wheel_velocity_target，4 维。
   wheel_torque_state，4 维。
   goal_state，6 维。
   在 Stand 里 goal_enabled=False，所以返回中性占位 [0, 0, 0, 0, 1, 0]
   last_action，8 维。
   local_navigation_map，6 x 21 x 13 = 1638 维。

Stand_Rewards:
   alive = 0.2
   forward_progress = 0.0
   velocity_tracking = 0.0
   overspeed = 0.0
   upright = -3.0
   base_height_low = 0.0
   base_vertical_velocity = -1.0
   base_roll_pitch_rate = -0.5
   lateral_velocity = -0.2
   leg_joint_deviation = -0.02
   wheel_joint_velocity = -0.002
   leg_joint_velocity = -0.005
   oint_limit_margin = 0.0
   wheel_semantic_velocity_symmetry = 0.0
   action_rate = -0.02
   action_magnitude = -0.002
   termination = -10.0
   wheel_contact_count = 2.0
   wheel_contact_balance = 1.0

Forward_Input:
   base_lin_vel，3 维，base frame 线速度，按 2.0 归一化。
   base_ang_vel，3 维，base frame 角速度，按 3.0 归一化。
   projected_gravity，3 维。
   leg_joint_pos_rel，4 维，g_* 相对默认角度。
   leg_joint_vel_rel，4 维，g_* 相对默认角速度。
   wheel_joint_vel_rel，4 维，w_* 相对默认角速度。
   hydraulic_stroke_state，4 维，当前是 stroke_command 的归一化值。
   hydraulic_effort_state，4 维。
   当前代码里 leg_hydraulic 是 position-target 模式，effort_actual 只是兼容占位，所以这 4 维现在实际上是 0 附近的兼容量
   wheel_velocity_target，4 维。
   wheel_torque_state，4 维。
   goal_state，6 维。
      goal_enabled=True
      goal_x_body=3.0
      goal_y_body=0.0
      use_neutral_map=True
      这个 6 维大致对应 x=3/5=0.6, y=0, distance=0.6, sin=0, cos=1, valid=1
   last_action，8 维。
   local_navigation_map，6 x 21 x 13 = 1638 维。

Forward_Rewards:
   alive = 0.2
   forward_progress = 2.0
   velocity_tracking = 0.0
   overspeed = 0.0
   upright = -3.0
   base_height_low = 0.0
   base_vertical_velocity = -1.0
   base_roll_pitch_rate = -0.5
   lateral_velocity = -0.2
   leg_joint_deviation = -0.02
   wheel_joint_velocity = 0.0
   leg_joint_velocity = -0.005
   joint_limit_margin = 0.0
   wheel_semantic_velocity_symmetry = 0.0
   action_rate = -0.02
   action_magnitude = -0.002
   termination = -10.0
   wheel_contact_count = 1.0
   wheel_contact_balance = 0.5