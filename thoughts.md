仿真地形 / terrain mesh
        ↓
传感器安装位姿：front_lidar_link / camera_link
        ↓
视场角限制 FOV
        ↓
距离限制 min_range / max_range
        ↓
遮挡判断 occlusion
        ↓
采样稀疏化 downsample
        ↓
噪声 / 丢点 / 延迟
        ↓
局部高程图 elevation map
        ↓
坡度、粗糙度、台阶、可通行性
        ↓
RL policy


整体流程：

激光雷达 + 相机
        ↓
特征处理 / 语义识别 / 几何分析
        ↓
多层可通行性地图
        ↓
地图信息 + 本体感知信息
        ↓
RL 策略
        ↓
局部路径规划 + 运动控制


local traversability map:
    height_map          高程
    slope_map           坡度
    roughness_map       粗糙度
    step_map            台阶/突变高度
    semantic_map        草、石头、泥、水、树干等语义
    confidence_map      语义置信度
    valid_mask          是否被传感器观测到
    cost_map            综合通行代价


本体感知信息
base angular velocity        车体角速度
projected gravity            车体姿态
linear velocity estimate     车体速度估计
joint positions              关节位置
joint velocities             关节速度
wheel velocities             轮速
motor currents / torques     电机电流或力矩
contact force                轮/足端接触力
previous action              上一步动作
command                      目标速度或目标方向



提出构型条件可通行性地图
提出目标引导的局部策略，同时输出运动和构型目标
仿真和实车一致的传感器受限训练

速度控制 vs 力矩控制


思路一：考虑车身稳定性
可通行性地图
   ↓
路径规划 / 速度控制
   ↓
机器人沿着安全方向走

升级成

可通行性地图
   ↓
判断前方地形高低、坡度、台阶、粗糙度
   ↓
同时控制：
1. 机器人行进速度
2. 机器人转向
3. 四个液压缸行程
4. 车身高度 / 俯仰 / 横滚姿态
   ↓
更稳定地通过复杂地形

输入：
obs =
[
  proprioception,          # 机器人自身状态
  actuator_state,          # 液压腿/轮毂电机状态
  task_goal,               # 目标相对位置
  local_navigation_map,    # 局部几何/可通行/障碍地图
  history                  # last_action 或短历史
]

base_lin_vel              3
base_ang_vel              3
projected_gravity         3
leg_joint_pos_rel          4
leg_joint_vel_rel          4
wheel_joint_vel_rel        4
hydraulic_stroke_state     4
hydraulic_effort_state     4
wheel_velocity_target      4
wheel_torque_state         4
goal_x_body_norm = clamp(goal_x_body / goal_range, -1, 1)
goal_y_body_norm = clamp(goal_y_body / goal_range, -1, 1)
goal_distance_norm = clamp(distance / goal_range, 0, 1)
goal_bearing_sin          1
goal_bearing_cos          1
goal_valid                1
height_map             地形相对高度
slope_map              坡度风险
roughness_map          粗糙度风险
step_map               台阶风险
obstacle_mask          障碍物/不可通行区域
traversability_map     可通行性
        traversability = f(slope, roughness, step, valid_mask)
        cost = w1 * slope + w2 * roughness + w3 * step + w4 * unknown_penalty
traversability = 1 - clamp(cost, 0, 1)
valid_mask             有效观测区域
last_action              8