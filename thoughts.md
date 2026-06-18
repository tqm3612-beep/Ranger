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