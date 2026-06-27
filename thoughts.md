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



可以分阶段，而且我建议你**必须分阶段**。
你的任务不应该一开始就训练：

```text
局部地图输入 → 判断地形/障碍 → 规划方向 → 控制轮子和液压腿 → 稳定通过
```

这会把太多问题混在一起，训练失败时很难判断到底是：

```text
模型站不住
轮子不转
液压腿动作映射错
奖励不合理
地形太难
地图输入异常
策略网络不够
```

更合理的路线是按能力逐层增加。

---

# 总体分阶段路线

我建议按这个顺序：

```text
阶段 0：物理模型和执行器基线验证
阶段 1：静态支撑 / 姿态恢复
阶段 2：平地平稳前进
阶段 3：简单地形通过
阶段 4：复杂地形上的速度跟踪和姿态稳定
阶段 5：向目标点靠近
阶段 6：障碍规避 / 可通行性地图输入
阶段 7：Sim2Real 随机化与实车部署适配
```

对你现在的情况来说，**阶段 0 和阶段 1 最重要**。
你现在机器人“原地下陷到限位”，说明还没有真正通过阶段 0/1。

---

# 阶段 0：物理模型和执行器基线验证

这个阶段不训练 PPO，只做手动测试。

目标是确认：

```text
机器人能站住
液压腿 action=0 对应默认支撑姿态
轮子固定速度时能向前走
不会原地下陷
不会一落地弹飞
不会某个关节打限位
```

## 0.1 零动作站立测试

动作全部为 0：

```text
leg_hydraulic action = 0
wheel_motor_csv action = 0
```

观察 5 秒。

合格标准：

```text
base 高度基本不变
roll/pitch 不明显发散
四个轮子接触正常
腿关节不慢慢压到限位
不会触发 root_height_low
```

如果这个测试失败，后面所有训练都没有意义。

你现在的问题大概率就在这里。

---

## 0.2 固定腿姿态测试

把液压腿固定到默认支撑角：

```text
g_lb = default
g_lf = default
g_rf = default
g_rb = default
```

轮子不动。

合格标准：

```text
机器人可以稳定站立
悬架不会慢慢下沉
```

如果固定腿姿态可以站住，但 PPO 控制时会下陷，说明问题在：

```text
leg_hydraulic action 映射
action=0 是否等于 default
液压腿 target 是否被策略写坏
```

重点检查是不是写成了：

```python
target = action * scale
```

更合理的是：

```python
target = default_target + action * scale
```

---

## 0.3 固定轮速前进测试

液压腿固定，给四个轮子固定速度。

例如：

```text
leg target = default
wheel target = constant velocity
```

合格标准：

```text
机器人能明显向前移动
不会原地空转
不会车身下陷
不会一加速就翻
```

如果轮子转但车不走，查：

```text
轮子 joint axis
轮子速度方向
轮子 collision
地面摩擦
wheel_motor_csv 是否真正作用到 joint
```

---

# 阶段 1：静态支撑 / 姿态恢复训练

这一阶段可以训练，但不要让机器人前进。

## 目标

让机器人学会：

```text
在平地和轻微不平地面上保持车身高度
保持 roll/pitch 稳定
受到小扰动后快速恢复
腿部动作平滑
不打限位
```

这个阶段可以理解为主动悬架的“稳定站立控制”。

不过对你的平台来说，不一定叫“直立”，更准确是：

```text
保持车身姿态稳定 + 保持轮地接触 + 保持合理离地高度
```

---

## 动作设置

建议先只训练液压腿：

```text
action dim = 4
只控制 g_lb, g_lf, g_rf, g_rb
轮子速度目标固定为 0
```

或者：

```text
轮子 joint 仍然存在，但 wheel target = 0
```

这样可以先让策略专注于悬架稳定。

---

## 观测输入

这个阶段不需要复杂地图。

可以用：

```text
base_lin_vel
base_ang_vel
projected_gravity
root_height
leg_joint_pos
leg_joint_vel
hydraulic_stroke_state
last_action
```

如果有地形扰动，可以加入简单 height scan，但不要一开始用复杂多通道地图。

---

## 奖励项

建议包括：

```text
车身高度奖励
roll/pitch 姿态奖励
竖直速度惩罚
roll/pitch 角速度惩罚
腿关节偏离默认惩罚
腿关节速度惩罚
action_rate 惩罚
关节限位惩罚
存活奖励
失败终止惩罚
```

示意：

```python
reward = (
    + w_alive
    - w_height * (base_z - target_z)^2
    - w_orientation * (roll^2 + pitch^2)
    - w_vertical_vel * vz^2
    - w_ang_vel * (wx^2 + wy^2)
    - w_joint_dev * ||q_leg - q_default||^2
    - w_joint_vel * ||dq_leg||^2
    - w_action_rate * ||a_t - a_t-1||^2
)
```

---

## 扰动设置

一开始不要太难。

可以逐步加入：

```text
轻微随机初始 roll/pitch
轻微随机 base 高度
轻微随机关节角
小外力推一下车体
低矮坡面
小高度差地形
```

不要一开始就碎石、台阶、障碍物一起上。

---

## 合格标准

这一阶段通过的标准可以设为：

```text
episode length 基本达到最大值
bad_orientation ≈ 0
root_height_low ≈ 0
base_vertical_velocity 明显下降
roll/pitch rate 明显下降
action_rate 不大
腿关节不打限位
```

从视频上看应该是：

```text
车体不下陷
不弹跳
不明显抖腿
受到轻微扰动后能恢复
```

---

# 阶段 2：平地平稳前进

阶段 1 通过后，再开始让机器人向前走。

## 推荐做法

不要一开始就让液压腿和轮子都完全自由。

可以分两步。

---

## 2.1 固定液压腿，只训练轮子

先把液压腿固定在默认支撑姿态：

```text
leg target = default
policy 只控制 wheel velocity
```

目标是验证：

```text
轮子控制 + forward reward 是有效的
```

这一步如果都学不会前进，就不要开放液压腿。

---

## 2.2 开放液压腿小范围调节

等轮子能带动车前进后，再开放液压腿，但动作范围要小：

```text
leg target = default + action * small_scale
```

比如只允许默认姿态附近微调。

目的不是让腿大幅运动，而是让它辅助：

```text
稳定车身
适应轻微地面起伏
减少俯仰/横滚
```

---

## 奖励项

平地前进阶段的核心奖励是：

```text
前进速度 / 位移奖励
姿态稳定奖励
高度稳定奖励
动作平滑惩罚
竖直抖动惩罚
侧向速度惩罚
偏航角速度惩罚
```

如果你是给定目标速度，可以用速度跟踪：

```python
reward_vel = exp(- (v_x - v_cmd)^2 / sigma)
```

如果只是希望向前走，可以用：

```python
reward_forward = v_x
```

但要注意，前进奖励不能太弱。
你之前日志里 `forward_progress` 明显小于 `base_vertical_velocity` 和 `action_rate`，所以机器人容易选择原地保守策略。

---

## 合格标准

```text
机器人能持续向前走
episode length 达到最大
forward_progress 明显上升
base_vertical_velocity 不大
action_rate 不大
upright 不恶化
不会边跳边走
不会轮子空转
```

视频上应该是：

```text
车身平稳
轮子持续转动
轨迹基本直
没有明显点头/弹跳
```

---

# 阶段 3：简单地形通过

平地前进稳定后，再加入地形。

地形难度建议逐步增加：

```text
平地
轻微坡面
缓慢起伏地形
低高度随机 rough terrain
小台阶
碎石/坑洼
```

不要一上来就给复杂越野地形。

---

## 训练方式

这一阶段可以继续让目标是：

```text
保持给定前进速度
保持姿态稳定
不触发失败终止
```

暂时不需要目标导航，不需要障碍规避。

---

## 地形 curriculum

可以按难度系数逐渐增加：

```text
terrain_level = 0: 平地
terrain_level = 1: 小坡度
terrain_level = 2: 小起伏
terrain_level = 3: 轻微坑洼
terrain_level = 4: 小台阶
terrain_level = 5: 混合地形
```

当机器人在当前难度下达到：

```text
episode length 高
forward_progress 高
failure rate 低
```

再提高地形难度。

---

## 奖励项增加

可以加入：

```text
坡面上保持姿态
通过地形时减少竖直冲击
轮子接触稳定
避免腿关节限位
```

对你这个主动悬架平台，重点不是让它像腿足机器人一样迈步，而是让它：

```text
根据地形调整车身姿态和轮地接触
```

---

# 阶段 4：复杂地形上的速度跟踪和姿态稳定

这一阶段是阶段 3 的加强版。

目标从“能往前走”变成：

```text
给定速度指令 v_cmd
机器人在不同地形上稳定跟踪速度
并保持车身姿态
```

观测可以开始加入局部几何地图：

```text
height
slope
step
roughness
valid_mask
```

但是建议先用仿真里生成的 GT/degraded map，而不是直接上真实传感器模拟。

原因是直接模拟激光雷达和深度相机会增加训练复杂度和计算开销。
第一版更轻量的路线是：

```text
仿真 GT 地形 → 加噪声/遮挡/降采样 → 局部几何地图 → policy
```

这样比直接把两台 LiDAR + 深度相机全放进训练环境更稳定，也更省算力。

---

# 阶段 5：向目标点靠近

等机器人已经能稳定速度跟踪后，再训练“向目标靠近”。

这一步不要直接让策略做全局路径规划。
更建议把目标转换成局部命令：

```text
目标点相对机器人位置 → 期望速度 v_cmd 和角速度 yaw_rate_cmd
```

然后策略负责底层运动：

```text
输入：局部目标方向 + 当前状态 + 局部地形
输出：轮速 + 悬架调节
```

也就是说，policy 不一定直接学“从当前位置规划到目标点”，而是学：

```text
朝目标方向稳定行驶
```

这和你之前对系统的设想是一致的：

```text
知道目标相对位置
朝靠近目标的大方向行进
不要求最短路径
最终安全到达
```

---

## 目标靠近奖励

可以包括：

```text
距离目标减少奖励
朝向目标奖励
速度方向与目标方向一致奖励
到达奖励
越界/碰撞惩罚
姿态稳定惩罚
动作平滑惩罚
```

例如：

```python
reward_progress_to_goal = previous_distance_to_goal - current_distance_to_goal
```

这个比单纯奖励 `v_x` 更适合目标导航。

---

## 合格标准

```text
机器人能朝目标方向移动
距离目标持续减小
不会因为追目标而翻车
不会严重绕圈
不会原地打转
```

---

# 阶段 6：障碍规避 / 可通行性地图输入

这是后期阶段，不建议太早做。

你可以有两种路线。

---

## 路线 A：局部规划器 + RL 控制器

这是更稳妥、更适合工程落地的方案。

结构是：

```text
局部几何地图 / 可通行性地图
        ↓
局部规划器生成局部目标或期望速度
        ↓
RL policy 负责稳定跟踪和悬架控制
```

优点：

```text
训练难度较低
安全性更好解释
调试更清楚
实车部署更稳
```

对你的论文也更容易讲清楚：

```text
可通行性地图负责环境理解
局部规划器负责方向选择
RL 负责复杂地形下的运动控制
```

---

## 路线 B：RL 直接输入地图并输出动作

结构是：

```text
局部几何/可通行性地图 + 目标方向
        ↓
policy
        ↓
轮速 + 液压腿动作
```

优点是端到端程度更高。
但缺点是：

```text
训练样本需求更大
奖励更难调
容易学到绕障失败或钻空子
对地图质量敏感
算力消耗更高
可解释性较弱
```

如果你要做论文创新，可以做这条路线，但建议不要一开始就这样训练。

更稳妥的是先用路线 A 做通，再逐步把局部规划能力并入策略。

---

# 阶段 7：Sim2Real 随机化与实车部署

最后才做 Sim2Real。

随机化内容包括：

```text
质量
惯量
摩擦系数
关节阻尼
执行器延迟
传感器噪声
地图遮挡
高度图误差
轮子半径误差
地面刚度
控制频率扰动
```

但是不要太早加入强随机化。
如果基础行为还没学会，强随机化只会让训练更难收敛。

建议顺序是：

```text
先无随机化学会
再轻随机化
再中等随机化
最后接近实车噪声
```

---

# 推荐给你当前项目的具体阶段表

## 阶段 0：站立基线

```text
训练：不训练
控制：固定腿 + 固定轮速 0
地形：平地
目标：验证不下陷、不打限位
```

通过后再进入下一阶段。

---

## 阶段 1：主动悬架稳定

```text
训练：PPO
动作：只控制 4 个腿部/液压缸
轮子：锁定 0 速度
地形：平地 + 轻微扰动
目标：保持 base 高度和姿态
```

---

## 阶段 2：平地前进

```text
训练：PPO
动作：先只控制 4 个轮子，再开放小范围腿部控制
地形：平地
目标：稳定前进，减少抖动
```

---

## 阶段 3：简单地形前进

```text
训练：从阶段 2 checkpoint 继续
动作：轮子 + 小范围腿部
地形：坡面、低起伏
目标：保持速度和车身姿态
```

---

## 阶段 4：局部地形感知

```text
训练：从阶段 3 checkpoint 继续
输入：height/slope/step/roughness/valid_mask
地形：随机地形
目标：根据地形调节悬架和速度
```

---

## 阶段 5：目标靠近

```text
训练：从阶段 4 checkpoint 继续
输入：目标相对位置 + 局部地图
输出：轮速 + 悬架动作
目标：朝目标靠近
```

---

## 阶段 6：障碍规避

```text
训练：从阶段 5 checkpoint 继续
输入：局部可通行性地图 / 障碍地图
目标：靠近目标，同时避开不可通行区域
```

---

# 很重要的一点：每阶段不要保留过强的旧奖励

分阶段训练时有一个常见坑：

> 第一阶段为了站稳，给了很强的姿态、动作、关节偏离惩罚；第二阶段想让它前进，但这些惩罚太强，机器人就不敢动。

所以阶段 2 开始后，需要重新平衡奖励。

例如阶段 1 可以比较重视：

```text
height
upright
action_rate
joint_deviation
```

但阶段 2 必须提高：

```text
forward_progress / velocity_tracking
```

否则机器人容易学成：

```text
站得很稳，但不愿意走
```

这和你之前训练日志里看到的情况很像。

---

# 对你当前状态的建议

你现在不要直接进入“目标靠近”或“障碍规避”。

你当前最应该做的是：

```text
阶段 0：零动作站立
阶段 0：固定轮速前进
阶段 1：主动悬架稳定
阶段 2：平地前进
```

尤其是现在机器人会“原地下陷到限位”，这说明：

```text
阶段 0 还没完全通过
```

在它不能稳定站住之前，继续训练复杂策略会非常低效。

---

# 最终建议

你的分阶段路线可以这样定：

```text
先解决能不能站住；
再解决能不能平稳走；
再解决能不能在不同地形上走；
再解决能不能朝目标走；
最后解决能不能基于地图避障。
```

对你这个项目，最推荐的训练路线是：

```text
1. 零动作稳定站立
2. 固定轮速能前进
3. PPO 学主动悬架姿态恢复
4. PPO 学平地速度跟踪
5. PPO 学简单地形速度跟踪
6. 加入局部几何地图
7. 加入目标方向
8. 加入障碍和可通行性地图
```

一句话概括：

> **可以分阶段，而且应该分阶段。先把“站稳”和“走稳”训练成可靠底层能力，再在这个基础上叠加目标导航和障碍规避。不要一开始就把感知、规划、控制全部交给一个策略同时学习。**
