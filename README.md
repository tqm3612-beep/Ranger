
# Isaac Lab 项目模板

## 概述

本项目/仓库是一个基于 Isaac Lab 构建项目或扩展的模板。
它允许你在 Isaac Lab 核心仓库之外的独立环境中进行开发。

**主要特点：**

* `隔离性`：在 Isaac Lab 核心仓库之外工作，确保你的开发内容保持自包含，不会污染官方源码。
* `灵活性`：该模板可以让你的代码作为 Omniverse 中的扩展运行。

**关键词：** extension、template、isaaclab

---

## 安装

* 按照 [安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
  推荐使用 conda 安装，因为这样可以更方便地从终端调用 Python 脚本。

* 将本项目/仓库克隆或复制到 Isaac Lab 安装目录之外的位置，也就是不要放在 `IsaacLab` 目录里面。

* 使用已经安装 Isaac Lab 的 Python 解释器，以 editable mode 安装本库：

  ```bash
  # 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
  # 请使用 'PATH_TO_isaaclab.sh|bat -p' 代替 'python'
  python -m pip install -e source/Ranger
  ```

---

## 验证扩展是否正确安装

### 1. 列出可用任务

注意：如果任务名称发生变化，可能需要更新 `scripts/list_envs.py` 文件中的搜索模式 `"Template-"`，这样任务才能被列出来。

```bash
# 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
# 请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
python scripts/list_envs.py
```

---

### 2. 运行一个任务

```bash
# 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
# 请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
```

其中：

```text
<RL_LIBRARY> 表示你选择的强化学习库，例如 rsl_rl、rl_games、skrl、sb3
<TASK_NAME> 表示具体任务名称
```

---

### 3. 使用 dummy agents 运行任务

这些 dummy agents 会输出零动作或随机动作。
它们用于检查环境配置是否正确。

#### 零动作 agent

```bash
# 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
# 请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
python scripts/zero_agent.py --task=<TASK_NAME>
```

#### 随机动作 agent

```bash
# 如果 Isaac Lab 没有安装在 Python venv 或 conda 中，
# 请使用 'FULL_PATH_TO_isaaclab.sh|bat -p' 代替 'python'
python scripts/random_agent.py --task=<TASK_NAME>
```

---

## 设置 IDE（可选）

如果要设置 IDE，请按以下步骤操作：

* 运行 VS Code Tasks：按下 `Ctrl+Shift+P`，选择 `Tasks: Run Task`，然后在下拉菜单中运行 `setup_python_env`。

运行该任务时，系统会提示你添加 Isaac Sim 安装路径的绝对路径。

如果一切执行正确，它会在 `.vscode` 目录下创建一个 `.python.env` 文件。
该文件包含 Isaac Sim 和 Omniverse 提供的所有扩展的 Python 路径。
这有助于 VS Code 在你写代码时索引 Python 模块，从而提供智能提示。

---

## 设置为 Omniverse 扩展（可选）

本模板提供了一个示例 UI 扩展。启用你的扩展后，该示例会加载：

```text
source/Ranger/Ranger/ui_extension_example.py
```

要启用你的扩展，请按以下步骤操作：

### 1. 将本项目/仓库的搜索路径添加到扩展管理器

* 通过 `Window` -> `Extensions` 打开扩展管理器。
* 点击 **Hamburger Icon**，也就是三横线菜单，然后进入 `Settings`。
* 在 `Extension Search Paths` 中输入本项目/仓库的 `source` 目录的绝对路径。
* 如果还没有添加 Isaac Lab 的扩展目录，也需要在 `Extension Search Paths` 中添加指向 Isaac Lab `source` 目录的路径，例如：

```text
IsaacLab/source
```

* 点击 **Hamburger Icon**，然后点击 `Refresh`。

---

### 2. 搜索并启用你的扩展

* 在 `Third Party` 分类下找到你的扩展。
* 打开开关以启用该扩展。

---

## 代码格式化

本模板提供了一个 pre-commit 模板，可以自动格式化你的代码。

安装 pre-commit：

```bash
pip install pre-commit
```

然后运行：

```bash
pre-commit run --all-files
```

---

## 故障排查

### Pylance 没有索引扩展

在某些 VS Code 版本中，部分扩展可能无法被正确索引。

这种情况下，可以在 `.vscode/settings.json` 中的 `"python.analysis.extraPaths"` 字段下添加你的扩展路径：

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/Ranger"
    ]
}
```

---

### Pylance 崩溃

如果遇到 `pylance` 崩溃，可能是因为索引的文件过多，导致内存不足。

一种解决方法是排除一些项目中用不到的 Omniverse 包。

具体做法是修改 `.vscode/settings.json`，在 `"python.analysis.extraPaths"` 字段下，注释掉一些不使用的包。

一些可能可以排除的包包括：

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // 动画相关包
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI 工具相关包
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI 工具相关包
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services 工具相关包
...
```
