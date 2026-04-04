# AutoResearch System Design

## 1. Overview

AutoResearch 是 SPML 的自动化实验迭代系统。定义研究协议（protocol），驱动自主循环：Researcher 修改代码并训练，Supervisor 评测、审查合规、管理 git 和调度。

## 2. End-to-End Flow

```
Phase 1: Protocol Definition（复用 SPML 主流程）
  autoresearch-create → ml-brainstorming → experiment-planning → ml-subagent-dev + VP

Phase 2: Handoff
  ml-subagent-dev Post-Completion Gate → 用户选 "Research" → autoresearch-handoff

Phase 3: Autonomous Loop（新 session）
  autoresearch-run → autoresearch（Supervisor 主循环）→ Final Report
```

**路由信号：** design doc 中的 `## Autoresearch Protocol` section → Post-Completion Gate "Research" 选项 → `autoresearch-protocol.md` + `run autoresearch at <dir>`

## 3. Protocol Definition

复用 SPML 主流程（brainstorming → planning → subagent-dev）构建和验证 baseline。这些 skill 不感知 autoresearch。

autoresearch 的特殊性：brainstorming 检测到意图后额外收集协议问题，design doc 多一个 `## Autoresearch Protocol` section。

**协议问题（逐一询问）：**

| # | 问题 | 对应 protocol 字段 |
|---|------|-------------------|
| 0 | Experiment directory | （路径，不写入 protocol） |
| 1 | Research question | `research_question` |
| 2 | Fixed（不可变的代码 + 条件） | `Fixed.files` + `time_limit` + `epoch_limit` |
| 3 | Variable（可变的代码 + 条件） | `Variable.files` + 可调范围 |
| 4 | Evaluation | `Eval.metric` + `direction` + `command` |
| 5 | Termination | `max_rounds` + `target` |
| 6 | Initial hints（可选） | experiences.md R0 Note |

`train_command` 和 `eval_command` 不问用户 — 在 build 阶段（VP L1）已确定，handoff 从 baseline 代码中提取写入 protocol。

**检测触发：** `autoresearch-create` 已调用 / 关键词匹配 / 模式匹配（搜索优化而非验证假设）。

## 4. Baseline Validation & Handoff

**VP L1 强制不可跳过。** Autoresearch 需要已验证能跑通的 baseline + baseline 指标 + 压力条件终止逻辑验证。跳过 L1 = 循环从 Round 1 卡住。

Handoff 条件：VP L1 passed + design doc 含 `## Autoresearch Protocol` section。

Handoff 产出：`autoresearch-protocol.md` + `experiences.md`（含 baseline 指标）。

## 5. Supervisor Loop

### 5.1 架构

```
Supervisor（主 session）
  ├── Researcher（后台 subagent）— 修改可变代码 + 训练
  ├── 评测 — Supervisor 自己跑 eval_command
  ├── 合规审查 — Supervisor 自己 git diff --name-only 检查
  └── Git — commit on improve, rollback on failure
```

**默认模式（Supervisor + Researcher）：** Supervisor 只派一个 subagent（Researcher）做创造性工作。评测和合规审查由 Supervisor 自己完成 — eval_command 是一条 bash 命令，合规检查是一条 `git diff --name-only`，不需要单独的 Agent。

**自定义 pipeline 模式：** 用户可在 protocol 中定义多个 Agent 角色（如拆分 Researcher / Reviewer / Evaluator）。Supervisor 按 pipeline 顺序调度。默认不启用。

### 5.2 Worktree 隔离

所有操作在 `git worktree add ../autoresearch-{name} HEAD` 创建的隔离 worktree 中。恢复时复用已有 worktree。结束时用户选择 merge / 保留 / 删除。

### 5.3 代码接口化

Brainstorming 阶段划分框架代码（Fixed.files）和可变代码（Variable.files）。划分写入 protocol。

- 可变代码是 Researcher 唯一修改点（取决于研究目标）
- 框架代码 Researcher 不碰（数据加载、评测、入口脚本、终止逻辑）
- 训练脚本（框架层）自带超时控制：在 time_limit 到达前触发 checkpoint 保存，然后正常退出。Bash timeout 只是兜底。
- Baseline 进入循环前必须跑通完整 pipeline（VP L1 保证）
- Supervisor 读 protocol 一次，提取信息注入 Researcher prompt。**Researcher 不读 protocol 文件。**

### 5.4 Per-Round 流程

```
每轮：
  0. Supervisor 创建 Task List（6 项）
  1. Researcher（subagent）设计策略 + 改代码（不跑训练）
  2. Supervisor 合规审查：git diff --name-only
  3. Supervisor 跑训练：Bash(train_command)  ← stdout 对用户可见
  4. Supervisor 跑评测：Bash(eval_command)
  5. Supervisor 判定 + git commit / rollback
  6. Check termination → 下一轮
```

**Researcher 职责（设计 + 编码，不跑训练）：**
- 只改 protocol 指定的可变文件
- 先写 strategy（experiences.md 当前 round 的 strategy 列），再改代码
- 不跑训练、不跑评测、不碰 Fixed

**Supervisor 职责（审查 + 执行 + 评测）：**
- 合规：`git diff --name-only`，碰固定层 → 直接 not_improved，跳过训练和评测
- 训练：`Bash(train_command)`，用户可直接看到 stdout
- 评测：`Bash(eval_command)`，客观指标
- 记录：更新 experiences.md 表格

### 5.5 Git 管理

只有 Supervisor 执行 git 写操作。Improved → commit；Not improved → rollback 代码 + 保留 experiences.md。

### 5.6 终止逻辑

**仅以下条件终止，无例外，无主观判断：**
- `target` 已设且达到 → `target_reached`
- `round == max_rounds` → `completed`
- 其他 → 继续

Supervisor 不得判断"已达理论上限"。协议说跑多少轮就跑多少轮。

### 5.7 Per-Round Task List

**每轮必须以 Task List 开始（包括异常恢复后）。** 防止跳步。

```
R{N}: Researcher        — design + code
R{N}: Compliance check
R{N}: Train              — stdout visible to user
R{N}: Evaluation
R{N}: Git
R{N}: Termination check
```

## 6. Supervisor Liveness

Supervisor 是语言模型，不是持久进程。三层机制保障循环持续运行：

| 层 | 触发条件 | 作用 |
|----|---------|------|
| Agent 通知 | Researcher 完成 | 正常推进（合规→评测→git→下一轮） |
| Per-round 超时计时器 | Researcher 超时（one-shot CronCreate, `time_limit * 2`） | 终止该轮，反思，下一轮 Step 0 |
| Session heartbeat | 循环断了（recurring CronCreate */30） | 提醒 Supervisor 恢复循环 |

```
正常：Researcher 完成 → 通知 → 取消计时器 → Supervisor 评测 → 下一轮
卡住：超时 → 计时器触发 → 终止该轮 → 反思 → 下一轮
异常：循环断了 → 30min heartbeat → 恢复
```

- Heartbeat：session-scoped，REPL idle 时才触发，不积压。
- **可选 sleep-check 汇报模式**：protocol 中设 `sleep_check: true` 开启。默认关闭。

**循环自治：** 每轮结束后立即进入下一轮，不等用户输入。

## 7. Anomaly Recovery

- **Researcher 超时/崩溃：** 记录到 experiences.md，rollback，重试一次。再失败则跳过该轮继续。恢复后回到 Step 0。
- **Session 中断：** 默认续跑。从 experiences.md 恢复状态（running → 继续；completed/target_reached → 询问用户是否追加）。git HEAD 一致性检查。
- **连续失败：** 5 轮连续 not_improved → plateau 警告，继续运行。

## 8. Artifacts

### 8.1 autoresearch-protocol.md

Supervisor 读取一次，提取信息注入 Researcher prompt。Researcher 不读此文件。

```markdown
# Autoresearch Protocol: <title>

research_question: <一句话>
max_rounds: 10
target: none
baseline: metric = value

## Fixed（不可变：代码 + 条件）
- files: data.py, eval.py, run.sh
- time_limit: 5min
- epoch_limit: 1

## Variable（可变：代码 + 条件）
- files: train.py
- 可调范围: lr, optimizer, loss, augmentation...

## Eval
- metric: accuracy
- direction: maximize
- command: python eval.py --checkpoint outputs/best.pt
```

### 8.2 experiences.md

表格格式，每轮两行（strategy + result），保持紧凑。Supervisor 负责维护，Researcher 只读。

```markdown
# Experiences

best: accuracy = 0.82 (R3)
rounds: 5 / 10
status: running

| Round | Strategy | Compliance | Result | Verdict | Insight | Note |
|-------|----------|------------|--------|---------|---------|------|
| 0 | baseline | ✅ | 0.31 | — | initial | lr>1e-3 会爆 |
| 1 | lr 1e-3→3e-4, cosine schedule | ✅ | 0.55 | ✅ commit | lr decay helped | |
| 2 | add mixup augmentation | ✅ | 0.49 | ❌ rollback | mixup hurt convergence | 用户: 试试只在前半段 mixup |
| 3 | early-stage mixup + AdamW | ✅ | 0.82 | ✅ commit | partial mixup + regularization | |
| 4 | increase model width 2x | ❌ fixed | — | ❌ rollback | touched model.py (fixed) | |
| 5 | label smoothing 0.1 | ✅ | 0.78 | ❌ rollback | no improvement | 用户: 参考 paper X 的方法 |
```

**Note 列：** 用户随时可以插入指导，Supervisor 写到当前轮的 Note 列。实验开始前的先验知识放在 R0 的 Note。Researcher 读表格时自然看到。

**窗口化：** Researcher 看到的 prompt 只包含 Summary + 最近 N 轮（默认 N=5）。旧轮的关键洞察由 Supervisor 浓缩进一句话追加到 Summary 下方。全量表格始终保留在文件中，但不注入 prompt。

## 9. Feature Checklist

### Protocol & Handoff
- [ ] Brainstorming 检测 autoresearch 意图 → 协议问题流程
- [ ] 划分 Fixed.files（框架）和 Variable.files（可变）
- [ ] VP L1 强制不可跳过
- [ ] 生成精简 protocol + 初始化 experiences.md

### Code Interface
- [ ] Supervisor 读 protocol 一次，注入 Researcher prompt
- [ ] Researcher 不读 protocol 文件，不读框架代码
- [ ] Baseline 进入循环前已跑通（VP L1）

### Per-Round Flow
- [ ] 每轮以 6 项 Task List 开始，清空上轮 tasks
- [ ] Researcher 以 run_in_background 派发（设计 + 改代码，不跑训练）
- [ ] Supervisor 合规审查：git diff --name-only
- [ ] Supervisor 跑训练：Bash(train_command)，stdout 对用户可见
- [ ] Supervisor 跑评测：Bash(eval_command)
- [ ] 循环自治：轮间不等用户输入

### Researcher
- [ ] 只改可变文件，先写 strategy 到 experiences.md，再改代码
- [ ] 不跑训练、不跑评测、不碰 Fixed

### Git & Termination
- [ ] 只有 Supervisor 执行 git 写操作，在 worktree 中
- [ ] Improved → commit，Not improved → rollback + 保留 experiences.md
- [ ] 仅 target_reached 或 max_rounds 终止，无主观判断
- [ ] 碰固定层 → 直接 not_improved，跳过训练和评测

### Liveness
- [ ] Researcher 通知 → Supervisor 评测 → 正常推进
- [ ] Per-round 超时计时器 → 终止该轮 → 反思 → 下一轮 Step 0
- [ ] Session heartbeat */30 → 恢复断裂的循环
- [ ] Sleep-check 汇报模式（可选，默认关闭）

### Recovery
- [ ] Researcher 崩溃：记录 + rollback + 重试一次
- [ ] Session 中断：默认续跑，不清理
- [ ] 5 轮连续失败：plateau 警告，继续运行

### experiences.md
- [ ] 表格格式，每轮一行，含 Note 列
- [ ] Supervisor 维护，Researcher 只读
- [ ] 窗口化：prompt 只注入 Summary + 最近 N 轮
- [ ] 用户先验知识写入 R0 的 Note
- [ ] 循环中用户输入追加到当前轮 Note，不暂停循环
