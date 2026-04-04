# AutoResearch System Design

## 1. Overview

AutoResearch 是 SPML 的自动化实验迭代系统。定义一个研究协议（protocol），驱动自主循环：Agent A 设计策略并修改代码，Agent B 独立评测并审查合规，Supervisor 管理 git 状态、调度和恢复。

目标：自动化研究员手动执行的"尝试 → 评估 → 学习 → 再试"循环，同时保证公平性（压力条件）、可追溯性（git + experiences.md）和隔离性（worktree）。

## 2. End-to-End Flow

```
[Phase 1: Protocol Definition — 复用 SPML 主流程]
  spml:autoresearch-create
    → spml:ml-brainstorming（autoresearch 模式，8 个协议问题）
      → spml:experiment-planning（不感知 autoresearch，正常生成计划）
        → spml:ml-subagent-dev（不感知 autoresearch，正常执行 + VP 验证）

[Phase 2: Handoff — autoresearch 专有]
  ml-subagent-dev Post-Completion Gate 检测 "## Autoresearch Protocol" section
    → 用户选择 "Research"
      → spml:autoresearch-handoff（验证 VP、生成协议文件）

[Phase 3: Autonomous Loop — autoresearch 专有]
  新 session: spml:autoresearch-run
    → spml:autoresearch（Supervisor 主循环）
      → Round 1..N: Agent A → Agent B → git commit/rollback
    → Final Report
```

**路由信号：**
- brainstorming → ml-subagent-dev：design doc 中的 `## Autoresearch Protocol` section
- ml-subagent-dev → handoff：用户在 Post-Completion Gate 选择 "Research"
- handoff → autoresearch：`autoresearch-protocol.md` 文件 + `run autoresearch at <dir>` 指令

## 3. Protocol Definition（复用 SPML 主流程）

### 3.1 与主流程的关系

AutoResearch 复用 SPML 的完整工作流（brainstorming → experiment-planning → ml-subagent-dev）来构建和验证 baseline 代码。这些 skill 不感知 autoresearch — 它们正常执行设计、规划、实现和 VP 验证。

autoresearch 的特殊性仅体现在：
- **ml-brainstorming** 检测到 autoresearch 意图后，额外收集 8 个协议问题
- **design doc** 包含额外的 `## Autoresearch Protocol` section
- **ml-subagent-dev** 的 Post-Completion Gate 检测该 section 并提供 "Research" 选项

### 3.2 协议定义问题（8 个，逐一询问）

| # | 问题 | 目的 |
|---|------|------|
| 0 | Experiment directory | 确定实验目录（已有 or 新建） |
| 1 | Research question | 研究目标 |
| 2 | Fixed Conditions | 迭代间不可变的条件 |
| 3 | Pressure Conditions | 每轮的公平限制（时间/epoch） |
| 4 | Variable Conditions | Agent A 可以调整的范围 |
| 5 | Evaluation | 评测指标、方向、eval command |
| 6 | Termination | 最大轮数、目标值 |
| 7 | Agent Boundary | A/B 职责划分（有默认值） |

### 3.3 检测触发条件

以下任一为真即进入 autoresearch 模式：
1. `spml:autoresearch-create` 已在当前会话中被调用
2. 关键词匹配：autoresearch / auto research / 自动研究 / 自动实验 等
3. 模式匹配：目标是搜索/优化而非验证单一假设

## 4. Baseline Validation Gate

### 4.1 设计决策：L1 不可跳过

**问题：** brainstorming 阶段的 validation scope 确认允许用户跳过任何 VP 层。对 toy task + autoresearch，容易出现"每轮训练本身就是验证"的想法，导致 L0/L1 被跳过。

**后果：** baseline 代码从未被验证能跑通，autoresearch 从 Round 1 就卡住。

**规则：** autoresearch 场景下，VP L1 Runtime Validation 强制开启，不可跳过。原因：
- autoresearch 需要一个已验证能跑通的 baseline 代码
- baseline 指标是 protocol 的 baseline value
- 压力条件终止逻辑需要经过实际验证

### 4.2 Handoff HARD-GATE

autoresearch-handoff 不允许在以下条件不满足时继续：

1. **VP L1 Runtime Validation passed** — 强制要求，如果被跳过则必须补跑
2. 所有其他已启用的 VP 层通过（L0 if enabled）
3. Base code 在压力条件下能正常运行（VP L1 验证了这一点）
4. 压力条件终止代码存在且有效（属于 Fixed Conditions）
5. Design doc 包含 `## Autoresearch Protocol` section

### 4.3 Handoff 产出

| 文件 | 内容 |
|------|------|
| `autoresearch-protocol.md` | 6 要素协议文档 |
| `experiences.md` | Summary + baseline 记录 |

启动指令：`run autoresearch at <experiment-dir>`（由 `spml:autoresearch-run` 处理）

## 5. Supervisor Loop

### 5.1 架构：三角色

```
Supervisor（主 session）
  ├── 调度 Agent A（后台 subagent）— 设计 + 编码 + 训练
  ├── 调度 Agent B（后台 subagent）— 独立评测 + 合规审查 + 记录
  └── Git 管理 — commit on improve, rollback on failure
```

**职责边界：**
- Supervisor：流程控制、git 写操作、终止判断、异常恢复。不设计策略，不写代码。
- Agent A：创造性工作。读 protocol 了解约束，读 experiences 学习历史，修改代码，训练。
- Agent B：独立审计。运行 eval_command（不读 training log），审查 Agent A 是否违反 Fixed Conditions。

### 5.2 Worktree 隔离

所有 autoresearch 操作在隔离的 git worktree 中进行，主工作目录不受影响。

- **新建：** `git worktree add ../autoresearch-{name} HEAD`
- **恢复：** 检测已有 worktree（`git worktree list`），直接复用
- 所有 git 操作（commit / rollback / checkout）在 worktree 中执行
- 结束时用户选择：merge / 保留 / 删除 worktree

### 5.3 Agent A Prompt 设计

关键约束：
- **必须先读 protocol**，特别关注 Fixed Conditions 和 Variable Conditions
- Fixed Conditions 不可修改，违反即该轮无效
- Variable Conditions 是创造空间的边界
- **必须先写 strategy.md**，明确说明改了哪些 Variable Conditions 及原因
- 训练后由压力条件自动终止

### 5.4 Agent B Prompt 设计

三部分职责：

**Part 1: 合规审查**
- 读 protocol 的 Fixed/Variable Conditions
- 读 Agent A 的 strategy.md
- diff 实际代码变更，验证：
  - 未违反 Fixed Conditions
  - 只修改了 Variable Conditions 范围内的内容
  - strategy.md 描述与实际代码一致
- 违反 Fixed Conditions → verdict 强制为 not_improved

**Part 2: 独立评测**
- 运行 protocol 中的 eval_command — 禁止用 training log 代替
- 记录自己评测的指标值，不用 Agent A 的输出

**Part 3: 记录经验**
- 追加 Round entry 到 experiences.md
- 包含：Strategy / Compliance / Result / Verdict / Insight

### 5.5 Git 管理

**Improved：**
```
cp experiences.md /tmp/backup
git add -A && git commit -m "autoresearch: round N — metric=value (improved)"
更新 experiences.md Summary
```

**Not improved：**
```
cp experiences.md /tmp/backup
git checkout -- . && git clean -fd    # rollback 代码
cp /tmp/backup experiences.md          # 恢复 experiences
更新 experiences.md Summary（只增 round count）
```

**HARD-GATE：** 只有 Supervisor 执行 git 写操作。Agent A/B 无 git 写权限。

### 5.6 终止逻辑

**HARD-GATE：仅以下条件终止，无例外，无主观判断。**

- `target` 已设 AND 指标达到 target → `target_reached`
- `round == max_rounds` → `completed`
- 其他情况 → 继续下一轮

如果 target 为 "none"，唯一终止条件是 max_rounds。Supervisor 不得判断"指标不可能再提升"或"已达理论上限"。协议说跑多少轮就跑多少轮。

### 5.7 Per-Round Task List

每轮开始时创建 task list，用户可追踪进度：

```
Round N/M
  ☐ Agent A: design + code + train
  ☐ Agent B: evaluate + review + record
  ☐ Git: commit or rollback based on verdict
  ☐ Check termination
```

## 6. Supervisor Liveness（重点）

### 6.1 问题

Supervisor 是语言模型，不是持久进程。它在"演"一个 for 循环——没有程序化的循环保障。随时可能因以下原因中断：
- 会话超时
- Context 溢出被压缩，丢失循环状态
- 模型自行判断"够了"提前停止
- Agent 完成通知未可靠触发

### 6.2 主动调度：Sleep-Check 循环

Supervisor 使用 sleep-check 模式主动管理调度，而非被动等待通知。

```
1. 派发 Agent（run_in_background: true）
2. 根据压力条件估算等待时间（如 time_limit=5min → sleep 5min）
3. Bash(sleep <seconds>)
4. 检查 Agent 是否完成（输出文件、completion signal）
5. 完成 → 进入下一步
6. 未完成 → sleep 缩短为前次的一半，再检查
7. 重复直到完成或超时
```

**关键优势：**
- Supervisor 始终持有控制权（sleep 总会结束）
- 可根据上下文自适应等待时间
- 不依赖自动通知机制的可靠性

### 6.3 兜底机制：CronCreate Heartbeat

```
CronCreate(
  cron: "*/5 * * * *",
  prompt: "Autoresearch heartbeat — check your loop status."
)
```

- **Session-scoped**：仅当前 session 有效，session 关闭即消失
- **5 分钟间隔**：在 sleep-check 正常运行时不触发（REPL busy），仅在循环意外中断后触发
- **作用**：anti-idle nudge，提醒 Supervisor 检查自己的循环状态

### 6.4 REPL Idle 语义

CronCreate 的 prompt 只在 REPL 处于 idle 状态（不在处理 query）时注入。

| 场景 | REPL 状态 | Cron 触发？ |
|------|----------|------------|
| Agent 在后台跑 + Supervisor 在 sleep-check | busy（执行 Bash sleep） | 不触发 |
| Agent 在后台跑 + Supervisor turn 已结束 | idle | 触发 |
| Sleep-check 循环意外崩溃 | idle | 触发 |

**不会积压：** Cron 是条件触发（时间匹配 AND REPL idle），不是消息入队。REPL 忙时 cron 跳过，不排队。

### 6.5 双层防护总结

```
主动层（Sleep-Check）：
  派 Agent → sleep 估算时间 → 醒来检查 → 未完成 → 缩短 sleep → 再查 → 完成 → 下一步
  ✓ Supervisor 始终活跃，不丢失控制

兜底层（CronCreate 5min）：
  Sleep-check 意外断了 → REPL idle → 5 分钟内 cron 触发 → Supervisor 重新进入循环
  ✓ 覆盖 sleep-check 本身崩溃的极端场景
```

**HARD-GATE：循环自治。** 每轮结束后立即进入下一轮，不等用户输入，不问用户问题，不停下来总结或提供选项。

## 7. Anomaly Recovery

### 7.1 Agent 超时/崩溃

1. 在 experiences.md 记录 `agent_error`
2. Rollback 部分代码变更
3. 重试一次。再失败则跳过该轮，继续下一轮。

### 7.2 Session 中断恢复

启动时检测 `experiences.md` Summary：
- `Status: running` + `Total rounds > 0` → 恢复模式
- 最后一轮无 Verdict → 中断发生在 round 中间，rollback 未提交变更，从该轮重新开始
- 最后一轮有 Verdict → 从下一轮继续

git HEAD 一致性检查：必须匹配最近一次 committed improvement（或 baseline）。

### 7.3 连续失败

5 轮连续 not_improved → 输出 plateau 警告，但继续运行（不停止）。

## 8. Artifacts

### 8.1 autoresearch-protocol.md

```markdown
# Autoresearch Protocol: <title>

## Research Question
## Environment
## Fixed Conditions
## Pressure Conditions
## Variable Conditions
## Evaluation（含 baseline value）
## Termination
## Agent Boundary
```

### 8.2 experiences.md

```markdown
# Experiment Experiences

## Summary
- Best result: {metric} = {value} (Round N)
- Total rounds: N / max
- Status: not_started / running / completed / target_reached

## Round 0: Baseline
## Round 1
- **Strategy**: ...
- **Compliance**: ✅ / ❌
- **Result**: metric = value
- **Verdict**: ✅ committed / ❌ rolled back
- **Insight**: ...
```

### 8.3 strategy.md

Agent A 每轮开始前写入，描述：
- 改了哪些 Variable Conditions
- 为什么这样改（基于 experiences 的学习）
- 预期效果

Agent B 读取并对比实际代码变更，验证一致性。

## 9. Feature Checklist

### Protocol Definition
- [ ] brainstorming 检测到 autoresearch 意图后进入 8 问题流程
- [ ] 8 个问题逐一询问，不批量
- [ ] design doc 包含 `## Autoresearch Protocol` section（6 要素完整）
- [ ] autoresearch 场景下 VP L1 不可跳过

### Handoff
- [ ] HARD-GATE：VP L1 必须通过才能 handoff
- [ ] 生成 `autoresearch-protocol.md`（6 要素从 design doc 提取）
- [ ] 初始化 `experiences.md`（含 baseline 指标）
- [ ] 启动指令为 `run autoresearch at <dir>`（不生成冗长 prompt 文件）

### Supervisor Startup
- [ ] 读取 autoresearch-protocol.md
- [ ] 创建或复用 git worktree（不在主库操作）
- [ ] 验证 worktree 状态（base commit + experiences.md）
- [ ] 检测恢复模式（experiences.md status + round count）
- [ ] 设置 CronCreate heartbeat（*/5, session-scoped）

### Main Loop
- [ ] 每轮创建 task list（4 个子任务）
- [ ] Agent A 以 `run_in_background: true` 派发
- [ ] Supervisor 进入 sleep-check 循环等待 Agent A
- [ ] Agent A 完成后，Agent B 以 `run_in_background: true` 派发
- [ ] Supervisor 进入 sleep-check 循环等待 Agent B
- [ ] 循环自治：轮间不等用户输入

### Agent A
- [ ] 先读 protocol，关注 Fixed/Variable Conditions
- [ ] 先写 strategy.md，再改代码
- [ ] strategy.md 明确说明改了哪些 Variable Conditions
- [ ] 不修改 Fixed Conditions（终止逻辑、评测逻辑等）

### Agent B
- [ ] Part 1: 合规审查 — diff 代码验证 Fixed Conditions 未被违反
- [ ] Part 2: 独立评测 — 运行 eval_command，不读 training log
- [ ] Part 3: 记录经验 — 包含 Compliance 字段
- [ ] 违反 Fixed Conditions → verdict 强制为 not_improved

### Git Management
- [ ] 只有 Supervisor 执行 git 写操作
- [ ] 所有 git 操作在 worktree 中
- [ ] Improved: git commit（含 experiences.md）
- [ ] Not improved: git rollback 代码 + 保留 experiences.md
- [ ] experiences.md 在 rollback 后不丢失

### Termination
- [ ] 仅 target_reached 或 max_rounds 终止
- [ ] target 为 none 时唯一终止条件是 max_rounds
- [ ] Supervisor 不做主观判断提前停止

### Liveness
- [ ] Agent 以 run_in_background 派发
- [ ] Supervisor 使用 sleep-check 主动监控
- [ ] Sleep 时间根据压力条件估算，逐步缩短
- [ ] CronCreate heartbeat 每 5 分钟，session-scoped
- [ ] Sleep 期间 REPL busy，cron 不触发不积压
- [ ] Sleep-check 崩溃 → REPL idle → cron 恢复

### Anomaly Recovery
- [ ] Agent 崩溃：记录 error，rollback，重试一次
- [ ] Session 中断：从 experiences.md 恢复状态
- [ ] 5 轮连续失败：输出 plateau 警告，继续运行

### Final Report
- [ ] 删除 watchdog cron
- [ ] 更新 experiences.md status
- [ ] 提供 worktree 处理选项（merge / 保留 / 删除）
- [ ] 输出完整报告（best metric, improvement, insights）
