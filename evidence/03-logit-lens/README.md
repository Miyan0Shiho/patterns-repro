# 03 — Logit Lens: prompt-associated tracked-token trajectories

→ 论文 §3.4（隐藏状态的描述性扩展）

**一句话**：在五个 selected AIME problems 上，不同 prompt conditions 呈现出不同的 tracked-token probability trajectories；这些结果只用于描述 representation organization，不建立 mechanism。

---

## 这个实验做什么

02-state-divergence 在单题上发现了三种隐藏状态轨迹。本实验将同一分析扩展到 **5 题 × 4 条件**，检验两个问题：

1. 单题发现的三种轨迹是否在别的题目上也稳定复现？
2. **term-first** 的隐藏状态轨迹属于哪种类型？

**方法**：选 5 道 AIME 2024 题目（覆盖对数/几何/数论/组合/代数五类），每道题用 4 种前缀各跑一次，追踪 Logit Lens 概率轨迹。

**结论**：
- 追踪 token 的 probability trajectories differ across prompt conditions on the selected problems.
- 这些条件标签是 post-hoc descriptions of the plotted trajectories, not validated cognitive states.
- 题目数量、target token、stopping rule 和 projection pipeline 的限制使结果保持 exploratory and descriptive.

---

## 实验条件

| 维度 | 详情 |
|------|------|
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B（29 层） |
| 题目 | AIME 2024 × 5 题（对数/几何/数论/组合/代数各一） |
| 条件 | 4（过早收敛 / 延迟承诺 / 持续振荡 / term-first） |
| 分析技术 | Logit Lens |
| 设备 | Mac MPS / float16 |

---

## 四种条件

| 论文条件 | Prompt | 追踪 Token |
|---------|--------|-----------|
| 过早收敛 | `"Wait, think."` | `<end_of_sentence>` |
| 延迟承诺 | `"Let's reason through this step by step."` | `<end_of_sentence>` |
| 持续振荡 | `"Let's think step by step."` | `Wait` |
| **term-first** | `"What is the query? Term first, then reasoning."` | `<end_of_sentence>` |

---

## 关键发现

| 发现 | 数值 | 解读 |
|------|------|------|
| 延迟承诺收敛层（5 题） | 全部 L20 | 该 selected setup 内的描述性观测，不是跨题稳定性证明 |
| term-first 收敛层（4/5 题） | L20 | 与 delayed commitment 在该样本中 layer index 相同 |
| term-first 异常题（Q4 组合） | L26，使用 fallback token | 不可与 EOS-tracked trajectories 直接比较 |
| 过早收敛峰值层范围 | L19–L27（分散） | 不能解释为稳定的 commitment timing |
| 持续振荡追踪概率 | 0.002–0.594 | 与 EOS 等不同 target token 的绝对值不可比 |
| 模型正确率 | 可判定的 Q1-Q3 上 11/12 条件正确 | 轨迹同时覆盖正确与错误推理 |

---

## 数据溯源

| Claim | 数据文件 |
|-------|---------|
| 5 题 × 4 条件 Logit Lens 概率轨迹 | `data/logit_lens_5q_4cond.json` |
| 5 题 × 4 条件准确率 | `data/accuracy_summary.json` |

---

## 图

| 论文图号 | 生成脚本 | 输入 | 输出 |
|----------|---------|------|------|
| Fig4 | `code/plot_logit_lens_4cond.py` | `data/logit_lens_5q_4cond.json` | `paper/figures/Fig4_Logit_Lens_4cond.*` |

---

## 复现

```bash
cd code/
python3 plot_logit_lens_4cond.py
```

---

## 与 02-state-divergence 的关系

| 维度 | 02-state-divergence | 03-logit-lens |
|------|-------------------|---------------|
| 范围 | 单题（AIME 2024-II-4） | 5 题（五类数学） |
| 条件 | 3（无 term-first） | 4（含 term-first） |
| 运行次数 | 32/条件（统计束） | 1/条件（跨题多样性优先） |
| 角色 | 描述单题轨迹差异 | 描述五个 selected problems 上的轨迹差异 |

两个实验共同支持一个描述性结论：prompt prefix 与 tracked-token probability trajectories 相伴变化，但现有设计不建立因果机制或 benchmark gain 的解释。

---

## 约束与注意事项

- **信任级：数据可追溯，解释受可比性限制**。JSON 可逐层核验，但 different target tokens and pipelines prevent mechanism claims.
- **正确率约束**：可判分的 Q1–Q3 上 1.5B 模型 11/12 条件正确（Q4/Q5 题目无唯一解，N/A）。轨迹同时覆盖正确与错误推理，写作时按此披露，不可表述为"全部答错"。
- **追踪 token 不可比**：三种条件追踪结束标记，persistent oscillation 追踪 "Wait"。绝对 probability magnitudes are not comparable across conditions.
- **5 题样本**：题目量有限，每格一次 generation，且 conditions are post hoc；不能推断 stable taxonomy or mechanism.
- **单模型**：仅 DeepSeek-R1-Distill-Qwen-1.5B，不支持跨模型泛化声称。
- **Wait 计数**：`logit_lens_5q_4cond.json` 的 `wait` 字段只统计生成的 continuation，不包含提示词。论文的单题 `1/1/21` Wait 值来自 `02-state-divergence/data/selected_controlled_case.json`，同样为 continuation-only。
