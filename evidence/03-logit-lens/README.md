# 03 — Logit Lens: prompt-associated tracked-token trajectories

→ 论文 Fig 5（5 题 × 4 条件 概率轨迹）+ Table S3（15 题 first-token peak-layer 分布）

**一句话**：在 5 个 selected AIME 2024 problems 上，4 种 prompt conditions 的 tracked-token probability trajectories 形态不同；这些结果只用于描述 representation organization，不建立 mechanism。

---

## 这个实验做什么

02-state-divergence 在单题上发现了三种隐藏状态轨迹。本实验将同一分析扩展到 **5 题 × 4 条件**，检验两个问题：

1. 单题发现的三种轨迹是否在别的题目上也稳定复现？
2. **Term-first** 的隐藏状态轨迹属于哪种类型？

**方法**：选 5 道 AIME 2024 题目（覆盖对数/几何/数论/组合/代数五类），每道题用 4 种前缀各跑一次，追踪 Logit Lens 概率轨迹。

---

## 实验条件

| 维度 | 详情 |
|------|------|
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B（29 层）|
| 题目 | AIME 2024 × 5 题（对数/几何/数论/组合/代数各一）|
| 条件 | 4（under_thinking / normal_thinking / over_thinking / term_first）|
| 分析技术 | Logit Lens |
| 设备 | Mac MPS / float16 |

---

## 四种条件

| 论文条件 | Prompt | 追踪 Token |
|---------|--------|-----------|
| Premature convergence | `"Wait, think."` | `<end_of_sentence>` |
| Delayed commitment | `"Let's reason through this step by step."` | `<end_of_sentence>` |
| Persistent oscillation | `"Let's think step by step."` | `Wait` |
| **Term-first** | `"What is the query? Term first, then reasoning."` | `<end_of_sentence>` |

---

## 关键发现

| 发现 | 数值 | 解读 |
|------|------|------|
| Delayed commitment 收敛层（5 题）| 全部 L20 | 该 selected setup 内的描述性观测，不是跨题稳定性证明 |
| Term-first 收敛层（4/5 题）| L20 | 与 delayed commitment 在该样本中 layer index 相同 |
| Term-first 异常题（Q4 组合）| L26，使用 fallback token | 不可与 EOS-tracked trajectories 直接比较 |
| Premature convergence 峰值层范围 | L19–L27（分散）| 不能解释为稳定的 commitment timing |
| Persistent oscillation 追踪概率 | 0.002–0.594 | 与 EOS 等不同 target token 的绝对值不可比 |
| 模型正确率 | 可判定的 Q1-Q3 上 11/12 条件正确 | 轨迹同时覆盖正确与错误推理 |

---

## 数据溯源

| Claim | 数据文件 |
|-------|---------|
| 5 题 × 4 条件 Logit Lens 概率轨迹 | `data/logit_lens_5q_4cond.json` |
| 5 题 × 4 条件准确率 | `data/accuracy_summary.json` |
| 15 题 first-token peak-layer 分布（Table S3）| `data/logit_lens_1.5b_full.json` |
| Table S3 派生统计 | `data/table_s3_peak_layers.json` |

---

## 图

| 论文图号 | 生成脚本 | 输入 | 输出 |
|----------|---------|------|------|
| Fig 5 | `code/Fig5_LogitLens.py` | `data/logit_lens_5q_4cond.json` | `figures/Fig5_Logit_Lens_4cond.{png,jpg,tiff,svg}` |
| Table S3 | `code/compute_peak_layers.py` | `data/logit_lens_1.5b_full.json` | `data/table_s3_peak_layers.json` |

---

## 复现

```bash
cd code/
python3 Fig5_LogitLens.py             # Fig 5
python3 compute_peak_layers.py        # Table S3 派生统计
```

无需 GPU；纯 Python 加载 JSON + matplotlib 出图，~5 秒。

---

## 约束与注意事项

- **追踪 token 不可直接横比**：三种条件追踪 `<end_of_sentence>`，persistent oscillation 追踪 `Wait`。绝对 probability magnitudes are not comparable across conditions.
- **5 题样本**：题目量有限，每格一次 generation，且 conditions are post hoc；不能推断 stable taxonomy or mechanism.
- **单模型**：仅 DeepSeek-R1-Distill-Qwen-1.5B，不支持跨模型泛化声称。
- **Wait 计数**：`logit_lens_5q_4cond.json` 的 `wait` 字段只统计生成的 continuation，不包含提示词。论文的单题 `1/1/21` Wait 值来自 `02-state-divergence/data/selected_controlled_case.json`，同样为 continuation-only。
