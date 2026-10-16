# 01 — CoT 收益的条件性

→ 论文 Fig 2（任务异质性）和 Fig 3（模型规模）

**一句话**：CoT 提示词不是"加了就好"的通用增强器——收益取决于任务类型和模型规模。

---

## 这个实验做什么

检验一个被广泛默认的前提："Let's think step by step" 是否应该在所有场景下都不加验证地使用。

**方法**：在 4 个数据集（GSM8K、MMLU、HumanEval、AIME 2024）上比较 curated prompt suite，并在 5 个 Qwen2.5 规模（0.5B–32B）上比较效果。suite 由 1 个 standard baseline、14 个 conventional Zero-shot CoT variants 和 1 个 post-selected Term-first diagnostic probe 组成。

**结论**：
1. 同一个 prompt 在不同任务上的效果方向不一致——在某个数据集上增益的模板，在另一个上可能退化。
2. 规模效应与 backend、dtype、token cap 和 offload 混杂，只能描述为 scale–implementation bundle，不能归因为 model capability。
3. prompt 的微小改动（如换一个词）会在题目级别改变"哪些题对、哪些题错"的集合，而非平滑地改分；该结果作为历史描述性证据使用。

**在论文中的角色**：对强形式 universal-enhancer hypothesis 提供描述性反证；不支持"小模型普遍无收益"或任何规模阈值。

---

## 实验条件

| 维度 | 详情 |
|------|------|
| 模型系列 | Qwen2.5（0.5B, 1.5B, 3B, 7B, 32B）|
| 数据集 | GSM8K（数学推理）、MMLU（知识问答）、HumanEval（代码生成）、AIME 2024（竞赛数学）|
| 提示词 | 15 种变体（Standard、14 个 CoT 前缀、Term-first）|
| 评测方式 | GSM8K/MMLU/HumanEval: pass@4 (n=4); AIME 2024: pass@1 (n=64, 4 轮合并)|
| 框架 | OpenCompass |

---

## 关键发现（论文 §2）

| 发现 | 具体数字 | 含义 |
|------|---------|------|
| GSM8K 上 worst CoT variant | −2.11 to −1.29 pp（95% CI）| 加 CoT 不一定提升，可能倒退 |
| Qwen2.5 1.5B/3B 多数 CoT 变体 | 负向 | 中间规模描述性结果；不能外推为"小模型普遍无收益" |
| Qwen2.5 32B 多数 CoT 变体 | 9/14 为正（端点不显著）| 非单调规模模式，不能视为稳定阈值 |
| Term-first on AIME 2024 | +0.10 pp (n=256) | 接近零；论文严格按"接近中性"措辞 |
| Term-first on HumanEval | +10.82 pp | 唯一强正向的 benchmark；论文作为说明性证据 |

---

## 数据溯源

| Claim | 数据文件 | 格式 |
|-------|---------|------|
| CoT 在 GSM8K/HumanEval 上频繁退化，MMLU/AIME 2024 增益有限 | `data/fig1_task_heterogeneity_scores.csv` | CSV (15 prompts × 4 datasets × 4 runs) |
| Scale–implementation bundle 下的非单调效果分布 | `data/fig2_model_scale_scores.csv` | CSV (15 prompts × 5 scales × 4 runs) |
| Term-first 在 AIME 2024 上 4 批合并 n=256 | `data/aime2024_qwen25_3b_n64_16k_scores/termfirst_repeatCDEF_aggregate_summary.csv` | CSV |
| Standard baseline on AIME 2024 n=64 | `data/aime2024_qwen25_3b_n64_16k_scores/standard_baselineB_aggregate_summary.csv` | CSV |
| 14 CoT variants on AIME 2024 n=64 each | `data/aime2024_qwen25_3b_n64_16k_scores/prompts15_aggregate_summary.csv` | CSV |
| 合并 panel 数据（4 数据集 × 5 规模）| `data/Fig2_TaskHeterogeneity_data.csv` | CSV (script Fig 2 读) |

---

## 图

| 论文图号 | 生成脚本 | 输入 | 输出位置 |
|----------|---------|------|---------|
| Fig 2 | `code/Fig2_TaskHeterogeneity.py` | `data/fig1_task_heterogeneity_scores.csv` + `data/Fig2_TaskHeterogeneity_data.csv` + `data/aime2024_qwen25_3b_n64_16k_scores/` | `figures/Fig2_*.{png,jpg,tiff,svg}` |
| Fig 3 | `code/Fig3_ScalingLaws.py` | `data/fig2_model_scale_scores.csv` | `figures/Fig3_*.{png,jpg,tiff,svg}` |

---

## 复现

```bash
cd code/
python3 Fig2_TaskHeterogeneity.py    # Fig 2: 任务异质性
python3 Fig3_ScalingLaws.py         # Fig 3: scale 效果分布
```

无需 GPU；纯 Python 数据处理 + matplotlib 出图，~30 秒。

---

## 约束与注意事项

- **Term-first 的来源说明**：本实验的 Term-first prompt 是 `What is the query? Term first, then reasoning.`，从"rephrase-first"提示词族 post-selected 而来（论文 §2.3）。
- **AIME 2024 多次取样的目的**：通过 n=64（baseline）/ n=256（term-first pooled across 4 batches）压低单批偶然性。论文中所有 AIME 数字均以"pooled mean"措辞呈现。
- **不可外推的声称**：脚本不会也不能生成"CoT 在 X 任务上必胜"等普适性结论；本实验是描述性的，非证伪。
