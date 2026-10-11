# 01 — CoT 收益的条件性

→ 论文案例 1（§2-3）

**一句话**：CoT 提示词不是"加了就好"的通用增强器——收益取决于任务类型和模型规模。

---

## 这个实验做什么

本实验检验一个被广泛默认的前提："Let's think step by step" 是否应该在所有场景下都不加验证地使用。

**方法**：在 4 个数据集（GSM8K、MMLU、HumanEval、AIME 2024）上比较 curated prompt suite，并在 5 个 Qwen2.5 规模（0.5B–32B）上比较效果。suite 由 1 个 standard baseline、14 个 conventional Zero-shot CoT variants 和 1 个 post-selected Term-first diagnostic probe 组成。

**结论**：
1. 同一个 prompt 在不同任务上的效果方向不一致——在某个数据集上增益的模板，在另一个上可能退化。
2. 规模效应与 backend、dtype、token cap 和 offload 混杂，只能描述为 scale–implementation bundle，不能归因为 model capability。
3. prompt 的微小改动（如换一个词）会在题目级别改变“哪些题对、哪些题错”的集合，而非平滑地改分；该结果作为历史描述性证据使用。

**在论文中的角色**：对强形式 universal-enhancer hypothesis 提供描述性反证；不支持“小模型普遍无收益”或任何规模阈值。

---

## 实验条件

| 维度 | 详情 |
|------|------|
| 模型系列 | Qwen2.5（0.5B, 1.5B, 3B, 7B, 32B） |
| 数据集 | GSM8K（数学推理）、MMLU（知识问答）、HumanEval（代码生成）、AIME 2024（竞赛数学） |
| 提示词 | 15 种变体（Standard、各种 CoT 前缀、Term-first 等） |
| 评测方式 | GSM8K/MMLU/HumanEval: pass@4 (n=4)；AIME 2024: pass@1 (n=64, 4 轮合并) |
| 框架 | OpenCompass |

---

## 关键发现

| 发现 | 具体数字 | 含义 |
|------|---------|------|
| GSM8K 上最差 prompt 比 baseline 低 | −2.8 pp | 加 CoT 不一定提升，可能倒退 |
| 0.5B–3B 多数 conventional 变体 ΔACC | 负值为主 | curated suite 内该 scale 区间的描述性结果；不能外推为“小模型普遍无收益” |
| 32B 多数 conventional 变体为正 | 9/14 为正，端点效应不显著 | 只能描述为非单调规模模式，不能视为稳定阈值 |
| Term-first 在 AIME 2024 上 | +0.10 pp（C-F pooled，n=256） | 接近零，不解释正负号 |
| AIME 2024 15 变体最佳 | ≈6.72% | 即使最佳变体，AIME 绝对分仍然很低 |

---

## 数据溯源

| Claim | 数据文件 | 格式 |
|-------|---------|------|
| CoT 在 GSM8K/HumanEval 上频繁退化，MMLU/AIME 2024 增益有限 | `data/fig1_task_heterogeneity_scores.csv` | CSV |
| Scale–implementation bundle 下的非单调效果分布 | `data/fig2_model_scale_scores.csv` | CSV |
| Prompt 微扰改变题目级对错分布 | `data/extracted_matrix_1000.npy` | NumPy (15×1319) |
| Term-first 在 AIME 2024 上效应接近零 | `data/aime2024_qwen25_3b_n64_16k_scores/` | CSV（多文件） |

---

## 图

| 论文图号 | 生成脚本 | 输入 | 输出位置 |
|----------|---------|------|---------|
| Fig1 | `code/Fig1_TaskHeterogeneity.py` | `data/fig1_task_heterogeneity_scores.csv` | `figures/Fig1_BarChart_TaskHeterogeneity.*` |
| Fig2 | `code/Fig2_ScalingLaws.py` | `data/fig2_model_scale_scores.csv` | `figures/Fig2_BarChart_ModelScale.*` |
| Fig4 | `code/Fig4_AnswerHeatmap.py` | `data/extracted_matrix_1000.npy` | `figures/Fig4_Heatmap_AnswerFragility.*` |
| AIME 补充 | `code/Fig1_TaskHeterogeneity_AIME2024.py` | `data/aime2024_*/*.csv` | `figures/Fig1_*AIME2024*` |

---

## 复现

```bash
cd code/
python3 Fig1_TaskHeterogeneity.py          # Fig1: 任务异质性
python3 Fig2_ScalingLaws.py                 # Fig2: scale 效果分布
python3 Fig4_AnswerHeatmap.py              # Fig4: 答案脆弱性热力图
python3 Fig1_TaskHeterogeneity_AIME2024.py # AIME2024 补充面板
```

---

## 约束与注意事项

- **Fig1/Fig2 CSV 数据源**：提取自历史硬编码数组，非直接来自 OpenCompass 原始输出。数值可审计，但完整溯源链不完整；V3.8 因此只使用 descriptive/exploratory wording，不报告 population inference。
- **Fig4 矩阵**：从 PNG 热力图反向提取。原始 OpenCompass JSON 已丢失（`data/data_fig4/` 仅有 `.out` 日志）。
- **AIME 2024 数据**：Standard B 为 `n=64`；Term-first 使用 C、D、E、F 四个重复批次汇总，共 `n=256`。Pooled mean 为 `6.0939%`，相对 Standard B 约 `+0.10 pp`。为避免单独选择某一批造成后选问题，主结果统一使用 pooled estimate。
- **term-first 规模数据（2026-08-06 重跑）**：`fig2_model_scale_scores.csv` 中 term-first 行的 0.5B/1.5B/7B/32B 四轮数据已重跑为真 4 轮独立（seeds 44–47，GSM8K pass@4，T=0.6）。旧数据（`workspace/results_termfirst_scale/`）为 2 真+2 复制，且 0.5B/1.5B 数值偏高（0.5B 41.26→36.80，1.5B 52.28→49.09）。3B 保留 group48 真独立数据（未重跑）。详见 `notes/C类_规模实验最小重跑方案.md`。
- **开发脚本**：`code/_dev/` 中的脚本为一次性提取/回退工具，非规范作图流程的一部分。

---

## 开发脚本

`code/_dev/` 目录下保留了一次性使用的开发脚本（矩阵提取、回退生成、文献分析），完整保留以备追溯。
