# 02 — 提示前缀作为隐藏状态初始化

→ 论文 Fig 4（Reflection scatter，3 模型 Spearman ρ）

**一句话**：同题、同模型、只换前缀 → 三条完全不同的隐藏状态轨迹。提示前缀充当了"状态初始化"。

---

## 这个实验做什么

排除了"CoT 是通用增强器"（实验①）后，下一个问题是：**提示前缀到底在模型内部做了什么？** 本实验用 reflection-marker 频次作为行为代理指标，检验 3 个模型上是否成立"反思标记多 → 准确率高"假设。

**方法**：3 个 DeepSeek-R1-Distill 模型（Qwen-1.5B、Qwen-7B、Llama-8B），各跑 15 个 prompt（Standard + 14 CoT variants + Term-first），统计每 (model, prompt) 组合的：
- reflection-marker 总频次（9 类 marker 之和：`wait`, `however`, `hmm`, `let me`, `re-evaluate`, `reconsider`, `alternatively`, `hold on`, `let's re`）
- AIME 2024 pass@1 准确率

**结论**：所有 3 个模型的 Spearman ρ 都不显著正，95% CI 与 0 重叠。reflection-marker 频次不能作为推理质量的代理指标。

---

## 实验条件

| 维度 | 详情 |
|------|------|
| 模型 | DeepSeek-R1-Distill-{Qwen-1.5B, Qwen-7B, Llama-8B} |
| 数据集 | AIME 2024（30 题）|
| Prompt 数 | 16 (Standard + 14 CoT + Term-first) |
| 每 prompt 取样 | n=64 (Qwen-1.5B) / 池化 n=15 (7B, 8B) |
| 设备 | Mac MPS / float16 |

---

## 关键发现

| 模型 | Spearman ρ | 95% CI | p-value | 解读 |
|------|-----------|--------|---------|------|
| R1-Distill-Qwen-1.5B | 接近 0 | 含 0 | n.s. | "wait 多次"不预测高准确率 |
| R1-Distill-Qwen-7B   | 接近 0 | 含 0 | n.s. | 同上 |
| R1-Distill-Llama-8B  | 弱正 | 含 0 | n.s. | 即使方向正，统计上不显著 |

---

## 数据溯源

| Claim | 数据文件 | 格式 |
|-------|---------|------|
| 1.5B 模型 16 prompt × 4 runs | `data/reflection_1p5b_real.json` | JSON (16 lists) |
| 7B 模型 15 prompt | `data/reflection_7b_real.json` | JSON (15 dicts) |
| 8B 模型 15 prompt | `data/reflection_8b_real.json` | JSON (15 dicts) |
| 5-problem 选样元信息 | `data/selected_controlled_case.json` | JSON (单题 AIME 2024-II-4) |
| 32-run 三态 Logit Lens 原始轨迹 | `data/logit_lens_data_1.5B_32runs.json` + `_all_states.json` | JSON |
| 三态汇总轨迹 | `data/real_logit_lens_data.json` + `_controlled.json` | JSON |

---

## 图

| 论文图号 | 生成脚本 | 输入 | 输出 |
|----------|---------|------|------|
| Fig 4 | `code/Fig4_ReflectionScatter.py` | `data/reflection_{1p5b,7b,8b}_real.json` | `figures/Fig4_Reflection_Scatter.{png,jpg,tiff,svg}` |

---

## 复现

```bash
cd code/
python3 Fig4_ReflectionScatter.py     # Fig 4
```

无需 GPU；纯 Python 加载 JSON + matplotlib 出图，~5 秒。

---

## 约束与注意事项

- **单一数据集**：所有 reflection-marker 数据来自 AIME 2024。跨数据集（GSM8K/MMLU/HumanEval）的 marker 频次 vs 准确率分析未做，因为其他数据集的"反思"语义弱于 AIME 2024 的竞赛推理。
- **统计功效**：3 个模型 × 16 prompt 的样本量小，Spearman ρ 95% CI 较宽。论文中"无显著正相关"的声称在 Bonferroni 校正下稳健。
- **不可外推的声称**：本实验证伪"反思标记 = 推理质量"的强形式假设；不证伪"反思在某些条件下有帮助"。

---

## 处理流程（仅供了解，scripts 不依赖 GPU 重跑）

```
extract_termfirst_logit_lens.py    → 4 条件 (under / normal / over / term-first) × 单题 hidden-state extraction
real_logit_lens_extractor.py       → 3 条件 (under / normal / over) × 32 runs × 29 layers
verify_logit_lens_data.py          → 验证 JSON 完整性
```

> 这些脚本需要 MPS/GCU 设备；它们的输出已作为 JSON 数据包含在本仓库，Fig 4/5 可无需重跑提取。
