# 02 — 提示前缀作为隐藏状态初始化

→ 论文 §3（机制分析）

**一句话**：同题、同模型、只换前缀 → 三条完全不同的隐藏状态轨迹。提示前缀充当了"状态初始化"。

---

## 这个实验做什么

排除了"CoT 是通用增强器"（案例 1）后，下一个问题是：**提示前缀到底在模型内部做了什么？** 本实验用 Logit Lens 技术在受控条件下直接观察。

**方法**：选一道 AIME 2024 竞赛题（II-4），用同一个模型（DeepSeek-R1-Distill-Qwen-1.5B），只改变 prompt 前缀，追踪 29 层隐藏状态在 32 次生成中的概率轨迹。

**三种条件**（仅前缀不同）：
- `"Wait, think."` → 预期：过早锁定答案
- `"Let's reason through this step by step."` → 预期：正常推理节奏
- `"Let's think step by step."` → 预期：过度反思、无法收敛

**结论**：三种前缀确实导向三条显著不同的隐藏状态轨迹——过早收敛（浅层锁定）、延迟承诺（深层重分配）、持续振荡（无法收敛）。这说明提示前缀的作用不是"编码解题步骤"，而是设定模型计算动力学的初始条件。

---

## 实验条件

| 维度 | 详情 |
|------|------|
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B（29 层） |
| 题目 | AIME 2024-II-4（单题受控） |
| 条件数 | 3（under / normal / over） |
| 每条件运行次数 | 32 |
| 分析技术 | Logit Lens（隐藏状态 → 词汇概率投影） |
| 设备 | Mac MPS / float16 |

---

## 三种条件详情

| 论文条件 | Prompt | 追踪 Token | 策略 |
|---------|--------|-----------|------|
| 过早收敛 (under) | `"Wait, think."` | `this` | `</think>` 后第一个 token |
| 延迟承诺 (normal) | `"Let's reason through this step by step."` | `have` | `</think>` 后第一个 token |
| 持续振荡 (over) | `"Let's think step by step."` | `Wait` | 序列末尾向前搜索 |

---

## 关键发现

| 发现 | 数值 | 解读 |
|------|------|------|
| under 在第 27 层达 prob=1.0 | **L27 / 1.0** | 过早锁定，深层无实质推理 |
| normal 峰值在第 26 层 | **L26 / 0.309** | 保持不确定性，深层重分配 |
| over 峰值在第 28 层 | **L28 / 0.240** | 振荡后无法收敛 |
| over 末层塌缩 | **L29 / 0.0068** | 从 0.240 暴跌至接近零 |
| under 末层维持 | **L29 / 0.951** | 锁定后不松手 |
| 生成长度比 | **5k vs 8.7k vs 71k tokens** | over 生成长度是 under 的 14 倍 |

---

## 数据溯源

| Claim | 数据文件 |
|-------|---------|
| 三态概率曲线（29 层） | `data/real_logit_lens_data_controlled.json` |
| 三态选择元信息（长度、Wait 计数） | `data/selected_controlled_case.json` |
| 32 次运行完整轨迹 | `data/logit_lens_data_1.5B_32runs_all_states.json` |

---

## 图

| 论文图号 | 生成脚本 | 输入 | 输出 |
|----------|---------|------|------|
| Fig7 | `code/plot_real_logit_lens_eeg.py` | `data/real_logit_lens_data_controlled.json` | `figures/Fig7_LogitLens_32runs_3d_eeg.*` |
| Fig3 | `code/plot_reflection_scatter.py` | `data/reflection_1p5b_real.json` | `figures/Fig3_Reflection_Scatter.*` |

---

## 复现

核心提取脚本（`real_logit_lens_extractor.py`）依赖 MPS/GPU，不可在纯 CPU 快速复现。绘图脚本可直接运行：

```bash
cd code/
python3 plot_real_logit_lens_eeg.py    # Fig7: 3D EEG
python3 plot_reflection_scatter.py     # Fig3: 反思词散点
```

---

## 处理流程

```
find_controlled_cases.py        → 扫描 AIME 预测，按长度 + Wait 计数选三态
real_logit_lens_extractor.py    → MPS 前向传播，逐 token 投影 29 层 → JSON
verify_logit_lens_data.py       → 验证均值轨迹一致性
plot_real_logit_lens_eeg.py     → 3D EEG 出图
```

---

## 约束与注意事项

- **信任级：最高**。完整代码链（选择 → 提取 → 验证 → 绘图），JSON 可逐层核验。
- **单题约束**：仅一道题（AIME 2024-II-4），不是统计样本。发现的普适性需要跨题验证（见 03-logit-lens）。
- **全错题**：1.5B 模型在这道竞赛数学题上答错。轨迹反映的是错误推理的隐藏状态模式。
- **Token 概率不可直接横比**：不同 token 追踪不同目标词，比较的是轨迹形态（收敛时机、晚期震荡），非绝对概率值。

---

## 开发脚本

`code/_dev/` 中的脚本为迭代优化和探索性分析的历史版本，完整保留以备追溯。
