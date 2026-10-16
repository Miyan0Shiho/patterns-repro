# 实验证据（自包含单元）

> 本目录包含论文全部实验的证据链。每个子文件夹是一个自包含的实验单元（code + data + README）。

## 给评审/老师看的总览

论文的核心论点是：**AI 研究中存在"看到热门策略 → 不验证 → 照搬"的盲从模式，CoT 提示词是一个典型案例**。三个实验构成递进的论证链：

| 实验 | 文件夹 | 回答的问题 | 核心结论 | 论文图 |
|------|--------|-----------|---------|--------|
| ① CoT 收益的条件性 | [`01-macro-heterogeneity/`](01-macro-heterogeneity/) | "Let's think step by step" 应该不加验证地当作默认前缀吗？ | **不应该**。收益取决于任务类型和模型规模，32B 以下多数变体退化 | Fig 2, Fig 3 |
| ② 前缀的状态初始化效应 | [`02-state-divergence/`](02-state-divergence/) | 只改 prompt 前缀，模型内部会发生什么？ | 三种前缀 → 三条完全不同的隐藏状态轨迹。前缀不是"编码解题步骤"，而是**设定计算动力学的初始条件** | Fig 4 |
| ③ term-first 跨题验证 | [`03-logit-lens/`](03-logit-lens/) | 实验②的单题发现稳定吗？term-first 属于哪种轨迹？ | 延迟承诺型在 5 题上**精确共享 L20 收敛层**；term-first 在 4/5 题上与之吻合 | Fig 5 |

### 论证逻辑链

```
案例 1（实验①）：CoT 不是通用增强器 → 排除第一个盲从假设
    │
    ├── "既然 CoT 不是万能的，什么在起作用？"
    │
案例 2（实验②+③）：提示前缀的真正作用是隐藏状态初始化
    │  不是编码解题步骤，不是反思标记的信号
    │
    └── "理解了机制，就能针对性地设计更轻更好的策略"
         → term-first 用 12 token 达到甚至超过重 CoT 的效果
```

## 每个实验文件夹的结构

```
<experiment>/
├── README.md     实验卡（数据溯源表 + 复现命令 + 信任级声明）
├── code/         复现脚本
└── data/         复现数据（CSV / JSON / NPY）
```

## 与论文图的对应

| 论文图 | 实验 | 脚本 | 入口数据 |
|--------|------|------|---------|
| Fig 2 (Task heterogeneity) | ① | `evidence/01/code/Fig2_TaskHeterogeneity.py` | `evidence/01/data/fig1_task_heterogeneity_scores.csv` + `evidence/01/data/aime2024_qwen25_3b_n64_16k_scores/` |
| Fig 3 (Model scale)        | ① | `evidence/01/code/Fig3_ScalingLaws.py` | `evidence/01/data/fig2_model_scale_scores.csv` |
| Fig 4 (Reflection scatter) | ② | `evidence/02/code/Fig4_ReflectionScatter.py` | `evidence/02/data/reflection_{1p5b,7b,8b}_real.json` |
| Fig 5 (Logit Lens 4 cond)  | ③ | `evidence/03/code/Fig5_LogitLens.py` | `evidence/03/data/logit_lens_5q_4cond.json` |

> 论文 Fig 1 是概念示意图，无对应脚本。

## 信任级声明

- **Fig 2/3 (实验①)**：基于 OpenCompass 跑 4 个数据集 × 5 个模型规模 × 15 个 prompt 变体的可复现数据。CSV 是直接输出。
- **Fig 4 (实验②)**：基于 DeepSeek-R1-Distill-Qwen-1.5B 在 3 个模型 × AIME 2024 的 reflection-marker counts，prompt-level aggregation。JSON 是直接输出。
- **Fig 5 (实验③)**：基于 Logit Lens 在 5 题 × 4 条件的概率轨迹。JSON 是直接输出。
