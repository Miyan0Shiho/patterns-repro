# 00 — Fig 1 Conceptual framework

→ 论文 Fig 1（概念示意图）

**一句话**：一张结构化的"AI 研究中过度因果化"批判示意图，分三段描述：① 热门策略被不加验证地接受 ② 本工作的三层检验（Case 1/2/Mechanism） ③ take-away。

## 这个图做什么

- **不**展示具体实验数据——它是论文的"导览图"
- 引导读者从"AI 研究的盲从现象"走向"本工作的检验方法"
- 与正文第 1 段（Bigger picture）配合，作为"一图说清论文做什么"

## 文件

| 文件 | 用途 |
|------|------|
| `data/ConceptualFramework.svg` | SVG 源（论文图的母版）|
| `data/icons/*.svg` + `*.png` | 图中嵌入的图标资源（17 个）|
| `code/build_concept_image.py` | Python 脚本：用 matplotlib 重新生成 PNG / SVG（4 格式）|
| `figures/Fig1_ConceptualFramework.{png,jpg,tiff,svg}` | 投稿用 4 格式输出 |

## 复现

```bash
cd code/
python3 build_concept_image.py
# 输出到 ../figures/Fig1_ConceptualFramework.{png,jpg,tiff,svg}
```

> **注**：脚本是 SVG 源（`data/ConceptualFramework.svg`）的 matplotlib 重渲染器。
> 如果只是**修改文字/箭头流向**，直接编辑 `data/ConceptualFramework.svg` 更简单。
> 如果**改版式/字体/颜色**，跑 `build_concept_image.py` 重新生成 PNG 系列。
> 脚本是"保险"，不是日常使用工具。

## 设计约束

- 12.333 × 5.36 in，2.30:1（与正文 1 栏宽匹配）
- 30 秒可读：每个组件 = 一个 claim
- 配色与论文其他图（Fig 2-5）一致（用 `scripts/plot_config.py` 中的 Okabe-Ito 色板）
- 字体：Arial / Helvetica / DejaVu Sans（与正文一致）

## 信任级

- **最高**：Fig 1 是导览图，论文其他实验不需要依赖它
- 修改 Fig 1 不影响 Fig 2-5 的任何数值
- 唯一约束：与正文中"Bigger picture"段（`draft.docx` ¶11-12）措辞一致
