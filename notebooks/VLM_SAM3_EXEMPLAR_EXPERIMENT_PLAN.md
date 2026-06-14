# VLM + SAM3 商品示例诊断实验

## 目标

构建一个用于多类别商品 exemplar 分割的诊断型 notebook：

```text
reference target folders
  -> RGBA alpha clean crops
  -> VLM category + prompt generation
  -> SAM3 text prompt candidate masks
  -> pluggable embedding ranking against each target
  -> debug artifacts and ranking files
```

第一版刻意只做诊断用途，不承诺生产级阈值或最终 mask。它主要用来回答下面几个问题：

- VLM 是否正确描述了目标商品？
- SAM3 是否召回了看起来合理的候选区域？
- DINOv2 / DINOv3 / SigLIP 是否把同风格候选排在了干扰项前面？

## 数据布局

推荐结构如下：

```text
data/
  references/
    target_shoe_001/
      ref_1.png
      ref_2.png
    target_cap_001/
      ref_1.png
      ref_2.png
    target_bag_001/
      ref_1.png
      ref_2.png
  production/
    image_001.png
```

每个 `target_id` 目录里应该放同一个商品/SKU 的多张参考图，不要把无关商品混在同一个目标目录里。

Notebook 也提供了根目录参考图的诊断兜底逻辑，但这只适合快速迁移。多类别实验应该使用按目标拆分的目录。

## VLM 接口

Notebook 使用 OpenAI Python SDK，并连接一个兼容 OpenAI 的自定义端点：

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["VLM_API_KEY"],
    base_url=os.environ["VLM_BASE_URL"],
)
VLM_MODEL = os.environ["VLM_MODEL"]
```

需要的环境变量如下：

```bash
export VLM_BASE_URL="https://your-openai-compatible-endpoint"
export VLM_API_KEY="..."
export VLM_MODEL="..."
```

VLM 会为每个目标返回一个 JSON 对象：

```json
{
  "target_id": "target_cap_001",
  "category": "baseball cap",
  "prompts": ["baseball cap", "cap", "hat"],
  "negative_prompts": ["hair", "head", "face"],
  "attributes": ["navy blue", "curved brim"]
}
```

Prompt JSON 会缓存到：

```text
outputs/vlm_sam3_diagnostic/vlm_prompts/<target_id>.json
```

如果缓存已经存在，Notebook 可以直接继续执行，而不需要再次调用 VLM。这支持 mock 和手工 prompt 实验。

## 流程

1. 环境自检：
   - Python 可执行文件
   - PyTorch / CUDA
   - SAM3 仓库和 BPE 文件
   - VLM 环境变量
2. 发现参考图：
   - 忽略 `.ipynb_checkpoints` 之类的隐藏路径
   - 将 `data/references/` 下每个非隐藏子目录视为一个目标
3. 参考图清理裁剪：
   - 如果存在 RGBA alpha，则按 `alpha > threshold` 裁剪
   - 将透明背景合成到白底上
   - 保存清理后的裁剪图供检查
4. VLM prompt 生成：
   - 把清理后的参考图发送给 VLM
   - 要求只返回 JSON
   - 缓存并展示 prompts
5. SAM3 候选生成：
   - 针对每张生产图和目标 prompt
   - 运行 SAM3 文本提示
   - 保存所有候选，而不只是最终匹配
6. Embedding 排序：
   - 为每个目标构建 reference embedding
   - 对每个 SAM3 候选的 masked crop 做 embedding
   - 按每个目标分别用 cosine similarity 排序候选

## DINOv3 和 LocateAnything 评估

DINOv3 适合作为当前 DINOv2 matcher 的升级候选，但它主要改善的是候选排序，不直接解决候选召回为空的问题。DINOv3 的核心价值在于更好的 dense features、局部对应和跨视角 visual embedding；因此本实验把 embedding 模型做成可配置项，允许在同一批 SAM3 candidates 上对比 `facebook/dinov2-base`、DINOv3 或 SigLIP。

推荐对比指标：

- top-1 是否是目标商品
- top-5 是否包含目标商品
- 目标商品与干扰项的 similarity margin
- 鞋、帽、包、衣服等不同类别是否需要不同阈值

LocateAnything 更可能改善召回阶段。它的定位是 vision-language grounding / detection，适合根据 VLM prompt 先输出 bbox 或 point，再交给 SAM2/SAM3 生成 mask。它不应替代 SAM mask，而应作为新的 candidate generator：

```text
VLM prompts
  -> LocateAnything bbox/point grounding
  -> SAM box/point prompted mask
  -> DINOv2/DINOv3/SigLIP ranking
```

由于 LocateAnything 较新，代码和权重生态可能还不稳定，第一版 notebook 只预留 `GroundingCandidateGenerator` 思路，不把它作为硬依赖。短期优先级是先完成 VLM + SAM3 text prompt 诊断链路，再把 DINOv3 作为 matcher 对照实验；如果 SAM3 text prompt 仍然召回差，再接入 LocateAnything bbox -> SAM mask 分支。

## 输出

```text
outputs/vlm_sam3_diagnostic/
  reference_clean/<target_id>/*.jpg
  vlm_prompts/<target_id>.json
  candidates/<image_stem>/<target_id>/
    *_bbox.jpg
    *_masked_white.jpg
    *_overlay.jpg
  rankings/
    candidate_ranking.csv
    candidate_ranking.json
```

## 验收检查

- Notebook JSON 有效，所有代码单元都能解析。
- 隐藏的 checkpoint 文件会被忽略。
- VLM prompt 缓存可以手工 mock。
- 每个目标都有独立的 prompts 和独立的 reference embedding。
- 即使后续阈值下没有候选通过，Notebook 仍然能展示候选和排序结果。
