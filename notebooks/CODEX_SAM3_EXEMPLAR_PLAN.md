# Codex 任务计划：修正 SAM3 以图/参考图切生产图 Demo

## 背景

当前 notebook 的问题不是数据路径，而是评估链路不纯：用户以为在评估 SAM3 以图切图，实际可能跑了 fallback segmenter + color_hist matcher。需要把 demo 改成可诊断、可替换、禁止静默降级的工程结构。

## 目标

实现一个可靠 pipeline：

```text
reference RGBA 商品图
  -> alpha clean crop
  -> SAM3 召回 production image 中所有可能鞋/帽候选 mask
  -> DINOv2/SigLIP 做同款过滤
  -> 输出 mask、crop、overlay、ranking、debug candidates
```

## 阶段 1：禁止静默 fallback

1. 检查 `exemplar_segment_demo.segmenters.build_segmenter()`、`pipeline.run_demo()`、`build_matcher()`。
2. 增加配置项：

```yaml
segmenter:
  type: sam3_text
  allow_fallback: false
matcher:
  type: dinov2
```

3. 如果指定 SAM3 但 import、权重加载、推理失败，必须 `raise RuntimeError`，不要自动 fallback。
4. notebook 输出真实 backend：segmenter class、matcher class、device、checkpoint、fallback enabled。

## 阶段 2：Reference RGBA alpha clean crop

1. 新增 `load_reference_object_crop(path, alpha_threshold=10, pad_ratio=0.08)`。
2. 如果图片有 alpha：用 `alpha > threshold` 找主体 bbox，加 padding，合成白底 RGB。
3. matcher embedding 必须使用 clean crop，不要直接 `read_image_rgb()`。
4. notebook 展示原 reference、alpha bbox、clean crop、alpha_area_ratio。

## 阶段 3：SAM3 text sanity check

1. 新增独立 notebook section，不走原 pipeline。
2. 直接调用：

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
```

3. 显式传入 BPE：

```python
bpe_path='/home/sunpcm/workspace/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'
```

4. 在 production image 上跑 prompt：`shoe`、`shoes`、`sneaker`、`footwear`。
5. 保存所有 masks、boxes、scores、overlay 到 `outputs/sam3_sanity/`。
6. 任何异常都直接显示 traceback，不 fallback。

## 阶段 4：保存全部 candidates，不只保存 matches

1. 保存每个 candidate：bbox crop、masked white-bg crop、overlay、bbox、area、aspect ratio、SAM3 score。
2. 即使 matches=0，也必须展示 top candidates。
3. 输出：

```text
outputs/debug/candidates/<image_stem>/000_prompt_shoe_score_0.921_bbox.jpg
outputs/debug/candidates/<image_stem>/000_prompt_shoe_score_0.921_masked_white.jpg
outputs/debug/candidates/<image_stem>/000_prompt_shoe_score_0.921_overlay.jpg
```

## 阶段 5：DINOv2 matcher 替换 color_hist

1. 新增 matcher：

```yaml
matcher:
  type: dinov2
  model_name: facebook/dinov2-base
  threshold: 0.55
  device: cuda
```

2. reference：clean crop -> DINOv2 embedding -> 多图平均后 normalize。
3. candidate：mask 白底 crop -> DINOv2 embedding -> cosine similarity。
4. 输出完整 ranking：candidate_id、prompt、sam3_score、dino_similarity、passed、bbox、paths。
5. 保留 color_hist 但不要默认使用。

## 阶段 6：SAM3 visual/exemplar backend

1. 阅读本机 `/home/sunpcm/workspace/sam3/examples/sam3_image_predictor_example.ipynb`。
2. 找到 visual box / visual exemplar 的官方调用方式。
3. 新增 segmenter：

```yaml
segmenter:
  type: sam3_exemplar
  text_prompt: shoe
  allow_fallback: false
```

4. 如果 visual exemplar API 一次接不稳，先保留 `sam3_text + DINOv2` 作为主链路。
5. 输出对比：fallback candidates、sam3_text candidates、sam3_exemplar candidates。

## 验收标准

1. notebook 明确打印当前是否使用 SAM3；不能出现用户不知道的 fallback。
2. reference clean crop 可视化正确，透明背景不参与 embedding。
3. SAM3 text prompt 能输出候选 mask debug 图。
4. DINOv2 ranking 里，完整同款鞋应显著高于背包/衣服。
5. 即使无匹配，也能查看 rejected candidates，定位是召回问题还是匹配问题。
6. 所有输出落盘，方便人工检查与阈值标定。
