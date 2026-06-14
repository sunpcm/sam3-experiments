# Exemplar-Guided Product Segmentation Demo

一个最小可运行的计算机视觉 demo：用目标商品的多张参考图，在生产图里找出同款帽子或鞋子的候选实例，并输出 mask、bbox、相似度、crop 和 debug 叠图。

当前版本重点是工程骨架和端到端跑通：

- `segmenter` 和 `matcher` 解耦。
- `SAM3Segmenter` 预留了 SAM 3 接口。
- 默认 `FallbackSegmenter` 不依赖大模型，会生成候选 mask。
- 默认 matcher 使用轻量 `color_hist` embedding，后续可替换成 DINOv2/CLIP 等视觉特征。

## 目录结构

按下面方式放数据：

```text
data/
  references/
    ref_front.jpg
    ref_side.jpg
    ref_angle.jpg
    # or:
    <target_id>/
      ref_front.jpg
      ref_side.jpg
      ref_angle.jpg
  production/
    image_001.jpg
    image_002.jpg
outputs/
  masks/
  crops/
  visualizations/
  results.json
```

如果参考图直接放在 `data/references/` 根目录，可以不显式指定 `target_id`。如果使用 `data/references/<target_id>/` 目录，也可以不指定 `target_id`，前提是只有一个非隐藏子目录；如果有多个目标商品目录，需要在配置或命令行指定。

Jupyter 自动生成的 `.ipynb_checkpoints` 会被代码忽略。

## 安装

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

默认读取 `configs/demo.yaml`：

```bash
python3 run_demo.py
```

指定目标商品和阈值：

```bash
python3 run_demo.py --target-id black_cap --threshold 0.78
```

## Notebook 实验

也可以直接打开 notebook 做交互式实验：

```text
notebooks/demo.ipynb
```

Notebook 会调用同一套 `exemplar_segment_demo` 模块，并提供 reference 预览、production 预览、pipeline 运行、debug visualization、crop 和 `results.json` 摘要展示。常用实验参数在 notebook 的第 2 节，包括 `TARGET_ID`、`THRESHOLD` 和 `MAX_PREVIEW`。

## 配置

主要配置在 `configs/demo.yaml`：

```yaml
data:
  references_dir: data/references
  production_dir: data/production
  outputs_dir: outputs
  target_id: null

segmenter:
  name: fallback

matcher:
  embedding_model: color_hist
  score_aggregation: topk_average
  top_k: 3
  threshold: 0.72
  nms_iou_threshold: 0.55
```

调参建议：

- 误检多：提高 `matcher.threshold`，例如 `0.78` 或 `0.82`。
- 漏检多：降低 `matcher.threshold`，例如 `0.62` 到 `0.70`。
- 候选太少：调低 `segmenter.min_area_ratio`，或打开/增加 `grid_sizes`。
- 候选太碎/太多：调高 `segmenter.min_area_ratio`，降低 `segmenter.max_candidates`。
- 同一实例重复输出：调低 `matcher.nms_iou_threshold`。

## 输出

每张生产图都会写入一条结果，即使没有检测到实例也会记录空数组：

```json
{
  "image": "data/production/example.jpg",
  "mask_png": "outputs/masks/example.png",
  "visualization": "outputs/visualizations/example.jpg",
  "instances": []
}
```

有实例时，每个实例包含：

- `bbox_xyxy`: `[x1, y1, x2, y2]`
- `similarity`: 与 reference embeddings 的相似度分数
- `segmenter_score`: 分割器候选分数
- `segmenter_source`: 候选来源，方便 debug
- `crop`: 候选裁剪图路径

## 接入 SAM 3

当前 `exemplar_segment_demo/segmenters/sam3.py` 是适配器占位。接入本地 SAM 3 时，只需要让它实现：

```python
def segment(self, image_rgb: np.ndarray) -> list[CandidateMask]:
    ...
```

其中 `CandidateMask` 包含：

- `mask`: `HxW` bool array
- `bbox`: `(x1, y1, x2, y2)`
- `score`: segmenter 置信度
- `source`: 调试字符串

然后把配置改为：

```yaml
segmenter:
  name: sam3
```

## 接入 DINOv2

当前 matcher 的最小实现是 `color_hist`，位置在 `exemplar_segment_demo/matcher.py`。替换方式：

1. 在 `ExemplarMatcher.__post_init__` 中允许新的 `embedding_model` 名字。
2. 在 `embed_crop()` 中调用 DINOv2/CLIP/image encoder。
3. 返回已经 L2 normalize 的一维 `np.ndarray`。

pipeline 其余部分无需修改。
