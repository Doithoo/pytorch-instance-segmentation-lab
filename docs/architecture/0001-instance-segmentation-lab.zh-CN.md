# ADR-0001：PyTorch 实例分割学习实验室实施规格

- 状态：v0.1.0 已实施；评估和数据划分决策由 [ADR 0002](0002-evaluation-and-splits.zh-CN.md) 修订
- 目标目录：`pytorch-instance-segmentation-lab`
- 参考项目：`pytorch-object-detection-lab`
- 项目类型：面向初学者的、可复现的 PyTorch 实例分割学习项目
- 首个可运行数据集：Penn-Fudan Pedestrian
- 首个主模型：torchvision Mask R-CNN
- 正式完整训练平台：Kaggle GPU（T4 或更新的兼容 NVIDIA GPU）
- 语言策略：代码和 CLI 使用英文，教程和关键说明提供中英文版本

## 1. 目标与边界

### 1.1 必须实现的目标

项目必须让学习者在不阅读全部源码的情况下完成下面的闭环：

```text
download -> prepare -> inspect -> dry-run -> train -> evaluate -> predict
```

闭环完成后，学习者应能解释并实际观察到：

1. 一张图片中可以有多个同类实例。
2. 一个实例同时由 `label`、`box` 和二值 `mask` 描述。
3. 实例分割模型训练时返回分类、边界框和掩码等多项损失。
4. 推理结果是数量可变的实例列表，而不是固定形状的类别图。
5. 预测掩码必须经过阈值化、尺寸还原和可视化，不能把 logits 当成最终掩码。
6. 评估应同时观察 bbox AP 和 mask AP，并以 mask 指标作为实例分割主指标。
7. 训练、验证、测试的成员关系由固定 manifest 决定，并记录到实验产物。
8. 不配置本地 CUDA，也能通过仓库提供的 Kaggle runner 完成全部训练、选模、测试评估和产物下载。

### 1.2 第一版范围

第一版只解决 **class-aware instance segmentation**：

- 默认类别：`background`、`person`。
- 默认数据：Penn-Fudan Pedestrian。
- 默认模型：Mask R-CNN with ResNet-50 FPN。
- 默认权重：`none` 和 torchvision 提供的 COCO 预训练权重。
- 默认评估：COCO-style mask AP，至少输出 `mask_map`、`mask_map_50`、`mask_map_75`；同时输出 bbox 指标。
- 默认推理：单张图片，保存机器可读掩码、彩色掩码和叠加图。
- 本地默认路径：CPU 完成数据检查、测试和真实 Mask R-CNN dry-run，不要求本地 CUDA。
- 正式完整训练：使用 Kaggle T4 或更新 GPU、全部 Penn-Fudan 训练样本、固定 20 epoch 参考配置和 COCO 预训练权重。
- Kaggle runner：从源码快照开始自动执行下载、准备、检查、dry-run、完整训练、best checkpoint 选取、test 评估和预测。

### 1.3 明确不在第一版范围内的内容

以下内容必须留到后续 milestone，不得为了“看起来完整”而在第一版中伪实现：

- panoptic segmentation。
- instance tracking、视频推理和实时摄像头。
- 多标签实例、开放词汇分割和文本提示。
- 自己从零实现完整 Mask R-CNN 的 RPN、ROIAlign 和 NMS。
- 复杂增强库、分布式训练和多机训练。
- 直接支持所有 YOLO-seg、LabelMe、CVAT 变体。
- 把语义分割的单张类别 mask 直接当成实例标注。

第一版要保留 provider 和模型 registry 扩展点，但只实现经过测试的 provider 和模型。

## 2. 与现有项目的关系

`pytorch-object-detection-lab` 是边界框检测项目；本项目是独立仓库和独立 Python 包，不修改检测项目，也不复用其包名。

可以复用其工程习惯：

- `src/` layout。
- YAML 配置加 `--set` 覆盖。
- registry + 显式外部工厂。
- manifest 固定数据成员。
- `best.pt` / `last.pt` checkpoint。
- 单元、集成、CLI 和端到端测试。
- 英文和中文 README、教程、指南。
- Kaggle runner 内嵌源码快照、GPU preflight、JSON 心跳、完整训练和 recorded-run 产物。

不能直接复用其任务契约：

- 检测 target 只有 boxes/labels；本项目必须额外提供 masks/area/iscrowd。
- 检测指标不能代替 mask AP。
- 检测可视化不能代替实例级颜色掩码和重叠关系展示。

目录中的 `pytorch-image-segmentation-lab` 是语义分割项目，也保持独立；它的 `[H, W]` 单张标签图契约不能直接用于本项目。

## 3. 目标目录架构

实施完成前，目录和文件职责应稳定在下面的形状。当前 scaffold 已创建目录；带有 `*.py`、YAML 和文档文件的实现将在后续阶段填充。

```text
pytorch-instance-segmentation-lab/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug.yml
│   │   ├── learning-question.yml
│   │   └── config.yml
│   ├── dependabot.yml
│   └── workflows/
│       └── ci.yml
├── artifacts/
│   └── .gitkeep                         # 本地训练和评测产物，不提交模型
├── configs/
│   ├── README.md
│   ├── README.zh-CN.md
│   ├── learning_minimal.yaml            # CPU/小样本/少轮数，必须可运行
│   ├── reference_maskrcnn.yaml          # 记录参考运行
│   ├── maskrcnn_resnet50_fpn.yaml       # 模型对照配置
│   ├── custom_dataset_example.yaml
│   └── custom_model_example.yaml
├── data/
│   ├── raw/.gitkeep                     # 下载数据，不提交
│   ├── processed/.gitkeep               # 规范化或缓存数据，不提交
│   └── manifests/
│       ├── .gitkeep
│       ├── dataset.yaml                 # 数据集版本、类别和统计摘要
│       ├── source.yaml                  # 下载 URL、文件哈希和来源
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── docs/
│   ├── README.md / README.zh-CN.md
│   ├── architecture/
│   │   └── 0001-instance-segmentation-lab.md / .zh-CN.md
│   ├── concepts/
│   │   ├── instance-target.md / .zh-CN.md
│   │   ├── maskrcnn-flow.md / .zh-CN.md
│   │   └── configuration-flow.md / .zh-CN.md
│   ├── guides/
│   │   ├── using-your-data.md / .zh-CN.md
│   │   ├── using-models.md / .zh-CN.md
│   │   ├── experiments.md / .zh-CN.md
│   │   ├── kaggle.md / .zh-CN.md
│   │   └── troubleshooting.md / .zh-CN.md
│   ├── reference/
│   │   ├── dataset-format.md / .zh-CN.md
│   │   ├── config-reference.md / .zh-CN.md
│   │   ├── checkpoint-schema.md / .zh-CN.md
│   │   ├── metrics.md / .zh-CN.md
│   │   ├── model-zoo.md / .zh-CN.md
│   │   └── penn-fudan.md / .zh-CN.md
│   ├── tutorial/
│   │   ├── README.md / .zh-CN.md
│   │   ├── 00-basics.md / .zh-CN.md
│   │   ├── 01-environment.md / .zh-CN.md
│   │   ├── 02-data-and-instances.md / .zh-CN.md
│   │   ├── 03-maskrcnn.md / .zh-CN.md
│   │   ├── 04-training.md / .zh-CN.md
│   │   └── 05-evaluation-and-inference.md / .zh-CN.md
│   └── recorded-run/
│       ├── README.md / .zh-CN.md
│       ├── config.yaml                 # Kaggle 实际解析配置，不是未解析模板
│       ├── run.yaml                    # 环境、GPU、版本、manifest 和源码摘要
│       ├── metrics.csv                 # 全部 20 epoch 训练与验证指标
│       ├── kaggle-run-summary.json     # 机器可读运行总览
│       ├── evaluation/                 # test 指标、逐图结果和代表图
│       └── kaggle/
│           ├── kernel-metadata.json
│           └── run_kaggle.py           # 由构建脚本生成并内嵌源码快照
├── examples/
│   ├── README.md / README.zh-CN.md
│   ├── 01_instance_target.py
│   ├── 02_mask_to_instances.py
│   ├── 03_detection_collate.py
│   ├── 04_minimal_training_loop.py
│   ├── 05_checkpoint_prediction.py
│   └── extensions/
│       ├── __init__.py
│       ├── my_dataset.py
│       └── my_segmenter.py
├── scripts/
│   ├── README.md / README.zh-CN.md
│   ├── __init__.py
│   ├── download_data.py
│   ├── preview_dataset.py
│   ├── prepare_data.py
│   ├── plot_metrics.py
│   ├── kaggle_runner.py               # 可维护的 runner 模板
│   └── build_kaggle_runner.py         # 确定性打包源码并生成提交脚本
├── src/instance_segmenter/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── extensions.py
│   ├── preflight.py
│   ├── py.typed
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py                  # InstanceTarget、ClassDefinition、LabelSchema
│   │   ├── dataset.py                 # Dataset 协议和通用实现
│   │   ├── collate.py                 # 可变数量实例的 batch 组合
│   │   ├── masks.py                   # instance-id mask 解析和校验
│   │   ├── transforms.py              # image/target 同步变换
│   │   ├── manifest.py                # 固定划分和哈希
│   │   ├── inspection.py              # 数据统计和坏样本报告
│   │   ├── providers.py               # provider 协议和工厂
│   │   ├── registry.py                # provider 注册表
│   │   ├── pennfudan.py               # Penn-Fudan 解析器
│   │   └── coco.py                    # COCO instances 适配器，第二阶段
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spec.py                    # 模型元数据和构造协议
│   │   ├── registry.py
│   │   ├── torchvision_models.py      # Mask R-CNN 构造器
│   │   └── extensions.py              # 外部模型工厂加载
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── train.py
│   │   ├── checkpoint.py
│   │   └── precision.py               # AMP 和 device 能力判断
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # bbox + mask COCO 指标
│   │   ├── evaluate.py
│   │   ├── comparison.py              # run 对比
│   │   └── visualization.py            # 实例级 overlay、错误案例
│   └── inference/
│       ├── __init__.py
│       └── predictor.py
├── tests/
│   ├── README.md / README.zh-CN.md
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── synthetic_instances.py
│   │   ├── external_datasets.py
│   │   └── external_models.py
│   ├── test_label_schema.py
│   ├── test_masks.py
│   ├── test_transforms.py
│   ├── test_dataset.py
│   ├── test_manifest.py
│   ├── test_data_inspection.py
│   ├── test_data_registry.py
│   ├── test_model_registry.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_checkpoint.py
│   ├── test_metrics.py
│   ├── test_visualization.py
│   ├── test_inference.py
│   ├── test_evaluation.py
│   ├── test_cli.py
│   ├── test_examples.py
│   ├── test_end_to_end.py
│   ├── test_packaging.py
│   └── test_kaggle_runner.py
├── CONTRIBUTING.md / CONTRIBUTING.zh-CN.md
├── LICENSE
├── MANIFEST.in
├── Makefile
├── README.md
├── README.zh-CN.md
├── pyproject.toml
└── uv.lock
```

### 3.1 技术基线

`pyproject.toml` 按以下约定实现：

- distribution name：`pytorch-instance-segmentation-lab`。
- import package：`instance_segmenter`。
- console script：`instance-segment = instance_segmenter.cli:main`。
- Python：`>=3.10,<3.13`，CI 覆盖 3.10、3.11、3.12。
- 核心依赖：`torch>=2.6,<3`、`torchvision>=0.21,<1`、`torchmetrics>=1.4,<2`、`pycocotools>=2.0.7,<3`、`numpy>=1.24,<3`、`pillow>=12,<13`、`pyyaml>=6,<7`。
- 开发依赖：`pytest`、`ruff`、`mypy`、`types-PyYAML`、`pre-commit`、`build`、`twine`、`matplotlib`。
- 构建后端：setuptools，使用 `src` layout，并发布 `py.typed` 和示例配置。
- 锁文件：使用 `uv.lock`；README 和 CI 使用 `uv sync --locked --extra dev`。

若实现时当前平台对上述 PyTorch 下限不兼容，可以在一次明确的依赖兼容提交中调整，但必须同步 `pyproject.toml`、CI、README 和 lockfile，不能只改其中一个。Matplotlib 只用于脚本和绘图路径；核心数据解析和训练模块不得在 import 时强制初始化绘图后端。

## 4. 核心数据契约

### 4.1 类别方案

所有数据 provider、模型、评估器和可视化必须共享一个 `LabelSchema`，不能各自维护类别编号。

第一版默认：

```yaml
classes:
  - id: 0
    name: background
    color: [32, 32, 32]
  - id: 1
    name: person
    color: [36, 180, 99]
ignore_index: 255
```

实例 `labels` 只允许出现可训练类别 `1..num_classes-1`；`background=0` 仅用于 schema 和调试，不作为实例输出。类别 id 必须从 0 连续排列，`ignore_index` 不得与类别 id 冲突。

### 4.2 Dataset 输出

`Dataset.__getitem__` 必须返回：

```python
image: torch.Tensor                  # [C, H, W], float32, normally [0, 1]
target: dict[str, torch.Tensor]
```

target 的规范字段：

```python
{
    "boxes": torch.FloatTensor[N, 4],       # xyxy, 0 <= x1 < x2 <= W
    "labels": torch.LongTensor[N],          # foreground class ids
    "masks": torch.BoolTensor[N, H, W],     # one binary mask per instance
    "image_id": torch.LongTensor[1],
    "area": torch.FloatTensor[N],
    "iscrowd": torch.LongTensor[N],
}
```

约束：

- 所有实例字段的第一维必须是同一个 `N`。
- `N=0` 必须有明确行为并经过测试，不能用伪造的零框代替空标注。
- `boxes` 必须由 mask 的非零区域计算并校验；不能信任越界或反向坐标。
- `area` 默认使用 mask 像素面积；评估适配 COCO 时再转换为所需格式。
- Penn-Fudan 第一版 `iscrowd` 全为 0。
- `masks` 必须保持实例独立，即使两个实例属于同一个类别也不能合并。
- Dataset 不应在 `__getitem__` 中随机改变 image 与 mask 的几何关系。

### 4.3 Batch 输出

实例数量和图像尺寸都可能不同，因此 DataLoader 使用：

```python
images: list[Tensor]
targets: list[dict[str, Tensor]]
```

`collate_fn` 只负责把样本组成列表，不做 padding、不偷偷堆叠 masks。模型适配器负责把列表传给 torchvision 模型。

## 5. 数据处理设计

### 5.1 Penn-Fudan provider

`download_data.py` 必须：

1. 下载固定上游压缩包到 `data/raw/downloads/`。
2. 校验 HTTP 下载完整性和 SHA-256；已存在且哈希正确时不重复下载。
3. 解压到明确的版本目录，不覆盖用户已有文件。
4. 把来源 URL、文件大小、SHA-256、下载时间和数据版本写入 `data/manifests/source.yaml`。

`prepare_data.py` 必须：

1. 找到同名 PNG image 与 pedestrian mask。
2. 从原始 mask 解析出每个独立 pedestrian instance，生成二值 mask 列表。
3. 对每个实例计算 bbox、area，并移除空 mask。
4. 检查 mask 与 image 的宽高完全一致。
5. 用稳定 hash 或固定 seed 生成 train/valid/test，保证同一输入得到同一划分。
6. 生成 CSV manifest 和 `dataset.yaml`，记录样本数、尺寸范围、实例数范围、类别和文件哈希。
7. 运行完整性校验：缺图、缺 mask、重复 image id、坏 PNG、空 split、越界 bbox、重叠异常都要给出可定位错误。

第一版默认划分固定为 `80% train / 10% valid / 10% test`，按稳定排序后切分；Penn-Fudan 170 张图片的预期数量是 `136 / 17 / 17`。不要在每次命令运行时重新随机切分。生成后提交 manifest；发布 runner 必须复用或逐字节验证这组 manifest。之后若采用其他权威 split，必须先更新 ADR，并在文档中明确来源和迁移方式。

### 5.2 mask 解析

`data/masks.py` 负责纯函数：

- `decode_instance_mask(path) -> list[BoolTensor]`。
- `masks_to_boxes(masks) -> FloatTensor[N, 4]`。
- `validate_instance_masks(masks, height, width, schema) -> None`。
- `remove_empty_instances(...)`。

解析规则必须覆盖：背景值、实例 id、全空 mask、非连续实例 id、图像边缘实例和两个实例相邻的情况。不能通过单纯的“所有前景像素合成一个 mask”实现实例分割。

### 5.3 成对变换

`data/transforms.py` 的几何变换必须同步作用于 image 和所有 instance masks，并在变换后重新计算 boxes/area：

- resize。
- horizontal flip。
- optional crop，第一版可先只实现安全的 resize/flip。
- normalize 只作用于 image，不作用于 mask。

变换必须使用最近邻插值处理 mask，禁止双线性插值生成新的 mask 类别值。变换结果必须重新过滤空实例并保持字段长度一致。

## 6. 模型设计

### 6.1 Registry

`ModelSpec` 至少包含：

- 稳定名称。
- provider 名称。
- 构造工厂。
- 是否支持预训练权重。
- 依赖和安装提示。
- 输出适配说明。
- 输入尺寸、类别数和权重限制。

第一版注册：

1. `maskrcnn_resnet50_fpn`：主教学模型，支持 `weights=none` 或 COCO 权重。
2. `maskrcnn_resnet50_fpn_v2`：可选对照模型，仅在当前 torchvision 版本可用时注册。

快速单元测试使用 `tests/fixtures/external_models.py` 中遵守同一输入/输出契约的轻量测试替身，不把测试替身注册成面向学习者的模型，也不声称它是 Mask R-CNN。`learning_minimal.yaml --dry-run` 仍须执行真正的 torchvision Mask R-CNN，但通过 batch size 1、小图、单 batch 和不下载权重控制成本。

模型构造器必须接收 `num_classes`，不能把 `2` 写死。使用预训练权重时，必须正确替换 box predictor 和 mask predictor 的类别维度，并在文档中解释哪些 backbone 权重被保留。

### 6.2 训练/推理接口

模型适配层统一成：

```python
losses: dict[str, Tensor] = model(images, targets)  # train mode
outputs: list[dict[str, Tensor]] = model(images)    # eval mode
```

训练 loss 至少记录：

- `loss_classifier`。
- `loss_box_reg`。
- `loss_mask`。
- `loss_objectness`。
- `loss_rpn_box_reg`。
- `loss_total`。

推理 output 必须规范为：

```python
{
    "boxes": FloatTensor[N, 4],
    "labels": LongTensor[N],
    "scores": FloatTensor[N],
    "masks": FloatTensor[N, 1, H, W],  # probability masks before threshold
}
```

`predictor.py` 决定 score threshold、mask threshold 和输出尺寸；模型本身不写死展示逻辑。保存结果时同时提供原始概率 mask 或阈值 mask 的清晰说明。

## 7. 配置和命令行

### 7.1 配置字段

YAML 配置至少包含：

```yaml
run:
  name: learning-minimal
  seed: 42
  output_dir: artifacts

data:
  provider: pennfudan
  root: data/raw
  manifest_dir: data/manifests
  image_size: [128, 128]
  batch_size: 1
  num_workers: 0
  train_limit: null
  valid_limit: null
  test_limit: null

model:
  name: maskrcnn_resnet50_fpn
  weights: none
  num_classes: 2

training:
  epochs: 2
  device: auto
  optimizer: sgd
  learning_rate: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  amp: auto
  grad_clip_norm: null
  score_threshold: 0.5
  mask_threshold: 0.5
```

实际实现可分组调整，但必须在 `show-config` 中输出解析后的完整配置，并拒绝未知字段、负数 epoch、非法类别数、非法权重组合和不存在的 provider。

配置优先级固定为：

```text
代码默认值 < YAML 文件 < --set key=value < 专用 CLI 参数
```

### 7.2 CLI

入口命令建议为 `instance-segment`，避免与其他仓库的 `detect` 和语义分割项目的 `segment` 混淆。

必须支持：

```bash
instance-segment --version
instance-segment list-models
instance-segment list-datasets
instance-segment show-config --config configs/learning_minimal.yaml
instance-segment verify-data --config configs/learning_minimal.yaml
instance-segment prepare-data --config configs/learning_minimal.yaml
instance-segment train --config configs/learning_minimal.yaml --dry-run
instance-segment train --config configs/reference_maskrcnn.yaml --set run.name=my-run
instance-segment evaluate --checkpoint artifacts/my-run/best.pt --split valid
instance-segment evaluate --checkpoint artifacts/my-run/best.pt --split test --plot
instance-segment predict --checkpoint artifacts/my-run/best.pt --image path/to/image.png --output artifacts/prediction
```

命令错误必须返回非零退出码并给出可执行的修复提示。CLI 层只编排，不承载 mask 解析和指标算法。

## 8. 训练、checkpoint 和实验产物

### 8.1 Trainer

`Trainer` 负责：

- seed 和 device 初始化。
- DataLoader 创建。
- train/eval mode 切换。
- loss 聚合和梯度更新。
- 可选 AMP，CPU 上自动关闭或使用安全路径。
- 每 epoch 记录训练 loss、验证 loss 和验证 mask 指标。
- 根据验证集 `mask_map`（即 IoU 0.50:0.95 的 COCO mask AP）选择 `best.pt`，并始终保存 `last.pt`。
- 支持从 `last.pt` 恢复 optimizer、scheduler、epoch、best score 和 RNG 状态。

训练期间不能在测试集上选最优模型，不能把 test 指标写进训练 epoch 的选择逻辑。

### 8.2 checkpoint schema

checkpoint 至少包含：

```python
{
    "format_version": 1,
    "model_name": str,
    "model_state": dict,
    "optimizer_state": dict | None,
    "scheduler_state": dict | None,
    "epoch": int,
    "best_metric": float,
    "label_schema": dict,
    "resolved_config": dict,
    "manifest_hashes": dict,
    "python_version": str,
    "torch_version": str,
    "rng_state": object,
}
```

加载 checkpoint 必须校验 format version、模型名、类别 schema 和必要的 tensor shape；不匹配时给出清晰错误，禁止静默加载。

每次运行目录至少保存：

```text
artifacts/<run-name>/
├── config.yaml
├── environment.txt
├── manifest-hashes.yaml
├── metrics.csv
├── best.pt
├── last.pt
├── evaluation/
│   ├── evaluation.json
│   ├── per_class.csv
│   ├── per_image.csv
│   └── visualizations/
└── predictions/
```

大于仓库合理体积的 checkpoint、原始数据和批量预测不得提交到 Git。

## 9. 评估和可视化

### 9.1 指标

优先使用稳定的现有实现，例如 `torchmetrics.MeanAveragePrecision` 配合 `pycocotools`，不要自行实现 COCO AP 的近似版本。

必须输出：

- `mask_map`，mask IoU 的 COCO AP `0.50:0.95`。
- `mask_map_50`。
- `mask_map_75`。
- `bbox_map`、`bbox_map_50`、`bbox_map_75`。
- `mar_100` 或等价召回指标。
- 每类指标，在第一版至少包含 `person`。
- 每图指标，包含预测实例数、GT 实例数、匹配数和错误类型。

指标实现必须处理：空预测、空 GT、不同尺寸 mask、score threshold、多个实例重叠和 `iscrowd` 字段。应区分“模型没有预测”和“预测被阈值过滤”。

### 9.2 错误分析

可视化至少包括：

1. image + GT boxes/masks。
2. image + predicted boxes/masks/scores。
3. GT 与 prediction 对照 overlay。
4. false positive、false negative 和低 IoU 匹配案例。
5. 每个实例使用稳定、可区分的颜色；同类不同实例不能被合并成同色单 mask。

所有图都要带图例或清晰标注，避免把 alpha overlay 当作 ground-truth 文件。预测命令要保存：

- `mask.png` 或每实例 mask 文件。
- `instances.json`，含 box、label、score、mask 路径或 RLE。
- `overlay.png`。

## 10. 教程与 examples 顺序

examples 必须小而可独立运行，教程再解释它们在完整项目中的位置：

1. `01_instance_target.py`：手工创建两个实例，打印 boxes/labels/masks/area。
2. `02_mask_to_instances.py`：把 instance-id mask 解析为多个二值 mask 并画出 bbox。
3. `03_detection_collate.py`：展示可变实例数量为什么需要 list batch。
4. `04_minimal_training_loop.py`：用 synthetic dataset 完成一次 forward/loss/backward/update。
5. `05_checkpoint_prediction.py`：加载 checkpoint、阈值化 mask、保存 overlay。

教程章节必须覆盖：

- 实例分割与语义分割、目标检测的区别。
- 一个 target 字典每个字段的含义。
- Penn-Fudan 原始 mask 如何变成实例列表。
- Mask R-CNN 的 backbone、RPN、ROI heads 和 mask head。
- 训练时多 loss 与推理时多实例 output 的差异。
- mask AP 与 bbox AP 的差异，以及为什么不能只看 pixel accuracy。
- 自定义同名图片、instance-id mask 数据的接入方式。

## 11. 测试策略

测试不下载真实数据和预训练权重；使用小型 synthetic fixture。需要模型依赖或网络的测试必须显式标记并默认跳过。

必须覆盖：

- schema：类别连续性、颜色、ignore index、序列化。
- mask parser：背景、多个实例、边缘实例、空实例、相邻实例、非法尺寸。
- boxes：坐标顺序、边界、面积和变换后的重算。
- transforms：水平翻转后 image/mask/box 一致，mask 仍为二值。
- dataset/collate：不同 N 和不同图像尺寸，空实例行为。
- manifest：固定划分、排序、哈希和重复样本检测。
- model registry：名称、依赖状态、类别数替换和输出 shape。
- training：loss 字段齐全、一次 optimizer step、dry-run 不泄漏测试集。
- checkpoint：保存、恢复、schema mismatch 和旧版本错误。
- metrics：空预测、空 GT、重复预测、mask/bbox 指标输入。
- visualization：文件生成、尺寸、颜色和不改变原始 mask。
- inference：score/mask threshold、CPU、输出 JSON。
- CLI：help、错误退出码、`--set` 优先级、关键命令 smoke test。
- packaging：`uv build`、安装后 `instance-segment --version`。
- documentation：README 中的命令、路径和文件必须存在。
- Kaggle packaging：源码快照可确定性重建、排除数据/产物/虚拟环境、metadata 可解析、生成 runner 不陈旧。

### 端到端验收

在纯 CPU 和 synthetic fixture 上必须通过：

```bash
uv run instance-segment train --config configs/learning_minimal.yaml --dry-run --device cpu
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

本地真实数据验收至少完成一次短流程，用于在提交前排除明显错误：

```text
download -> prepare -> preview -> dry-run -> short train -> valid evaluate -> predict
```

此外必须在 Kaggle GPU 完成一次正式参考运行：

```text
embedded source -> GPU preflight -> download -> prepare -> inspect -> dry-run
-> full 20-epoch train -> select best on valid -> evaluate test once -> predict -> summary
```

Kaggle 参考运行的 resolved config、manifest hash、源码摘要、全部 epoch metrics、测试指标、运行摘要和样例图进入 `docs/recorded-run/`；不提交下载数据、完整预测缓存或大 checkpoint。

## 12. Kaggle GPU 完整训练规格

Kaggle 不是附加示例，而是第一版推荐的正式训练环境。本地路径负责安装、数据检查、CPU dry-run 和开发测试；Kaggle 路径负责完整训练及可发布结果。用户不需要本地 CUDA，也不需要创建或挂载 Kaggle Dataset。

### 12.1 “完整训练”的严格定义

一次可发布的 Kaggle 参考运行必须同时满足：

1. 使用仓库已提交的固定 Penn-Fudan train/valid/test manifest，要求 split 数量为 `136/17/17`，并记录每个文件的 SHA-256；如需在 Kaggle 重新生成，生成结果必须与内嵌 manifest 逐字节一致。
2. train split 的全部样本参与训练，不设置 `train_limit`，不使用只覆盖部分数据的 sampler。
3. 从 COCO 预训练的 `maskrcnn_resnet50_fpn` 开始，替换 box/mask predictor 以适配 `background + person` 两类。
4. 完成固定的 20 个 epoch；默认不启用 early stopping，诊断运行不得冒充完整运行。
5. 每个 epoch 后只在 valid split 计算 bbox/mask 指标，以 valid `mask_map` 选择 `best.pt`。
6. test split 在训练和选模完成前完全不可见；训练结束后只用 `best.pt` 做一次最终 test 评估。
7. 至少运行一次 checkpoint 单图预测并生成 `instances.json`、实例 mask 和 overlay。
8. Kaggle 任务最终状态为 `COMPLETE`，且 summary 的 `completed_epochs` 明确等于 20。

20 epoch 是第一版发布基线，不代表 Penn-Fudan 的理论最优轮数。若真实运行发现 Kaggle 会话上限、显存或明显欠拟合问题，先记录证据，再通过更新 ADR 和参考配置修改基线；不能在 runner 中静默改变轮数。

### 12.2 参考训练配置

`configs/reference_maskrcnn.yaml` 至少表达下面的语义；字段名称应与最终 config dataclass 保持一致：

```yaml
run:
  name: reference-maskrcnn
  seed: 42
  output_dir: artifacts

data:
  provider: pennfudan
  root: data/raw
  manifest_dir: data/manifests
  batch_size: 2
  num_workers: 0                 # runner 在 Kaggle 覆盖为 2
  horizontal_flip: 0.5
  train_limit: null
  valid_limit: null
  test_limit: null
model:
  name: maskrcnn_resnet50_fpn
  weights: coco_v1
  num_classes: 2
training:
  epochs: 20
  optimizer: sgd
  learning_rate: 0.005
  momentum: 0.9
  weight_decay: 0.0005
  scheduler: step
  step_size: 6
  gamma: 0.1
  amp: false                     # runner 在 Kaggle 覆盖为 true
  best_metric: mask_map
  score_threshold: 0.5
  mask_threshold: 0.5
  evaluate_every: 1
device: cpu                      # runner 必须显式覆盖为 cuda
```

模型 factory 把 `coco_v1` 映射到明确的 torchvision weights enum，禁止使用会随版本改变含义的模糊字符串。resolved config 必须保存 Kaggle 实际覆盖后的 `device=cuda`、`amp=true`、绝对数据/产物路径和 `num_workers=2`。

### 12.3 Kernel metadata

`docs/recorded-run/kaggle/kernel-metadata.json` 采用脚本任务：

```json
{
  "id": "yashowhoo/pytorch-instance-segmentation-lab-penn-fudan-gpu",
  "title": "PyTorch Instance Segmentation Lab Penn-Fudan GPU",
  "code_file": "run_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_tpu": "false",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

用户只需替换 `id` 中的 Kaggle 用户名。任务保持 Internet 开启，以便直接下载 Penn-Fudan 和 COCO 权重；所有 source 列表保持为空。开发阶段默认 private，公开参考结果时再由维护者确认数据许可、日志和页面可见性。

### 12.4 确定性源码快照

Kaggle 非交互任务不能依赖用户临时挂载本地压缩包。`scripts/build_kaggle_runner.py` 必须：

1. 从明确 allowlist 收集 `src/instance_segmenter/`、必要的 `scripts/`、`data/manifests/` 固定划分、参考配置和项目元数据。
2. 排除 `.git`、`.venv`、cache、测试缓存、`data/raw`、`data/processed`、`artifacts`、checkpoint、`.env`、`kaggle.json`、任何 token/credentials 文件和已有生成 runner。
3. 使用稳定文件排序、固定 tar metadata 和 gzip timestamp 生成可重复的压缩字节。
4. 把压缩内容编码进 `docs/recorded-run/kaggle/run_kaggle.py`，并写入源码 archive 的字节数和 SHA-256。
5. 支持 `--check` 模式：重新构建后如与已提交 runner 不一致则返回非零，供 CI 阻止陈旧快照。
6. 安全解压到 `/kaggle/working/project`，拒绝绝对路径和 `..` 路径穿越。

`run_kaggle.py` 是生成产物，但必须提交，因为 Kaggle CLI 只上传该 kernel 目录。可维护的执行逻辑放在 `scripts/kaggle_runner.py`，不要直接手改生成文件中的 base64。

### 12.5 Runner 执行阶段

runner 必须按顺序执行，并以单行 JSON 输出 `phase`、`status`、耗时和关键计数：

1. `install_project`：解压内嵌源码，将项目和 `src` 加入 import path；验证必要依赖。不得用 pip 替换 Kaggle 的 torch/torchvision CUDA 栈；缺少 `torchmetrics` 或 `pycocotools` 时才安装已约束的轻量依赖并记录版本。
2. `gpu_preflight`：确认 `torch.cuda.is_available()`、打印 GPU 名称/数量/compute capability/CUDA/PyTorch 版本；要求 T4 或更新的兼容设备。当前 Kaggle 若分配 P100 且 PyTorch 不含 `sm_60` 内核，应尽早失败并提示换 T4。即使页面显示 T4 x2，第一版只使用 `cuda:0`。
3. `download_pennfudan`：直接从固定官方 URL 下载并校验 SHA-256，不调用 `kagglehub`，不挂载外部 Dataset。
4. `prepare_data`：把内嵌的固定 manifests 复制到 `/kaggle/working/manifests`，或用相同规则重新生成后逐字节比较；验证全部文件成员、SHA-256、dataset identity 和 `136/17/17` split counts，任何漂移都立即失败。
5. `inspect_data`：验证全部样本，并保存一张包含多个独立实例的 dataset preview。
6. `dry_run`：用真实 Mask R-CNN 和独立的新 optimizer 完成一个 batch 的 forward/loss/backward/update，然后释放对象和 CUDA cache；正式训练必须重新构造模型，不能继承 dry-run 参数。
7. `training`：加载参考配置，覆盖 Kaggle 路径、`device=cuda`、`amp=true`、`num_workers=2`，完成全部 20 epoch。
8. `evaluation`：加载 `best.pt`，只对完整 test split 评估一次 bbox 和 mask AP。
9. `prediction`：从 test 中选择固定 manifest image id，保存机器可读实例结果和 overlay。
10. `finalize`：计算 checkpoint/source/config 哈希，写 summary，检查必要产物存在，然后输出最终 JSON。

下载、dry-run、训练和评估可能数分钟没有普通日志；这些阶段至少每 60 秒输出一次 heartbeat。所有 print 必须 `flush=True`，确保 Kaggle 页面及时显示状态。

### 12.6 路径与产物

Kaggle 内固定使用：

```text
/kaggle/working/project/              # 内嵌源码解压目录
/kaggle/working/data/                 # 临时下载数据
/kaggle/working/manifests/            # 本次固定划分
/kaggle/working/artifacts/
├── reference-maskrcnn/
│   ├── config.yaml
│   ├── environment.txt
│   ├── manifest-hashes.yaml
│   ├── metrics.csv
│   ├── best.pt
│   ├── last.pt
│   ├── evaluation/
│   └── predictions/
├── dataset-preview.png
└── kaggle-run-summary.json
```

只有 `/kaggle/working` 下的结果会作为 Kernel Output 保留。runner 不得把最终产物写到只读的 `/kaggle/input` 或临时 project 子目录。

`kaggle-run-summary.json` 至少包含：

- run status、开始/结束时间和总耗时。
- Python、torch、torchvision、torchmetrics、pycocotools 版本。
- GPU 名称、compute capability、CUDA 版本和使用的 device。
- source archive SHA-256、resolved config SHA-256、dataset identity、split hashes/counts。
- completed epochs、best epoch、best valid `mask_map`。
- training/evaluation/prediction 分阶段耗时。
- test bbox/mask AP/AR、image/target/prediction 数量。
- `best.pt` SHA-256 和各主要产物的相对路径。

异常时 runner 必须在 `artifacts/kaggle-run-failure.json` 写入失败 phase、异常类型、消息、traceback、已完成 epoch 和 elapsed time，然后重新抛出异常，使 Kaggle 状态保持 `ERROR`。失败运行不能生成伪造的 successful summary。

### 12.7 提交、状态和下载体验

双语 Kaggle 指南必须给出可直接执行的命令：

```bash
uv tool install kaggle
kaggle auth login
uv run python scripts/build_kaggle_runner.py --check
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <username>/pytorch-instance-segmentation-lab-penn-fudan-gpu
kaggle kernels output <username>/pytorch-instance-segmentation-lab-penn-fudan-gpu \
  --file-pattern 'artifacts/.*' -p kaggle-output
```

指南要解释账户/GPU 验证、开启 Internet、T4 x2 但只使用一张、心跳日志、常见 CUDA capability 错误、COCO 权重下载失败、任务时限、产物位置和重跑行为。README 第一屏应把 Kaggle 完整训练作为推荐路径，而不是埋在开发章节。

### 12.8 自动化与人工验收

普通 CI 不尝试启动 Kaggle GPU，但必须测试：

- runner 构建两次字节完全一致。
- allowlist 包含运行所需源码和参考配置，denylist 中没有数据、checkpoint、cache、`.env`、Kaggle 凭据或其他 secrets。
- `--check` 能发现源码改变后的陈旧 runner。
- metadata JSON 字段和 `code_file` 正确。
- 生成脚本可被 Python 解析，内嵌 archive 可解码、安全列举并通过 SHA-256。
- runner 阶段编排可通过 monkeypatch 在本地执行，不触发网络或 GPU。
- summary/failure schema 有单元测试。

发布前人工验收必须保留 Kaggle URL 和最终 `COMPLETE` 状态，核对日志中的 GPU、20 epoch、split counts、best epoch、test image count，并实际使用 `kaggle kernels output --file-pattern 'artifacts/.*'` 下载产物。下载后的 `best.pt` 必须能在本地 CPU 执行一张图片预测。

## 13. 分阶段实施计划

### Phase 0：工程初始化

交付：`pyproject.toml`、`Makefile`、CI、LICENSE、pre-commit、双语 README、包入口、CLI 空命令。

验收：开发环境能安装，`instance-segment --help` 和 `--version` 正常，空项目测试可执行。

### Phase 1：数据契约和 synthetic 闭环

交付：`schema.py`、`masks.py`、`collate.py`、`transforms.py`、synthetic fixtures 和对应单元测试。

验收：手工 target、多个实例、空实例、同步翻转和 batch 组合全部有测试；不依赖真实下载。

### Phase 2：Penn-Fudan 数据管线

交付：下载器、provider、parser、manifest、inspection、preview、`verify-data` 和 `prepare-data`。

验收：下载可重复，哈希可校验，manifest 稳定，预览图能看出每个 pedestrian 的独立 mask/box，坏数据有明确错误。

### Phase 3：模型 registry 和模型 smoke

交付：`ModelSpec`、registry、torchvision Mask R-CNN 构造器、轻量测试替身和模型测试。

验收：`list-models` 正常；num_classes 改变时 predictor 维度正确；synthetic batch 可以 train/eval forward。

### Phase 4：训练和 checkpoint

交付：config loader、trainer、precision、checkpoint、`train`、metrics CSV 和 dry-run。

验收：CPU dry-run 完成一次参数更新；`best.pt`/`last.pt` 生成；中断后可恢复；测试集不参与选 best。

### Phase 5：评估、推理和错误分析

交付：COCO mask/bbox 指标、evaluate、predictor、overlay、per-class/per-image 报告。

验收：空预测和多实例重叠通过测试；真实验证/测试命令生成机器可读结果和可读图片。

### Phase 6：教程和自定义扩展

交付：5 个 examples、6 个教程章节、自定义 image + instance-id mask provider、外部模型工厂示例。

验收：新用户只看教程能完成最小路径；自定义 provider 不修改核心 registry 即可接入。

### Phase 7：Kaggle 完整训练和发布准备

交付：固定 20 epoch 参考 config、确定性 runner 构建器、内嵌源码的 Kaggle runner、kernel metadata、双语 Kaggle 指南、成功 GPU 运行记录、环境信息、结果摘要、贡献指南和发布检查。

验收：Kaggle 状态为 `COMPLETE`；20 个 epoch 全部完成；每轮 valid mask AP 有记录；只用 valid 选择 best；best checkpoint 在 test 上评估一次；README 中的每个结果能追溯到 resolved config、manifest hash、源码 SHA-256 和 Kaggle 页面；CI、构建、runner 新鲜度和文档链接全部通过。

## 14. 实施约束和决策规则

1. 先完成数据契约和 synthetic 测试，再接真实数据和大型模型。
2. 任何模型/数据扩展都必须通过 registry 或显式工厂，不能在 CLI 中堆 if/else。
3. 所有几何变换都必须有 image-mask-box 一致性测试。
4. 不用手写 AP 算法；优先使用经过验证的 `torchmetrics`/`pycocotools`。
5. 训练性能优化不能牺牲可读性；初学者路径必须保留可追踪的中间值。
6. 本地默认命令不能强制下载预训练权重；网络依赖必须在提示中说明。Kaggle 参考配置明确使用 COCO 权重并要求 `enable_internet: true`，不受此限制。
7. 任何记录的指标必须带 split、阈值、类别 schema、manifest hash 和 checkpoint 信息。
8. 发现已有未提交改动时，先与改动共存，不使用破坏性 Git 命令。
9. 每个阶段完成后运行对应测试，再进入下一阶段；不要最后才集中补测试。
10. Kaggle runner 只能由受测试的构建脚本生成；任何影响运行的源码或配置变化都必须重新生成 runner，并由 CI 检查快照是否陈旧。
11. Kaggle 完整训练不得设置 `train_limit`、跳过 epoch 或用 dry-run 指标代替；未完成 20 epoch 的运行只能记录为失败或诊断运行。

## 15. 完成定义

当且仅当以下条件全部满足，项目才可称为第一版完成：

- 新环境可按 README 安装并执行 CLI。
- Penn-Fudan 可下载、校验、生成固定 manifest。
- 预览能显示多个独立实例。
- CPU dry-run 完成真实 forward/backward/update。
- Mask R-CNN 可训练并保存可恢复 checkpoint。
- valid/test 都输出 bbox 和 mask COCO 指标。
- 单图预测同时生成 JSON、实例掩码和 overlay。
- 空标注、空预测、相邻实例、重叠实例有自动化测试。
- 文档清楚说明实例分割与现有检测项目、语义分割项目的区别。
- Kaggle runner 能从空白任务自动完成数据下载到 test 评估的完整流程，无需挂载外部 Dataset 或手动上传源码压缩包。
- 至少一次 T4 或更新 GPU 的任务状态为 `COMPLETE`，且 `completed_epochs=20`、全部训练样本均被使用、测试集只在选模后评估。
- 用户可用 Kaggle CLI 只下载 `artifacts`，并从中获得 `best.pt`、`last.pt`、metrics、test evaluation、预测图和运行摘要。
- 仓库中的 recorded-run 不含大 checkpoint，但保留可审计的真实指标、配置、manifest/source SHA-256、checkpoint SHA-256 和 Kaggle 运行 URL。
- `pytest`、`ruff`、`mypy`、构建、Kaggle runner 新鲜度和文档检查通过。
