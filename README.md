# PackingProof Vision

通用 YOLO 视觉模型训练、评估和导出底座。当前实现支持 YOLO-OBB，任务名、类别名、
模型角色和数据来源都由配置决定；面单识别只是其中一个任务。

仓库只保存代码、测试和 CI。图片、样本包、训练 run、权重和 Release 产物不进入 Git。

## 环境

Python 解析顺序：

1. `PACKINGPROOF_VISION_PYTHON`
2. 仓库根目录 `.venv`
3. 系统 `python`

检查 GPU 环境：

```powershell
pwsh -NoProfile -File scripts\Test-Environment.ps1 -RequireCuda
```

创建独立训练环境：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements\cuda128.txt
uv pip install --python .venv\Scripts\python.exe -r requirements\train.txt
```

## 任务配置

数据根目录通过环境变量提供：

```powershell
$env:PACKINGPROOF_VISION_DATASET_ROOT = "D:\datasets\packingproof"
Copy-Item config\waybill-obb.example.json config\waybill-obb.local.json
```

建议数据根目录按用途分层，避免原始包、训练集和模型产物混放：

```text
packingproof/
  raw/
    archives/
    PackingProof_RecognitionSamples_*/
  prepared/
    waybill_yolo_obb_v6/
  holdouts/
    independent_test/
  artifacts/
    models/
    runs/
    pretrained/
    contact-sheets/
```

配置分为两层：

- `task`：任务名称、任务类型、模型角色和类别列表
- `dataset`：数据目录、独立测试集、切分参数和数据来源

`dataset.sources[].tags` 完全可选。它不会变成模型类别，不会改变图片或标注，也不会
影响默认训练切分；只会写入 `dataset_report.json` 和 `sample_index.json`，方便区分样本
来源。例如，一批退货录像可以这样标：

```json
{
  "name": "return_samples_20261101",
  "path": "...",
  "tags": ["return"]
}
```

不关心来源统计时，直接删掉 `tags` 即可。只有显式把 `"tags"` 加入 `stratify_by` 时，
它才会参与 train/val 分层；训练器本身不会读取 `tags`。因此发退货、发货、不同现场或
不同工位的全部样本仍然是同一个任务，只有类别由 `task.classes` 定义。

默认切分按 `class_presence` 和 `capture_phase` 分层，并始终按
`sourceFingerprint` 分组，确保同一录像的多帧不会跨 train/val。

如果部分现场产生大量空画面，可在 `dataset` 中设置可选上限：

```json
"max_negative_ratio": 0.2
```

启用后所有正样本都会保留，负样本按来源录像、采集前后阶段和来源包分层抽取；同一来源
录像最多保留一个负样本，避免静止背景重复。该值表示负样本最终占总图片的比例，范围为
`[0, 1)`。

对专门收集的误检困难负样本，可以在来源上设置：

```json
{
  "name": "pc_ui_hard_negative",
  "path": "...",
  "tags": ["hard-negative", "pc-ui"],
  "keep_all_negatives": true
}
```

这类来源的空标签图片会全部保留，并优先占用普通负样本额度；普通空画面仍按
`max_negative_ratio` 抽样。

准备脚本会：

- 同时支持 AnyLabeling JSON 和 YOLO TXT
- 使用 `task.classes` 校验类别 ID
- 将多边形裁到 `[0,1]`
- 丢弃低于面积阈值的退化框
- 输出 `data.yaml`、`task.json`、`dataset_report.json` 和 `sample_index.json`

## 训练与评估

```powershell
pwsh -NoProfile -File scripts\Test-Repository.ps1
pwsh -NoProfile -File scripts\Invoke-Prepare.ps1 -Force
```

面单 OBB 当前示例：

```powershell
pwsh -NoProfile -File scripts\Invoke-Train.ps1 `
  -Data "$env:PACKINGPROOF_VISION_DATASET_ROOT\prepared\waybill_yolo_obb_v5\data.yaml" `
  -Model "yolo11n-obb.pt" `
  -Name "waybill_yolo11n_obb_v5"

pwsh -NoProfile -File scripts\Invoke-Evaluate.ps1 `
  -Data "$env:PACKINGPROOF_VISION_DATASET_ROOT\holdouts\independent_test\data.yaml" `
  -Models @(
    "$env:PACKINGPROOF_VISION_DATASET_ROOT\artifacts\models\waybill_yolo11n_obb_v4_best.pt",
    "$env:PACKINGPROOF_VISION_DATASET_ROOT\artifacts\models\waybill_yolo11n_obb_v5_best.pt"
  )
```

微调时可显式覆盖 `-Optimizer`、`-Lr0`、`-Patience` 和 `-Workers`，避免使用
Ultralytics 的自动优化器参数。例如低学习率微调：

```powershell
pwsh -NoProfile -File scripts\Invoke-Train.ps1 `
  -Data "$env:PACKINGPROOF_VISION_DATASET_ROOT\prepared\waybill_yolo_obb_v6\data.yaml" `
  -Model "$env:PACKINGPROOF_VISION_DATASET_ROOT\artifacts\models\wallbill_yolo11n_obb_v4_best.pt" `
  -Name "waybill_yolo11n_obb_v6" `
  -Optimizer AdamW -Lr0 0.0005 -Patience 12 -Batch 16
```

训练脚本从相邻的 `task.json` 读取任务名称和模型角色，并把训练参数、数据集 SHA256、
类别列表和独立测试结果写入 PyTorch/ONNX 的 `packingproof.*` 元数据。

评估默认只写 `runs/independent-eval/report.json`，不会修改参考模型。需要归档到最终
模型时再显式传入 `--update-metadata`。

## 扩展其他任务

[package-check-obb.example.json](config/package-check-obb.example.json) 展示了装件检查
任务的配置骨架。替换 `task.classes` 和数据集即可训练多类别 OBB 模型，不需要复制推理
或数据转换代码。

当前底座支持 `type: "obb"`。后续增加普通检测、分割或分类时，应通过独立任务适配器扩展，
不要把任务特有规则写进现有 OBB 代码。

## Release 产物

只生成本地待审产物，不上传：

```powershell
python scripts\package_model_release.py `
  --model "$env:PACKINGPROOF_VISION_DATASET_ROOT\artifacts\models\waybill_yolo11n_obb_v5_best.onnx" `
  --version 5 `
  --training-run waybill_yolo11n_obb_v5 `
  --model-id waybill-yolo11n-obb-v5 `
  --release-repository PackingProof/PackingProof-Vision `
  --asset-name vision-obb.onnx
```

产物包含：

- 稳定的运行时角色文件，例如 `vision-obb.onnx`
- `model-manifest.json`
- `SHA256SUMS.txt`
- `release-notes.md`

模型版本由 Release 标签、锁定清单和 ONNX 元数据表达，文件名保持通用。

## CI

`.github/workflows/ci.yml` 在 Ubuntu 和 Windows 上验证：

- Python 语法
- 通用 OBB 数据转换和分组切分
- 可选 tags 行为
- Release 产物打包与哈希
- PowerShell 环境解析脚本语法

CI 不下载样本，也不执行 GPU 训练。真实数据验证和训练在挂载数据集的机器上运行。

## 数据安全

- 不要提交 `PackingProof_RecognitionSamples_*`
- 不要提交 `artifacts/models/`、`artifacts/runs/`、`dist/`
- 不要提交 `.pt`、`.onnx`、样本 ZIP 或本机配置
- 公开 Release 前复核样本授权、人物隐私、Ultralytics/AGPL 义务和模型指标
