# nuScenes 数据与评测

`tools/nuscenes` 提供以下功能：

- 在不依赖 MM 系列库的情况下生成与 UniDriveVLA 对齐的增强版 nuScenes info PKL。
- 使用 Max 生成六个未来 waypoint，并保存为 UniDriveVLA 的 `token -> trajectory` PKL。
- 计算 UniDriveVLA 对齐的 STRICT、UniAD 和 STP-3 规划指标。

## 生成增强版 info PKL


```bash
python tools/nuscenes/nuscenes_converter.py nuscenes \
  --root-path data/nuscenes \
  --canbus data/nuscenes \
  --out-dir data/UniDriveVLA_Data \
  --extra-tag nuscenes \
  --version v1.0
```

输出：

```text
data/UniDriveVLA_Data/nuscenes_infos_train.pkl
data/UniDriveVLA_Data/nuscenes_infos_val.pkl
```

## Max 推理与评测

checkpoint 必须配置 `pred_len=6`。UniAD/STP-3 评测默认读取 UniDriveVLA
原版 `data/UniDriveVLA_Data/planing_gt_segmentation_val`。

```bash
torchrun --nproc_per_node=8 tools/nuscenes/test.py \
  --model-path /path/to/checkpoint \
  --traj-file data/UniDriveVLA_Data/nuscenes_traj_val.jsonl \
  --info-pkl data/UniDriveVLA_Data/nuscenes_infos_val.pkl \
  --result-dir results \
  --eval \
  --batch-size 1 \
  --num-workers 8 \
  --n-samples 100 \
  --seed 42 \
  --enable-thinking true
```

nuScenes 推理与数据准备统一使用
`models.max_v1.prompt_template.NUSCENES_SYSTEM` 作为 system prompt。

`--result-dir` 默认为 `results/`。每次运行会创建格式为
`YYYYMMDD-HHMMSS` 的时间戳子目录，并将推理结果保存为
`RESULT_DIR/YYYYMMDD-HHMMSS/max_pred_trajs.pkl`，格式为：

```python
{sample_token: np.ndarray(shape=(6, 2), dtype=np.float32)}
```

同时指定 `--eval` 时，评测结果保存在同一时间戳子目录的
`planning_metrics.json`。仅推理时移除 `--eval`。通过
`--seg-pkl` 可以覆盖默认的 segmentation PKL 路径。`--n-samples`
使用 `--seed` 从全量 val 集中随机选择子集，省略时运行全部样本。

## 评测已有预测 PKL

指定 `--pred-pkl` 后跳过模型推理：

```bash
python tools/nuscenes/test.py \
  --pred-pkl work_dirs/nuscenes/max_pred_trajs.pkl \
  --info-pkl data/UniDriveVLA_Data/nuscenes_infos_val.pkl \
  --result-dir results \
  --eval
```

此模式只在新的时间戳子目录写入 `planning_metrics.json`，不会复制或
覆盖已有预测 PKL。评测 JSON 的 metadata 会记录 `prediction_pkl`、
完整测试集的 `total_samples`，以及实际参与评测的
`evaluated_samples`。STRICT、UniAD 和 STP-3 分别通过
`valid_samples` 记录各自有效的评测样本数。

与 UniDriveVLA README 对标时，使用 STP-3 表格中的 `L2` 和
`obj_box_col`（`Col`，百分比）结果。
