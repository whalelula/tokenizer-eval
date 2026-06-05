# Instrument t-SNE

`run-instrument-tsne` 会先准备 manifest，然后抽取 SAME/MERT embedding，最后运行 t-SNE、保存图和基础指标。

默认配置在 `configs/instrument_classification.yaml`。该配置已经启用 valid split 推荐的 pitch 分层采样：pitch `31-96`、`pitch_bin_size: 1`、`max_per_family: 200`。采样器优先保证同一个 pitch 内不同 instrument family 的样本数一致，其次保证每个 family 总量一致。

## 推荐运行

如果已经生成 pitch-stratified manifest，可以显式复用它：

```powershell
run-instrument-tsne `
  --nsynth-root data/nsynth `
  --split valid `
  --manifest outputs/nsynth_valid_manifest_pitch_stratified.csv `
  --models same mert `
  --device cuda `
  --output-dir outputs/nsynth_instrument_pitch_stratified
```

也可以让命令按配置自动生成 manifest：

```powershell
run-instrument-tsne `
  --nsynth-root data/nsynth `
  --split valid `
  --models same mert `
  --device cuda `
  --output-dir outputs/nsynth_instrument_pitch_stratified
```

## 本地 CPU 小样本试跑

```powershell
run-instrument-tsne `
  --nsynth-root data/nsynth `
  --split valid `
  --models mert `
  --device cpu `
  --max-per-family 20 `
  --output-dir outputs/nsynth_instrument_cpu
```

SAME 在 CPU 上也可以跑，但通常会慢很多：

```powershell
run-instrument-tsne `
  --nsynth-root data/nsynth `
  --split valid `
  --models same `
  --device cpu `
  --max-per-family 10
```

## Pitch 分层相关参数

`run-instrument-tsne` 会把采样参数传给 manifest 准备流程：

```powershell
run-instrument-tsne `
  --nsynth-root data/nsynth `
  --split valid `
  --models mert `
  --device cuda `
  --pitch-stratified `
  --pitch-bin-size 1 `
  --max-per-family 200 `
  --output-dir outputs/nsynth_instrument_pitch_stratified
```

可用参数：

- `--pitch-stratified`：命令行强制启用 pitch 分层；优先控制同一 pitch 内 family 数量一致。
- `--no-pitch-stratified`：命令行关闭配置文件里的 pitch 分层。
- `--pitch-bin-size 1`：pitch stratum 宽度。
- `--max-per-pitch N`：限制每个 pitch stratum 的样本数；实际会按 family 数向下取整以保持同 pitch 内 family 等量。
- `--keep-incomplete-pitch-strata`：保留未覆盖所有 family 的 pitch strata。

## 分步抽取 embedding

只抽取 MERT embedding：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest_pitch_stratified.csv `
  --model mert `
  --output outputs/mert_embeddings.npz `
  --device cuda
```

只抽取 SAME embedding：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest_pitch_stratified.csv `
  --model same `
  --output outputs/same_embeddings.npz `
  --device cuda
```

## 输出文件

一键脚本会在 `--output-dir` 下为每个模型生成：

- `embeddings.npz`：模型 embedding。
- `embeddings.metadata.csv`：对应的 NSynth 元数据。
- `embeddings.summary.json`：embedding 摘要。
- `tsne.csv`：t-SNE 坐标 + metadata。
- `tsne.png`：单模型 t-SNE 图。
- `metrics.json`：silhouette、kNN accuracy、macro F1。

同时会生成一个横向对比图，例如 `same_vs_mert_tsne.png` 或 `mert_layers_tsne.png`。
