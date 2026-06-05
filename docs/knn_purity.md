# kNN Purity

kNN purity 衡量 embedding 空间中每个样本最近的 `k` 个邻居里，有多少和它属于同一个标签。默认 timbre label 使用 NSynth 的 `instrument_family_str`。

```text
Purity@k(i) = #{j in NN_k(i), y_j = y_i} / k
Purity@k = mean_i Purity@k(i)
```

## 评估已有 embedding

先用 `run-instrument-tsne` 或 `extract-tokenizer-embeddings` 生成 `.npz` embedding 和同名 `.metadata.csv`，然后：

```powershell
run-knn-purity `
  --input-dir outputs/nsynth_instrument_pitch_stratified `
  --k 5 10 30 50 `
  --label-col instrument_family_str `
  --normalization standardize
```

命令会扫描 `--input-dir` 下匹配 `*/embeddings.npz` 的模型目录，例如 `same_same_s/`、`same_same_l/`、`mert_layer_18/`。

## 使用 pitch 分层采样评估

如果 embedding 是在较大的 manifest 上抽取的，也可以在 kNN purity 评估前按 metadata 行号同步下采样 embedding 和 metadata：

```powershell
run-knn-purity `
  --input-dir outputs/nsynth_instrument_pitch_stratified `
  --pitch-stratified `
  --max-per-family 200 `
  --pitch-bin-size 1 `
  --label-col instrument_family_str `
  --sample-family-col instrument_family_str `
  --pitch-label-col pitch
```

相关参数：

- `--pitch-stratified`：评估前按 pitch strata 下采样；优先保证同一个 pitch 内各 family 数量一致。
- `--sample-family-col instrument_family_str`：family 均衡使用的 metadata 列。
- `--pitch-label-col pitch`：pitch strata 使用的 metadata 列，同时也是 Pitch Purity 的默认标签列。
- `--max-per-family 200`：每个 family 的目标上限。
- `--pitch-bin-size 1`：每个 MIDI pitch 一个 stratum。
- `--max-per-pitch N`：限制每个 pitch stratum 的样本数；实际会按 family 数向下取整以保持同 pitch 内 family 等量。
- `--keep-incomplete-pitch-strata`：保留未覆盖所有 family 的 pitch strata。

如果 embedding 已经来自 `outputs/nsynth_valid_manifest_pitch_stratified.csv`，通常不需要再次传 `--pitch-stratified`。当需要从更大的 embedding 缓存中抽出同一类 pitch-family balanced 子集时，再打开这个选项。

## 常用参数

- `--k 5 10 30 50`：自定义主表的 k 值。
- `--normalization standardize`：先标准化再用 cosine distance 找邻居。
- `--normalization l2`：只做 L2 归一化后找邻居。
- `--pca-components 128`：主表在 kNN 前先做 PCA；不传或传 `0` 表示 Raw embedding。
- `--pitch-label-col pitch`：Pitch Purity 使用的 metadata 列。
- `--source-label-col instrument_source_str`：Source Purity 使用的 metadata 列。
- `--pitch-source-k 10`：Pitch/Source Purity 使用的 k。
- `--robustness-k 10`：稳健性表使用的 k。
- `--robustness-pca-components 64 128 256`：稳健性表比较的 PCA 维度。
- `--embeddings path/to/embeddings.npz ...`：不按目录扫描，手动指定一个或多个 embedding 文件。

## 序列级 embedding pooling

如果保存的是序列级 embedding，命令会在评估前做 clip-level pooling；默认沿 axis 1 做 mean pooling。已有 SAME/MERT 抽取器默认输出 clip-level mean pooled 向量，因此通常不需要额外设置。

```powershell
run-knn-purity `
  --embeddings outputs/mert_embeddings.npz `
  --clip-pooling mean `
  --clip-pool-axis 1
```

## 输出文件

输出默认写回 `--input-dir`，或写到 `--output-dir`：

- `knn_purity_report.html`：HTML 报告，包含 timbre/pitch/source purity 主表和 Raw/PCA 稳健性表。
- `knn_purity_summary.csv`：主表宽表，包含 purity、随机 baseline 和 delta。
- `knn_purity_robustness.csv`：稳健性宽表，默认比较 Raw、PCA-64、PCA-128、PCA-256 的 `Purity@10`。
- `knn_purity_long.csv`：长表，便于后续画图或统计。
- `knn_purity.json`：完整指标结果。
