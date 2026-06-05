# NSynth Data

本项目使用 NSynth 官方 split 结构：

```text
data/nsynth/nsynth-valid/examples.json
data/nsynth/nsynth-valid/audio/*.wav
```

也支持直接把 `examples.json` 和 `audio/` 放在 `data/nsynth/` 下。

## 下载并生成普通 manifest

```powershell
prepare-nsynth `
  --nsynth-root data/nsynth `
  --split valid `
  --download `
  --manifest outputs/nsynth_valid_manifest.csv `
  --max-per-family 200
```

如果已经下载并解压过，可以不传 `--download`：

```powershell
prepare-nsynth `
  --nsynth-root data/nsynth `
  --split valid `
  --manifest outputs/nsynth_valid_manifest.csv `
  --max-per-family 200
```

## Pitch 分层采样

`outputs/nsynth_valid_pitch_family_report.html` 显示 valid split 中 pitch `31-96` 都覆盖 10 个 instrument family。因此推荐的数据准备策略是：

- 只保留 pitch `31-96`
- 以 1 个 MIDI pitch 为一个 stratum
- 过滤掉没有覆盖所有目标 family 的 pitch strata
- 第一优先级：同一个 pitch 内，不同 `instrument_family_str` 的样本数一致或尽量一致
- 第二优先级：在第一条约束下，让每个 `instrument_family_str` 的总样本量一致或尽量一致
- 因为优先控制 pitch 内的 family 构成，每个 pitch 的总样本量不一定相同

生成本次推荐 manifest：

```powershell
prepare-nsynth `
  --nsynth-root data/nsynth `
  --split valid `
  --manifest outputs/nsynth_valid_manifest_pitch_stratified.csv `
  --max-per-family 200 `
  --pitch-min 31 `
  --pitch-max 96 `
  --pitch-stratified `
  --pitch-bin-size 1 `
  --seed 42
```

当前检查结果：

```text
rows: 2000
pitch range: 31-96
pitch strata: 66
instrument families: 10
samples per family: 200
within-pitch family spread: 0
samples per pitch: 10-40
per-pitch per-family quota: 1-4
```

## 采样参数

- `--max-per-family 200`：每个 instrument family 目标上限。
- `--pitch-min 31` / `--pitch-max 96`：pitch 范围过滤。
- `--pitch-stratified`：启用 pitch 分层采样；采样器优先让同一 pitch 内各 family 数量一致。
- `--pitch-bin-size 1`：每 1 个 MIDI pitch 一个 stratum；例如设为 `2` 会按两半音分箱。
- `--max-per-pitch N`：显式限制每个 pitch stratum 的最大样本数。为了保持同 pitch 内 family 等量，实际每个 pitch 最多会使用 `floor(N / family_count) * family_count` 条。
- `--keep-incomplete-pitch-strata`：保留没有覆盖所有目标 family 的 pitch strata。默认会过滤掉这些 strata。
- `--seed 42`：控制可复现实验抽样。

## Test split 示例

```powershell
prepare-nsynth `
  --nsynth-root data/nsynth `
  --split test `
  --download `
  --manifest outputs/nsynth_test_manifest.csv `
  --max-per-family 50
```
