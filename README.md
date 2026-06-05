# Tokenizer Evaluation

用于评估 pre-trained audio tokenizer / representation model 的小型实验框架。当前重点是：在 NSynth 上比较 SAME 和 MERT 的 instrument family 表征能力，生成 manifest、embedding、t-SNE 可视化和 kNN purity 指标。

## 当前核心流程

1. 准备环境  
   见 [docs/setup.md](docs/setup.md)。

2. 准备 NSynth manifest  
   默认推荐使用 valid split 的 pitch 分层采样：pitch 31-96，同一个 pitch 内 10 个 instrument family 的样本数优先保持一致，其次保持每个 family 总量一致。

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

   详细说明见 [docs/nsynth_data.md](docs/nsynth_data.md)。

3. 抽取 embedding 并运行 t-SNE

   ```powershell
   run-instrument-tsne `
     --nsynth-root data/nsynth `
     --split valid `
     --manifest outputs/nsynth_valid_manifest_pitch_stratified.csv `
     --models same mert `
     --device cuda `
     --output-dir outputs/nsynth_instrument_pitch_stratified
   ```

   详细说明见 [docs/instrument_tsne.md](docs/instrument_tsne.md)。

4. 评估 kNN purity

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

   详细说明见 [docs/knn_purity.md](docs/knn_purity.md)。

## 配置文件

默认实验配置在 [configs/instrument_classification.yaml](configs/instrument_classification.yaml)。其中已经启用 valid split 推荐的 pitch 分层采样：

- `dataset.pitch_min: 31`
- `dataset.pitch_max: 96`
- `dataset.pitch_stratified: true`
- `dataset.pitch_bin_size: 1`
- `dataset.pitch_require_all_families: true`

## 代码入口

- NSynth 数据准备：`src/tokenizer_evaluation/datasets/nsynth.py`
- manifest CLI：`prepare-nsynth`
- embedding/t-SNE CLI：`run-instrument-tsne`
- kNN purity CLI：`run-knn-purity`
- SAME 模型封装：`src/tokenizer_evaluation/models/same.py`
- MERT 模型封装：`src/tokenizer_evaluation/models/mert.py`
- 指标实现：`src/tokenizer_evaluation/metrics.py`
- 可视化实现：`src/tokenizer_evaluation/visualization.py`

## 更多文档

- [docs/project_framework.md](docs/project_framework.md)：评估框架概览。
- [docs/setup.md](docs/setup.md)：安装、可选依赖、镜像源和 HF 缓存设置。
- [docs/nsynth_data.md](docs/nsynth_data.md)：NSynth 下载、manifest 生成、pitch 分层采样。
- [docs/instrument_tsne.md](docs/instrument_tsne.md)：SAME/MERT embedding 与 t-SNE 流程。
- [docs/knn_purity.md](docs/knn_purity.md)：kNN purity 指标、输出文件和常用参数。
