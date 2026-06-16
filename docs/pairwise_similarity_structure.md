# Pairwise Similarity Structure

这个评估用于比较两个已经抽取好的 latent space 是否保留了相似的样本间结构。
输入是两个 embedding bundle：`embeddings.npz` 和同名的 `.metadata.csv`。

## 支持的选项

数据集选项：

- `nsynth`: NSynth
- `medley-solo-db`: medley-solo-db
- `mtg-jamendo`: MTG-Jamendo

tokenizer 选项：

- `mert`: MERT
- `hubert`: HuBERT
- `beats`: BEATs
- `atst`: ATST
- `stable-audio-open-vae`: stable audio open VAE, based on `Stability-AI/stable-audio-tools`
- `same`: SAME
- `music2latents`: Music2Latents
- `wavcube`: WavCube
- `speechtokenizer`: SpeechTokenizer
- `x-codec`: X-Codec
- `mucodec`: MuCodec

当前命令比较的是已保存 latent bundle，因此 tokenizer 选项用于记录和校验实验设置。
如果某个 tokenizer 还没有本项目内置抽取器，可以先用外部脚本生成兼容的 `.npz`
和 `.metadata.csv`，再运行这个评估。

## 指标定义

对两个 latent bundle 分别做 clip-level pooling，然后对每个样本向量做 L2 归一化。
在每个 latent space 内计算 cosine similarity matrix，并且只使用非对角样本对。

全局结构一致性：

```text
SSS_global = corr(rank(v_A), rank(v_B))
```

这里 `v_A` 和 `v_B` 是两个 latent space 中所有唯一非对角 pair 的 cosine similarity
向量，`corr` 是 Pearson correlation，因此整体等价于 Spearman rank correlation。
它回答：A 里越相似的样本 pair，在 B 里是否也越相似？

局部结构一致性：

```text
SSS_local@k = mean_i |NN_A^k(i) intersect NN_B^k(i)| / k
```

它回答：对每个样本 `i`，A 中最近的 `k` 个邻居，在 B 中是否仍然也是邻居？
取值范围是 `[0, 1]`。

## 用法

如果两个 bundle 来自同一个 manifest 且行顺序一致：

```powershell
run-pairwise-structure `
  --dataset nsynth `
  --tokenizer-a same `
  --tokenizer-b mert `
  --embeddings-a outputs/nsynth_instrument/same/embeddings.npz `
  --embeddings-b outputs/nsynth_instrument/mert/embeddings.npz `
  --k 5 10 30 50
```

如果两个 bundle 的行顺序不确定，用 metadata 中唯一列对齐，例如 NSynth 的 `id`
或 `note_str`：

```powershell
run-pairwise-structure `
  --dataset nsynth `
  --tokenizer-a same `
  --tokenizer-b mert `
  --embeddings-a outputs/same/embeddings.npz `
  --embeddings-b outputs/mert/embeddings.npz `
  --join-col id `
  --k 5 10 30
```

如果数据集很大，可以先抽样，避免构建过大的 pairwise matrix：

```powershell
run-pairwise-structure `
  --dataset mtg-jamendo `
  --tokenizer-a music2latents `
  --tokenizer-b stable-audio-open-vae `
  --embeddings-a outputs/music2latents/embeddings.npz `
  --embeddings-b outputs/stable_audio_open_vae/embeddings.npz `
  --join-col audio_path `
  --max-items 5000 `
  --seed 42
```

## 输出

默认输出到：

```text
outputs/pairwise_structure/<dataset>/<tokenizer-a>_vs_<tokenizer-b>/
```

文件包括：

- `pairwise_similarity_structure.json`: 完整配置、对齐信息和指标结果。
- `pairwise_similarity_structure_summary.csv`: 单行宽表，包含 `SSS_global` 和各个 `SSS_local@k`。
- `pairwise_similarity_structure_long.csv`: 长表，方便后续画图或统计。
- `pairwise_similarity_structure_metadata.csv`: 实际参与比较的对齐后 metadata。
