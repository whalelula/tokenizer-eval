# Tokenizer Evaluation

本项目用于评估给定 pre-trained tokenizer / audio representation model 的能力，长期框架分为三类：

- **Reconstruction**：编码-解码后比较音频重建质量。
- **Semantic Representation**：检查 latent embedding 是否保留语义标签，例如 instrument、pitch、style 等。
- **Downstream Generation**：把 tokenizer 输出用于生成任务，评估可控性、质量和多样性。

本次已实现的任务是：在 NSynth 上评估 SAME 和 MERT 的 instrument classification / clustering 能力，输出 latent embedding、t-SNE 聚类可视化和简单 kNN probe 指标。

## 官方实现依据

- SAME：参考 Stability AI 官方 `stable-audio-3` 仓库的 `AutoencoderModel.from_pretrained(...).encode(...)` 推理方式。仓库：[https://github.com/Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3)
- SAME-S 模型页：[https://huggingface.co/stabilityai/SAME-S](https://huggingface.co/stabilityai/SAME-S)
- MERT：参考官方仓库和 Hugging Face 用法，使用 `Wav2Vec2FeatureExtractor` + `AutoModel` 并读取 `hidden_states`。仓库：[https://github.com/yizhilll/MERT](https://github.com/yizhilll/MERT)
- MERT-v1-95M 模型页：[https://huggingface.co/m-a-p/MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M)
- NSynth：使用官方 `examples.json` + `audio/*.wav` 结构。数据集：[https://magenta.tensorflow.org/datasets/nsynth](https://magenta.tensorflow.org/datasets/nsynth)

## 安装

建议 Python 3.10+。推荐用 conda 管理 Python 环境，或者使用 Python 自带的 `venv`；二者任选其一即可。

方案 A：使用 conda。

```powershell
conda create -n tokenizer-eval python=3.10 -y
conda activate tokenizer-eval
pip install -e .
```

方案 B：使用 Python venv。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

如果要跑 SAME，还需要安装官方 SAME 代码：

```powershell
pip install "tokenizer-evaluation[same]"
```

如果上面的 Git 依赖安装失败，也可以直接安装官方仓库：

```powershell
pip install git+https://github.com/Stability-AI/stable-audio-3.git
```

如果本地 pip 下载依赖较慢或 PyPI 连接失败，可以改用国内镜像源安装。清华源：

```powershell
pip install -e ".[same]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

阿里云源：

```powershell
pip install -e ".[same]" -i https://mirrors.aliyun.com/pypi/simple
```

也可以只给官方 SAME 仓库安装时的 PyPI 依赖使用镜像源：

```powershell
pip install git+https://github.com/Stability-AI/stable-audio-3.git -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：镜像源主要解决 PyPI 依赖包下载问题；如果失败原因是无法访问 GitHub，本机仍然需要能访问 `https://github.com/Stability-AI/stable-audio-3.git`。

## 准备 NSynth

可以让脚本下载并解压官方 split。建议先用 `valid` split 验证流程。

```powershell
prepare-nsynth `
  --nsynth-root data/nsynth `
  --split valid `
  --download `
  --manifest outputs/nsynth_valid_manifest.csv `
  --max-per-family 200
```
或者下载 `test`

```powershell
prepare-nsynth `
  --nsynth-root data/nsynth `
  --split test `
  --download `
  --manifest outputs/nsynth_test_manifest.csv `
  --max-per-family 50
```

如果你已经手动下载了 NSynth，目录只需要满足下面结构之一：

```text
data/nsynth/nsynth-valid/examples.json
data/nsynth/nsynth-valid/audio/*.wav
```

或：

```text
data/nsynth/examples.json
data/nsynth/audio/*.wav
```

## 一键运行 SAME + MERT t-SNE

默认配置在 `configs/instrument_classification.yaml`，默认按 GPU 推理，即 `device: cuda`。

云端 GPU：

```powershell
run-instrument-tsne `
  --nsynth-root data/nsynth `
  --split valid `
  --models same mert `
  --device cuda `
  --output-dir outputs/nsynth_instrument
```

本地 CPU 小样本试跑：

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

## 分步运行

只生成 manifest：

```powershell
prepare-nsynth --nsynth-root data/nsynth --split valid --manifest outputs/nsynth_valid_manifest.csv --max-per-family 200
```

只抽取 MERT embedding：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model mert `
  --output outputs/mert_embeddings.npz `
  --device cuda
```

只抽取 SAME embedding：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model same `
  --output outputs/same_embeddings.npz `
  --device cuda
```

## 输出结果

一键脚本会在 `outputs/nsynth_instrument/` 下生成：

- `same/embeddings.npz`：SAME latent embedding。
- `same/embeddings.metadata.csv`：对应的 NSynth 元数据。
- `same/tsne.csv`：t-SNE 坐标 + 标签。
- `same/tsne.png`：SAME 单图。
- `same/metrics.json`：silhouette、kNN accuracy、macro F1。
- `mert/...`：MERT 对应结果。
- `same_vs_mert_tsne.png`：SAME 和 MERT 的横向对比图。

## 核心流程与代码文件

- NSynth 预处理：`src/tokenizer_evaluation/datasets/nsynth.py`
  - 读取 `examples.json`
  - 检查 `audio/*.wav`
  - 按 `instrument_family_str` 做 balanced sampling
  - 输出 manifest CSV
- SAME 推理：`src/tokenizer_evaluation/models/same.py`
  - 加载 `stable_audio_3.AutoencoderModel`
  - 调用 `encode(waveform, sample_rate)`
  - 对 latent sequence 做 mean 或 mean+std pooling
- MERT 推理：`src/tokenizer_evaluation/models/mert.py`
  - 加载 `Wav2Vec2FeatureExtractor` 和 `AutoModel`
  - 重采样到 MERT 采样率
  - 读取指定 hidden layer 并做 pooling
- embedding 缓存：`src/tokenizer_evaluation/embeddings.py`
  - 保存 `.npz` 和 `.metadata.csv`
  - 默认不重复抽取，除非加 `--overwrite`
- t-SNE：`src/tokenizer_evaluation/reduction.py`
  - 标准化 embedding
  - 高维时先 PCA 到 50 维
  - 再运行 t-SNE
- 指标：`src/tokenizer_evaluation/metrics.py`
  - silhouette
  - kNN accuracy
  - kNN macro F1
- 可视化：`src/tokenizer_evaluation/visualization.py`
  - 单模型散点图
  - SAME vs MERT 横向对比图
- 一键入口：`src/tokenizer_evaluation/cli/run_instrument_tsne.py`

## 配置说明

常用参数都在 `configs/instrument_classification.yaml`：

- `runtime.device`: 默认 `cuda`。本地 CPU 运行时传 `--device cpu` 覆盖。
- `dataset.max_per_family`: 每个 instrument family 最多采样多少条。
- `models.same.model_name`: 默认 `same-s`，GPU 上可改成 `same-l`。
- `models.same.chunk_size` / `models.same.overlap`: SAME chunked encode 的 latent frame 参数。
- `models.mert.layer`: 默认 `-1`，表示最后一层 hidden state。
- `tsne.perplexity`: t-SNE perplexity；样本很少时脚本会自动调低。
- `runtime.overwrite`: 是否重新抽取 embedding。

## 建议实验设置

先用 `valid` split 和较小 `max_per_family` 验证依赖、下载和图片输出。确认流程正确后，在 GPU 服务器上提高 `max_per_family`，并分别尝试 MERT 的不同 layer，例如 `-1`、`6`、`8`，比较 `metrics.json` 和 t-SNE 图。
