# Tokenizer Embedding Extractors

本项目现在可以通过 `extract-tokenizer-embeddings` 抽取以下 tokenizer /
representation model 的 clip-level latent embedding。

已经内置的模型：

- `same`
- `mert`
- `hubert`
- `beats`
- `atst`
- `stable-audio-open-vae`
- `music2latents`
- `wavcube`
- `speechtokenizer`
- `x-codec`
- `x-codec-pre-quantized`
- `x-codec-post-quantized`
- `x-codec-ssl-latent`
- `x-codec-vae-latent`
- `mucodec`
- `dac`

## 安装建议

先安装与你机器匹配的 PyTorch / torchaudio，避免 `pip install -e .` 自动拉取不合适的
CUDA 大包。例如 CUDA 11.8：

```powershell
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

CPU 环境：

```powershell
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

然后按需安装本项目和 extras：

```powershell
pip install -e .
pip install -e ".[hubert]"
pip install -e ".[xcodec]"
pip install -e ".[stable-audio-open]"
pip install -e ".[music2latents]"
pip install -e ".[speechtokenizer]"
pip install -e ".[dac]"
```

国内或网络不稳定时，可以加 PyPI 镜像：

```powershell
pip install -e ".[hubert]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：PyPI 镜像只加速 PyPI 依赖。`stable-audio-open` 默认安装 PyPI 上的
`stable-audio-tools`，不需要 clone GitHub。如果你需要官方仓库的最新代码，可以先手动 clone
再安装：

```powershell
git clone --depth 1 https://github.com/Stability-AI/stable-audio-tools.git external/stable-audio-tools
pip install -e external/stable-audio-tools
pip install -e ".[stable-audio-open]"
```

形如 `git+https://github.com/...` 的 GitHub 依赖仍然需要能访问 GitHub。

Hugging Face 模型下载可以使用镜像和大容量缓存目录：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
```

## 通用命令

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model hubert `
  --output outputs/hubert/embeddings.npz `
  --device cuda `
  --batch-size 1
```

输出仍然是项目统一的 bundle：

- `embeddings.npz`
- `embeddings.metadata.csv`
- `embeddings.summary.json`

所有 extractor 默认输出 clip-level 1D 向量。对于序列 latent，默认 `--pooling mean`。
常用可选值是 `mean`、`mean_std`、`flatten`。

## HuBERT

官方代码：`facebookresearch/fairseq` 的 HuBERT example。项目里使用 Hugging Face
`AutoModel` wrapper，以便和 MERT 共用抽取路径。

默认：

- `--model-name facebook/hubert-base-ls960`
- `--sample-rate` 来自 feature extractor，通常是 `16000`
- `--layer -1`

示例：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model hubert `
  --model-name facebook/hubert-large-ls960-ft `
  --layer 12 `
  --pooling mean `
  --output outputs/hubert_large_l12/embeddings.npz
```

## BEATs

官方代码：`microsoft/unilm/beats`。官方 checkpoint 通过 `BEATs.py` 中的
`BEATsConfig` 和 `BEATs.extract_features` 加载。

需要先准备官方 checkpoint，并让 `BEATs.py` 在 import path 中。推荐：

```powershell
git clone https://github.com/microsoft/unilm.git external/unilm
```

示例：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model beats `
  --repo-path external/unilm/beats `
  --checkpoint-path checkpoints/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt `
  --pooling mean `
  --output outputs/beats/embeddings.npz
```

## ATST

官方代码：`Audio-WestlakeU/audiossl`，使用
`audiossl.methods.atstframe.embedding.load_model` 和 scene/timestamp embedding helper。

示例：

```powershell
pip install -e external/audiossl

extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model atst `
  --checkpoint-path checkpoints/atstframe_base.ckpt `
  --representation scene `
  --output outputs/atst_scene/embeddings.npz
```

如果要抽 timestamp/frame-level 表示：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model atst `
  --checkpoint-path checkpoints/atstframe_base.ckpt `
  --representation timestamp `
  --pooling mean_std `
  --output outputs/atst_timestamp/embeddings.npz
```

## Stable Audio Open VAE

官方代码：`Stability-AI/stable-audio-tools`。wrapper 会调用
`get_pretrained_model`，优先使用模型上的 `encode`，否则查找 generation model 的
`pretransform.encode`。

示例：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_test_manifest.csv `
  --model stable-audio-open-vae `
  --model-name stabilityai/stable-audio-open-1.0 `
  --pooling mean `
  --output outputs/stable_audio_open_vae/embeddings.npz
```

## Music2Latents

官方代码：`SonyCSLParis/music2latent`，使用 `music2latent.EncoderDecoder.encode`。

默认：

- `--sample-rate 44100`
- `--channels 2`
- `--extract-features` 默认打开

示例：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model music2latents `
  --pooling mean `
  --output outputs/music2latents/embeddings.npz
```

如果需要指定本地 inference checkpoint：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model music2latents `
  --load-path-inference checkpoints/music2latent.pt `
  --output outputs/music2latents_local/embeddings.npz
```

## WavCube

官方代码：`yanghaha0908/WavCube`。官方特征抽取路径是
`Vocos.from_config(...).feature_extractor.infer(audio)`。

需要 config 和 checkpoint：

```powershell
git clone https://github.com/yanghaha0908/WavCube.git external/WavCube

extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model wavcube `
  --repo-path external/WavCube `
  --config-path external/WavCube/configs/wavcube.yaml `
  --checkpoint-path checkpoints/wavcube.ckpt `
  --pooling mean `
  --output outputs/wavcube/embeddings.npz
```

## SpeechTokenizer

官方代码：`ZhangXInFD/SpeechTokenizer`。默认会用 `huggingface_hub` 下载
`fnlp/SpeechTokenizer`，也可以传本地 `config.json` 和 `SpeechTokenizer.pt`。

默认表示：

- `--representation quantized`: 默认使用 `forward_feature` 的第一层连续 RVQ embedding
  (`--layers 0`)，也就是 quantization 后的第一个连续层
- `--representation semantic`: 使用 semantic token 层输出
- `--representation codes`: 使用离散 code IDs 转 float 后 pooling

示例：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model speechtokenizer `
  --representation quantized `
  --pooling mean `
  --output outputs/speechtokenizer/embeddings.npz
```

本地 checkpoint：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model speechtokenizer `
  --config-path checkpoints/SpeechTokenizer/config.json `
  --checkpoint-path checkpoints/SpeechTokenizer/SpeechTokenizer.pt `
  --representation quantized `
  --output outputs/speechtokenizer_local/embeddings.npz
```

## X-Codec

官方模型 wrapper 已集成在 Hugging Face Transformers 的 `XcodecModel`。默认表示是
`--representation quantized`：先得到进入 quantizer 的连续 latent，再只取第一个
codebook/quantizer 解码后的连续 embedding。`pre_quantized` 和 `codes` 仍然保留，
但需要显式指定。

另有两个用于分析 quantization 之前分支 latent 的独立 tokenizer 名称：

- `x-codec-ssl-latent`: 取 SSL/semantic encoder 的最后一层 hidden state
  (`semantic_model(...).hidden_states[-1]`)
- `x-codec-vae-latent`: 取 VAE/acoustic encoder 的最后一层 latent
  (`acoustic_encoder(...)`)

示例：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model x-codec `
  --model-name hf-audio/xcodec-hubert-general `
  --representation quantized `
  --pooling mean `
  --output outputs/xcodec/embeddings.npz
```

semantic 和 acoustic 分支 concat 融合后、quantization 前的连续 latent：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model x-codec-pre-quantized `
  --model-name hf-audio/xcodec-hubert-general `
  --pooling mean `
  --output outputs/xcodec_pre_quantized/embeddings.npz
```

quantization 后第一个连续 latent：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model x-codec-post-quantized `
  --model-name hf-audio/xcodec-hubert-general `
  --pooling mean `
  --output outputs/xcodec_post_quantized/embeddings.npz
```

SSL encoder 最后一层 latent：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model x-codec-ssl-latent `
  --model-name hf-audio/xcodec-hubert-general `
  --pooling mean `
  --output outputs/xcodec_ssl_latent/embeddings.npz
```

VAE encoder 最后一层 latent：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model x-codec-vae-latent `
  --model-name hf-audio/xcodec-hubert-general `
  --pooling mean `
  --output outputs/xcodec_vae_latent/embeddings.npz
```

## MuCodec

官方代码：`tencent-ailab/MuCodec`。项目使用官方 `generate.MuCodec` 加载模型，
但不会走 `sound2code` 的离散 code 输出；而是读取 `fetch_codes_batch` 返回的
MuEncoder 连续 latent，再通过 `rvq_muencoder_emb` 的第一个 quantizer，取
quantization 后的第一个连续层做 pooling。

`--layer-num` 沿用官方 helper 的语义：传入的层号会由官方 `MuCodec` wrapper 内部转换成
0-based encoder layer index，用来决定送入 quantizer 的 MuEncoder 层。默认是 `7`，
作为声学细节和高层语义之间的折中。

示例：

```powershell
git clone https://github.com/tencent-ailab/MuCodec.git external/MuCodec

extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model mucodec `
  --repo-path external/MuCodec `
  --checkpoint-path checkpoints/mucodec.pt `
  --layer-num 7 `
  --pooling mean `
  --output outputs/mucodec/embeddings.npz
```

## DAC

官方代码：`descriptinc/descript-audio-codec`。项目使用官方 `dac.DAC` 模型加载和
`model.encode(..., n_quantizers=1)`，默认取第一个 RVQ/codebook 量化后的连续表示
`z`，不使用 quantizer 之前的 `latents`，也不把离散 `codes` 当 embedding。

默认会下载官方 `44khz` / `8kbps` 权重；也可以用 `--model-name 24khz`、
`--model-name 16khz`，或通过 `--checkpoint-path` 指定本地权重。44.1kHz 的 16kbps
模型可以传 `--model-bitrate 16kbps`。

示例：

```powershell
extract-tokenizer-embeddings `
  --manifest outputs/nsynth_valid_manifest.csv `
  --model dac `
  --model-name 44khz `
  --representation quantized `
  --pooling mean `
  --output outputs/dac/embeddings.npz
```
