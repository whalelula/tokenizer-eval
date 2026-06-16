# Setup

建议 Python 3.10+。可以用 conda 或 Python 自带 `venv`。

## Conda

```powershell
conda create -n tokenizer-eval python=3.10 -y
conda activate tokenizer-eval
```

先按运行环境安装 PyTorch。CUDA 11.8：

```powershell
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

CPU：

```powershell
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

只准备 NSynth manifest：

```powershell
pip install -e .
```

跑 MERT + t-SNE/指标/画图：

```powershell
pip install -e ".[mert,viz]"
```

跑 SAME + MERT + t-SNE/指标/画图：

```powershell
pip install -e ".[same,mert,viz]"
```

## Venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

## 可选依赖

- `mert`：安装 `transformers`，用于 MERT 推理。
- `viz`：安装 `matplotlib` 和 `scikit-learn`，用于 t-SNE、指标和可视化。
- `same`：安装官方 `stable-audio-3`，用于 SAME 推理。
- `stable-audio-open`：安装 PyPI 上的 `stable-audio-tools`，用于 Stable Audio Open VAE 推理。
- `eval`：等价于 `mert + viz`，不包含 SAME。

`pyproject.toml` 没有把 `torch/torchaudio` 放进基础依赖，是为了避免基础安装自动解析并下载体积很大的 PyTorch/CUDA wheel。建议先手动安装 PyTorch，再按任务安装 extras。

## 镜像源

PyPI 下载慢时可以换国内镜像：

```powershell
pip install -e ".[same,mert,viz]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```powershell
pip install -e ".[same,mert,viz]" -i https://mirrors.aliyun.com/pypi/simple
```

如果 `same` 的 Git 依赖安装失败，也可以直接安装官方仓库：

```powershell
pip install git+https://github.com/Stability-AI/stable-audio-3.git
```

镜像源主要解决 PyPI 依赖包下载问题；如果失败原因是无法访问 GitHub，本机仍然需要能访问 `https://github.com/Stability-AI/stable-audio-3.git`。

`stable-audio-open` 使用 PyPI 发布的 `stable-audio-tools`，GitHub 连接不稳定时也可以直接安装：

```powershell
pip install -e ".[stable-audio-open]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Hugging Face 镜像

AutoDL 等环境下载 Hugging Face 模型较慢时：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0
```

## 官方实现依据

- SAME：Stability AI 官方 `stable-audio-3` 的 `AutoencoderModel.from_pretrained(...).encode(...)` 推理方式。
- SAME-S 模型页：https://huggingface.co/stabilityai/SAME-S
- MERT：官方仓库和 Hugging Face 用法，使用 `Wav2Vec2FeatureExtractor` + `AutoModel` 并读取 `hidden_states`。
- MERT-v1-95M 模型页：https://huggingface.co/m-a-p/MERT-v1-95M
- NSynth：官方 `examples.json` + `audio/*.wav` 结构。
