<div align="center">

# tangut-mixocr

**Line-Level Mixed Tangut–Chinese OCR & Proofreading System**  
**夏汉混排整行 OCR 识别模型与校对系统**

[![Live Demo](https://img.shields.io/badge/Demo-ocr.flagrain.cn-blue?style=flat-square)](https://ocr.flagrain.cn)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime%20(CPU)-005CED?style=flat-square)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

[**English**](#-english) | [**简体中文**](#-简体中文)

</div>

---

<a id="-english"></a>

## Overview

**tangut-mixocr** is a lightweight line-level OCR model that recognizes mixed Tangut–Chinese text—along with Latin phonetic transcriptions, punctuation, and footnote markers—in a single pass without script routing or character-level segmentation.

### Key Features

- **Unified Script Recognition**: Built on a **5.7M-parameter CNN + BiLSTM + CTC** architecture over a unified **14,666-character vocabulary** (covering all 5,878 Unicode Tangut characters, 8,428 Han characters, Latin transcriptions, punctuation, and markers).
- **High Accuracy & Zero Script Confusion**: Achieves **98.9%** accuracy on Tangut characters in scanned Tangutological monographs, with **0.00%** of Tangut characters misread as Chinese.
- **CPU-Friendly Inference**: Runs smoothly on a standard 2-core CPU via **ONNX Runtime** (no PyTorch dependency required for inference).
- **Full-Stack Proofreading Web App**: Ships with a self-hostable web application (`FastAPI` + vanilla HTML/JS) supporting image/PDF upload, line-by-line interactive correction (one-click candidate replacement, Tangut lookup by phonetic reconstruction or Chinese gloss), admin review, and training data export.

---

## Benchmarks

| Evaluation Set | `mixocr-v1` Accuracy |
| :--- | :---: |
| **Scanned Monograph Pages — Tangut** (874 chars, manual ground truth) | **98.9%** |
| └─ *Tangut misread as Han characters* | *0.00%* |
| **Scanned Monograph Pages — Han** (agreement rate with PP-OCRv5\*) | **99.0%** |
| **Scanned Dictionary Headwords** (208 items, verified against printed index) | **99.5%** |
| **Typeset Lecture Notes — Tangut / Han** (67 born-digital lines) | **98.6% / 97.6%** |

> \* *Note: The Han character benchmark measures agreement against high-confidence outputs from PP-OCRv5 rather than manual ground truth.*

---

## Quick Start (CLI)

Requires **Python 3.10+**. Inference runs purely on ONNX Runtime and does not require PyTorch.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Recognize a full page (auto text-line detection + line-by-line recognition)
python ocr.py --page page.png

# 3. Recognize pre-cropped text line images
python ocr.py line1.png line2.png

# 4. Batch recognize specific pages of a PDF and export to JSONL
python ocr.py --pdf book.pdf --pages 120-121 --out result.jsonl
```

---

## Web Application

A live instance of this project is hosted at **[https://ocr.flagrain.cn](https://ocr.flagrain.cn)**.

The `web/` directory contains a complete, self-hostable web pipeline:  
`Upload Image / PDF` → `Line-by-Line OCR` → `Interactive Proofreading` (instant candidate swap; search Tangut by phonetic reconstruction or Chinese gloss) → `Admin Review` → `Training Data Export`.

```bash
pip install -r web/requirements.txt
python web/tools/prepare_models.py
cd web && TANGUT_OCR_ADMIN_TOKEN=your_secret_token python -m app
```

---

## Repository Structure

```text
├── ocr.py                 # CLI inference script (ONNX Runtime, no PyTorch required)
├── weights/               # Model weights: mixocr-v1.pt (PyTorch), mixocr-v1.onnx, charset.json
│   ├── det/               # PP-OCRv6 small text-line detection model (PaddleOCR, Apache-2.0)
│   └── init/              # Legacy pretrained weights for initialization
├── mixocr/                # Training and evaluation package
│   ├── core/              # Model definition, preprocessing, charset, synthetic/real data, metrics
│   ├── run/               # Scripts for training, evaluation, ONNX export, textbank & pseudo-labeling
│   └── data/charset.json  # Unified 14,666-character vocabulary (shared by training & inference)
└── web/                   # Web app (FastAPI + vanilla HTML/JS), deployment scripts & docs
```

---

## Training

Training requires PyTorch and a CUDA-enabled GPU (`mixocr-v1` was trained for 60,000 steps in ~9 hours on a laptop RTX 3080 8GB):

```bash
pip install -r requirements-train.txt

# Train from scratch
python mixocr/run/train.py --name my-run --steps 60000

# Fine-tune / continue training from v1 weights
python mixocr/run/train.py --name v2 --init weights/mixocr-v1.pt --steps 20000

# Export checkpoint to ONNX
python mixocr/run/export_onnx.py mixocr/runs/v2/best.pt
```

### Training Data & External Resources

Training samples are synthesized on the fly. However, reproducing `v1`-level performance requires several external resources specified via environment variables (see the top of `mixocr/core/config.py` for the full list):

| Resource | Environment Variable / Command | Fallback Behavior (if omitted) |
| :--- | :--- | :--- |
| **Fonts** (Tangut: *Noto Serif Tangut*, *Tangut N4694*, *Yinchuan*; Chinese: Songti, etc.) | `MIXOCR_FONT_DIRS` | Searches system font directories automatically (requires ≥1 Tangut font and ≥1 Chinese font) |
| **Textbank** (real Chinese sentences, Tangut texts, mixed lecture lines) | `MIXOCR_TEXTBANK` | Falls back to random character sampling from the vocabulary |
| **Tangut Phonetics & Glosses** | `MIXOCR_TANGUT_TSV`, `MIXOCR_LEXICON` | Disables synthetic "dictionary-entry style" layout generation |
| **Scanned Dictionary Headwords** (real cropped Tangut characters) | Run `cd mixocr && python -m core.real` to build `mixocr/data/real/dict_heads.pkl` (requires `MIXOCR_HEAD_ANCHORS`, etc.) | Trains without mixing real scanned headwords |
| **Evaluation Sets** | Run `run/build_evalsets.py` to build `mixocr/data/eval/*.pkl` | Trains without validation-based model selection |

> **Note on Data Availability**: The textbank, evaluation sets, and scanned dictionary images are derived from copyrighted publications and personal research materials, and therefore are not redistributed in this repository. All scripts used to build them are provided under `mixocr/run/`.

---

## License & Acknowledgements

The source code and `mixocr` model weights are released under the [MIT License](LICENSE). Third-party components included in this repository retain their respective licenses (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details):

- **PP-OCRv6 small** text-line detection model (PaddleOCR, Apache-2.0 License)
- **Noto Serif Tangut** web font (SIL Open Font License 1.1)

The initial weights for `mixocr-v1` were initialized from an earlier line-level OCR model trained by the author in 2024.

---
---

<a id="-简体中文"></a>

## 项目简介

**tangut-mixocr** 是一个夏汉混排整行 OCR 识别模型：将西夏字、汉字、标点、注码和拉丁拟音字母放在同一个统一字表里一次读出，无需预先切分字种。

### 特性

- **夏汉统一字表识别**：基于 **5.7M 参数量的 CNN + BiLSTM + CTC** 架构，词表共收录 **14,666 个字符**（涵盖全部 5,878 个 Unicode 西夏字、8,428 个汉字，以及拉丁拟音字母、标点与注码）。
- **高准确率与零字种误判**：在西夏学专著扫描书页上，西夏字识别准确率达 **98.9%**（其中被误识为汉字的比例为 **0.00%**）。
- **轻量级 CPU 推理**：基于 **ONNX Runtime** 推理，无需安装 PyTorch，在普通 2 核 CPU 上即可流畅运行。
- **校对网站**：配套完整的自托管 Web 应用（`FastAPI` + 原生 HTML/JS），支持图片/PDF 批量上传、逐行校对、按拟音或汉义检索西夏字及导出训练数据。

---

## 评测表现

| 评测集 | `mixocr-v1` 准确率 |
| :--- | :---: |
| **扫描书页西夏字**（874 字，人工转写真值） | **98.9%** |
| └─ *其中错成汉字* | *0.00%* |
| **扫描书页汉字**（与 PP-OCRv5 的一致率\*） | **99.0%** |
| **词典扫描字头**（208 个，按印刷编号核对） | **99.5%** |
| **讲义西夏字 / 汉字**（电子排版 67 行） | **98.6% / 97.6%** |

> \* *注：汉字一项的参照标准是 PP-OCRv5 的高置信度输出，而非人工校对真值。*

---

## 快速使用

环境要求：**Python 3.10** 及以上（命令行推理仅依赖 ONNX Runtime，无需 PyTorch）。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 整页识别：自动检测文本行并逐行识别
python ocr.py --page 整页.png

# 3. 识别已切好的单行或多行图像
python ocr.py 行图1.png 行图2.png

# 4. 批量识别 PDF 指定页码并输出为 JSONL
python ocr.py --pdf 书.pdf --pages 120-121 --out 结果.jsonl
```

---

## 在线网站与本地部署

本项目的模型已部署于个人网站：**[https://ocr.flagrain.cn](https://ocr.flagrain.cn)**

`web/` 目录包含一个完整的可自托管网站工作流：  
`上传图片或 PDF` → `逐行识别` → `逐行校对`（候选字一点即换；支持按拟音、汉义查西夏字）→ `审核后台` → `导出训练数据`。

```bash
pip install -r web/requirements.txt
python web/tools/prepare_models.py
cd web && TANGUT_OCR_ADMIN_TOKEN=换成你的口令 python -m app
```

---

## 仓库结构

```text
├── ocr.py                 # 命令行识别脚本（基于 ONNX，不需要 PyTorch）
├── weights/               # 模型权重：mixocr-v1.pt（PyTorch）、mixocr-v1.onnx、charset.json
│   ├── det/               # PP-OCRv6 small 文本行检测模型（PaddleOCR，Apache-2.0）
│   └── init/              # 旧模型的初始权重
├── mixocr/                # 训练与评测代码
│   ├── core/              # 模型定义、预处理、字表、合成数据、真实数据、评测指标
│   ├── run/               # 训练、评测、导出 ONNX、构建文本库/评测集/伪标签
│   └── data/charset.json  # 14,666 字的统一字表（训练和推理共用）
└── web/                   # 网站前后端（FastAPI + 原生 HTML/JS）、部署脚本与说明
```

---

## 模型训练

训练需要 PyTorch 和一块显卡（`mixocr-v1` 在 8 GB 显存的笔记本 RTX 3080 上训练 60,000 步，耗时约 9 小时）：

```bash
pip install -r requirements-train.txt

# 从头训练
python mixocr/run/train.py --name my-run --steps 60000

# 从 v1 权重继续微调训练
python mixocr/run/train.py --name v2 --init weights/mixocr-v1.pt --steps 20000

# 导出最优权重为 ONNX 格式
python mixocr/run/export_onnx.py mixocr/runs/v2/best.pt
```

### 训练资源与环境变量配置

训练数据全部采用在线合成策略，但要复现接近 `v1` 的识别效果，还需要配置几项不随仓库发布的外部资源（均通过环境变量指定，完整列表见 `mixocr/core/config.py` 开头）：

| 资源 | 环境变量 / 生成方式 | 缺失时的默认回退行为 |
| :--- | :--- | :--- |
| **字体**（西夏：Noto Serif Tangut、Tangut N4694、银川体；汉字：宋体等） | `MIXOCR_FONT_DIRS` | 自动在系统字体目录中查找；至少需要一套西夏字体和一套汉字字体 |
| **文本库**（真实汉文句子、西夏文原文、讲义混排行） | `MIXOCR_TEXTBANK` | 使用字表里的随机字符代替 |
| **西夏字拟音表、单字释义** | `MIXOCR_TANGUT_TSV`、`MIXOCR_LEXICON` | 不生成“字典式释义”合成版式 |
| **词典扫描字头**（真实扫描的西夏字） | 运行 `cd mixocr && python -m core.real` 生成 `mixocr/data/real/dict_heads.pkl`（需 `MIXOCR_HEAD_ANCHORS` 等） | 训练时不混入真实扫描字头 |
| **评测集** | 运行 `run/build_evalsets.py` 生成 `mixocr/data/eval/*.pkl` | 仅执行训练，不按验证集指标挑模型 |

> **数据版权说明**：文本库、评测集和词典扫描图像来自受版权保护的出版物及整理者的个人资料，因此不随仓库公开分发；用于生成这些数据的构建脚本均保留在 `mixocr/run/` 目录下。

---

## 许可与致谢

本项目代码和 `mixocr` 模型权重以 [MIT 许可](LICENSE) 发布。仓库中包含的第三方组件保留各自的原始许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：

- 文本行检测模型 **PP-OCRv6 small**（PaddleOCR，Apache-2.0 许可）；
- 网页端西夏字体 **Noto Serif Tangut**（SIL OFL 1.1 许可）。

`mixocr-v1` 的初始权重来自作者 2024 年训练的旧版整行识别模型。

感谢Claude Opus 5.5在代码上的支持。