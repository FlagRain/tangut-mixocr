# tangut-mixocr：夏汉混排整行 OCR

**English summary.** A line-level OCR model that reads mixed Tangut–Chinese text (plus Latin transcription, punctuation and footnote markers) in a single pass, with no script routing. It is a 5.7M-parameter CNN + BiLSTM + CTC recogniser over a unified 14,666-character set (all 5,878 Unicode Tangut characters, 8,428 Han characters). On scanned pages of a Tangut studies monograph it reads **98.9%** of Tangut characters correctly (none misread as Chinese). It runs on a 2-core CPU through ONNX Runtime, and ships with a self-hostable web app for crowd-sourced correction.
---

一个整行识别模型：西夏字、汉字、标点、注码和拟音字母放在同一个字表里，一次读出，不需要先分字种。]


| 评测集  | mixocr-v1 |
| ---  | --- |
| 扫描书页西夏字（874 字，人工转写真值） | **98.9%** |
| 其中错成汉字 | 0.00% |
| 扫描书页汉字（与 PP-OCRv5 的一致率*） | **99.0%** |
| 词典扫描字头（208 个，按印刷编号核对） | 99.5% |
| 讲义西夏字 / 汉字（电子排版 67 行）| 98.6% / 97.6% |

\* 汉字一项的参照是 PP-OCRv5 的高置信度输出，不是人工真值

## 快速使用

需要 Python 3.10 以上。

```bash
pip install -r requirements.txt
python ocr.py --page 整页.png                    # 整页：自动找行，逐行识别
python ocr.py 行图1.png 行图2.png                 # 已经切好的行图
python ocr.py --pdf 书.pdf --pages 120-121 --out 结果.jsonl
```

## 网站

`web/` 是一个完整的网站：上传图片或 PDF → 逐行识别 → 逐行校对（候选字一点即换；按拟音、汉义查西夏字）→ 审核后台 → 导出训练数据。

```bash
pip install -r web/requirements.txt
python web/tools/prepare_models.py 
cd web && TANGUT_OCR_ADMIN_TOKEN=换成你的口令 python -m app 
```

## 仓库结构

```
ocr.py                 命令行识别（ONNX，不需要 PyTorch）
weights/               mixocr-v1.pt（PyTorch）、mixocr-v1.onnx、charset.json
  det/                 PP-OCRv6 small 文本行检测模型（PaddleOCR，Apache-2.0）
  init/                旧模型的权重
mixocr/                训练代码
  core/                模型、预处理、字表、合成数据、真实数据、评测
  run/                 训练、评测、导出 ONNX、建文本库/评测集/伪标签
  data/charset.json    14,666 字的字表（训练和推理共用）
web/                   网站（FastAPI + 原生 HTML/JS）、部署脚本与说明
```

## 训练

训练需要 PyTorch 和一块显卡（v1 在 8 GB 的笔记本 RTX 3080 上训练 60,000 步，约 9 小时）：

```bash
pip install -r requirements-train.txt
python mixocr/run/train.py --name my-run --steps 60000
python mixocr/run/train.py --name v2 --init weights/mixocr-v1.pt --steps 20000   # 从 v1 继续
python mixocr/run/export_onnx.py mixocr/runs/v2/best.pt
```

训练数据全部在线合成，但要接近 v1 的效果，还需要几样不随仓库发布的东西，都用环境变量指定（完整列表见 `mixocr/core/config.py` 开头）：

| 资源 | 环境变量 | 没有时 |
| --- | --- | --- |
| 字体：西夏 Noto Serif Tangut、Tangut N4694、银川体；汉字宋体等 | `MIXOCR_FONT_DIRS` | 自动在系统字体目录里找；至少要一套西夏字体和一套汉字字体 |
| 文本库（真实汉文句子、西夏文原文、讲义混排行） | `MIXOCR_TEXTBANK` | 用字表里的随机字代替 |
| 西夏字拟音表、单字释义 | `MIXOCR_TANGUT_TSV`、`MIXOCR_LEXICON` | 不生成“字典式释义”版式 |
| 词典扫描字头（真实扫描的西夏字） | `cd mixocr && python -m core.real` 生成 `mixocr/data/real/dict_heads.pkl`（需 `MIXOCR_HEAD_ANCHORS` 等） | 不混入真实字头 |
| 评测集 | `run/build_evalsets.py` 生成 `mixocr/data/eval/*.pkl` | 只训练、不挑模型 |

文本库、评测集和词典扫描图像来自受版权保护的出版物和整理者的个人资料，所以不发布；生成它们的脚本都在 `mixocr/run/`。

## 许可与致谢

代码和 mixocr 权重以 MIT 许可发布（[LICENSE](LICENSE)）。仓库里的第三方组件保留各自的许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：

- 文本行检测模型 PP-OCRv6 small（PaddleOCR，Apache-2.0）；
- 网页里的西夏字体 Noto Serif Tangut（SIL OFL 1.1）。

mixocr-v1 的初始权重来自作者2024年训练的旧的整行模型。
