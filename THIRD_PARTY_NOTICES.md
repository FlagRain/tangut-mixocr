# 第三方组件

本仓库的代码和 mixocr 权重按 [LICENSE](LICENSE)（MIT）发布。下列文件来自第三方，保留各自的许可。

## PP-OCRv6 small 文本检测模型

- 文件：`weights/det/PP-OCRv6_det_small.onnx`
- 来源：PaddlePaddle [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)；ONNX 版本由 [RapidOCR](https://github.com/RapidAI/RapidOCR) 转换发布。
- 许可：Apache License 2.0。

## Noto Serif Tangut 字体

- 文件：`web/static/fonts/NotoSerifTangut.woff2`
- 版权：Copyright 2022 The Noto Project Authors（https://github.com/notofonts/tangut）
- 许可：SIL Open Font License 1.1（https://openfontlicense.org）。

## 训练时用到、但不随仓库发布的字体

训练合成数据时用过以下字体。它们不在本仓库中，重新训练需要自行取得：

- Noto Serif Tangut、Tangut N4694、Tangut Yinchuan（西夏）
- Windows 自带的宋体、仿宋、楷体、黑体、微软雅黑、细明体，华文宋体、华文仿宋、华文楷体，方正小标宋，Noto Serif/Sans SC/TC（汉）
- Times New Roman、Arial、Calibri、Cambria（拉丁字母）
