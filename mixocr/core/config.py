"""路径与字体配置

  MIXOCR_FONT_DIRS     额外的字体目录（用系统路径分隔符隔开）。字体按文件名在这些目录和系统字体目录里找
  MIXOCR_TANGUT_TSV    西夏字表 TSV（列：li, unicode, GHC, …, GX2021, …），用于字典式释义版式
  MIXOCR_LEXICON       单字释义 JSONL（每行 source_text 西夏字、target_text 汉义），同上
  MIXOCR_TEXTBANK      文本库目录（zh.txt / tg.txt / mixed.txt），默认 mixocr/data/textbank；没有就用随机字串
  MIXOCR_DATA          生成的数据（评测集、词典字头、伪标签行），默认 mixocr/data
"""
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../tangut-mixocr/mixocr
REPO = ROOT.parent
WEIGHTS = REPO / "weights"
CHARSET_JSON = ROOT / "data" / "charset.json"


def _env_path(name, default=None):
    v = os.environ.get(name)
    return Path(v) if v else default


DATA = _env_path("MIXOCR_DATA", ROOT / "data")
RUNS = _env_path("MIXOCR_RUNS", ROOT / "runs")
RESULTS = _env_path("MIXOCR_RESULTS", ROOT / "results")
TEXTBANK = _env_path("MIXOCR_TEXTBANK", ROOT / "data" / "textbank")

# 继续训练的起点
CODEX_V8 = WEIGHTS / "init" / "codex-line-ctc-v8.pt"

TANGUT_TSV = _env_path("MIXOCR_TANGUT_TSV")
LEXICON = _env_path("MIXOCR_LEXICON")
CHINA_8105 = _env_path("MIXOCR_CHINA_8105")

# 重建文本库 / 评测集 / 伪标签时用到的原始资料
ANNOTATION_DIR = _env_path("MIXOCR_ANNOTATION_DIR")     # 课程标注 *.md
LECTURE_DIR = _env_path("MIXOCR_LECTURE_DIR")           # 讲义 PDF
BOOK_OCR = _env_path("MIXOCR_BOOK_OCR")                 # 扫描书逐页 PP-OCR 结果：<书>/p0001.json + page_images/
DICT_RAW_PAGES = _env_path("MIXOCR_DICT_RAW_PAGES")     # 词典逐页 PP-OCR 结果
JLYX_REALTEST = _env_path("MIXOCR_JLYX_REALTEST")       # 真实书页西夏文行测试集（lines.jsonl + crops/）
LECTURE_DEV = _env_path("MIXOCR_LECTURE_DEV")           # 讲义 dev 行图与真值
SCAN_HOLDOUT = _env_path("MIXOCR_SCAN_HOLDOUT")         # 词典扫描留出集 locked_test.jsonl
HEAD_ANCHORS = _env_path("MIXOCR_HEAD_ANCHORS")         # 词典字头候选 heads.jsonl
SOURCE_ROOT = _env_path("MIXOCR_SOURCE_ROOT")           # 上面这些记录里的相对图片路径以它为根
JLYX_TEST_PAGES = (50, 115)          # 《经律异相》整理研究中作测试的页，训练文本和训练图像都不用
LECTURE_DEV_DOCS = ("gr_e882bd36ae99", "gr_f2e6a7844d50", "gr_2f4dee1f94d8")


def require(path, what):
    if path is None or not Path(path).exists():
        raise SystemExit(f"缺少{what}。请用环境变量指定（见 mixocr/core/config.py 开头的说明）。")
    return Path(path)


# ---------------------------------------------------------------- 字体
FONT_DIRS = [Path(p) for p in os.environ.get("MIXOCR_FONT_DIRS", "").split(os.pathsep) if p] + [
    REPO / "fonts",
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts", Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".local/share/fonts", Path.home() / ".fonts",
    Path("/Library/Fonts"), Path("/System/Library/Fonts"), Path.home() / "Library/Fonts",
]

# name -> (文件名, ttc 序号, 可变字体实例)。v1 训练时用的是 Windows 上的字体；在 Linux 上可装 Noto CJK 代替
_FONT_FILES = {
    "tg_noto":       ("NotoSerifTangut-Regular.ttf", 0, None),
    "tg_n4694":      ("Tangut+N4694+V3.005.ttf", 0, None),
    "tg_yinchuan":   ("TangutYinchuan.ttf", 0, None),
    "han_simsun":    ("simsun.ttc", 0, None),
    "han_stsong":    ("STSONG.TTF", 0, None),
    "han_notoserif": ("NotoSerifSC-VF.ttf", 0, b"Regular"),
    "han_notoserif_b": ("NotoSerifSC-VF.ttf", 0, b"SemiBold"),
    "han_notoserif_tc": ("NotoSerifTC-VF.ttf", 0, b"Regular"),
    "han_fangsong":  ("simfang.ttf", 0, None),
    "han_stfangsong": ("STFANGSO.TTF", 0, None),
    "han_kai":       ("simkai.ttf", 0, None),
    "han_stkai":     ("STKAITI.TTF", 0, None),
    "han_hei":       ("simhei.ttf", 0, None),
    "han_yahei":     ("msyh.ttc", 0, None),
    "han_notosans":  ("NotoSansSC-VF.ttf", 0, b"Regular"),
    "han_stzhongsong": ("STZHONGS.TTF", 0, None),
    "han_mingliu":   ("mingliu.ttc", 0, None),
    "han_xbs":       ("FZXBSJW.TTF", 0, None),
    "han_extb":      ("simsunb.ttf", 0, None),
    "han_cjk_serif": ("NotoSerifCJK-Regular.ttc", 2, None),      # Linux 常见（2 = 简体中文）
    "han_cjk_sans":  ("NotoSansCJK-Regular.ttc", 2, None),
    "lat_times":     ("times.ttf", 0, None),
    "lat_timesi":    ("timesi.ttf", 0, None),
    "lat_arial":     ("arial.ttf", 0, None),
    "lat_calibri":   ("calibri.ttf", 0, None),
    "lat_cambria":   ("cambria.ttc", 0, None),
    "lat_dejavu_serif": ("DejaVuSerif.ttf", 0, None),
}


@lru_cache(maxsize=None)
def _find_font(filename):
    low = filename.lower()
    for d in FONT_DIRS:
        if not d.exists():
            continue
        p = d / filename
        if p.exists():
            return p
        for q in d.rglob("*"):
            if q.name.lower() == low:
                return q
    return None


def _fonts():
    out = {}
    for name, (fn, idx, inst) in _FONT_FILES.items():
        p = _find_font(fn)
        if p:
            out[name] = (p, idx, inst)
    return out


FONTS = _fonts()

_HAN_WEIGHTS = {"han_simsun": 6, "han_stsong": 4, "han_notoserif": 3, "han_notoserif_b": 1, "han_notoserif_tc": 1,
                "han_fangsong": 2, "han_stfangsong": 1, "han_kai": 2, "han_stkai": 1, "han_hei": 1, "han_yahei": 1,
                "han_notosans": 1, "han_mingliu": 1, "han_xbs": 1, "han_cjk_serif": 4, "han_cjk_sans": 1}
_LATIN_WEIGHTS = {"lat_times": 4, "lat_timesi": 1, "lat_arial": 1, "lat_calibri": 1, "lat_cambria": 2,
                  "lat_dejavu_serif": 2}

TANGUT_FONTS = [n for n in ("tg_noto", "tg_n4694", "tg_yinchuan") if n in FONTS]
HAN_FONTS = {n: w for n, w in _HAN_WEIGHTS.items() if n in FONTS and n != "han_stzhongsong"}
HAN_VAL_FONT = "han_stzhongsong" if "han_stzhongsong" in FONTS else next(iter(HAN_FONTS), None)  # 合成验证集专用
HAN_FALLBACK = [n for n in ("han_simsun", "han_notoserif", "han_cjk_serif", "han_mingliu", "han_extb") if n in FONTS]
LATIN_FONTS = {n: w for n, w in _LATIN_WEIGHTS.items() if n in FONTS}
