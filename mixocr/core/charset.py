import json
import unicodedata
from collections import Counter

import torch

from . import config as C

PATH = C.CHARSET_JSON

PUNCT = ("（）［］【】〔〕〈〉《》「」『』“”‘’、。，：；？！…—–·・～％＋－／＝＜＞〇"
         "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽❶❷❸❹❺❻❼❽❾❿㈠㈡㈢㈣㈤"
         "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ¹²³⁴⁵⁶⁷⁸⁹⁰°§※△○●□■☆★→←↑↓×÷±．｜")
LATIN = ("ɨəɣŋɦʔɕʑʂʐɛɔɯʷʲʰˑːɑæøœɐɪʊʌɒβθðʃʒçɲɳʈɖɭɽ"
         "àáâãäåèéêëìíîïòóôõöùúûüýÿāēīōūăĕĭŏŭǎěǐǒǔǜśźńṅṇṃṛṣṭḍḥẓẹịọụạỵĩẽũỹ"
         "̣̱̀́̂̃̄̆̇̈̌")


def is_han(ch):
    cp = ord(ch)
    return 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x323AF


def build(v8_chars):
    chars = list(v8_chars)
    have = set(chars)
    add = []
    for line in open(C.require(C.CHINA_8105, "通用规范汉字表码位文件"), encoding="utf-8"):
        if line.startswith("U+"):
            add.append(chr(int(line.split()[0][2:], 16)))
    cnt = Counter(ch for p in ("zh", "mixed") for ch in (C.TEXTBANK / f"{p}.txt").read_text(encoding="utf-8")
                  if is_han(ch))
    add += [ch for ch, n in cnt.most_common() if n >= 3]
    add += list(PUNCT) + list(unicodedata.normalize("NFC", LATIN))
    for ch in add:
        if ch not in have and not ch.isspace():
            chars.append(ch)
            have.add(ch)
    return ["<b>"] + chars


def load():
    return json.loads(PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    ck = torch.load(C.CODEX_V8, map_location="cpu", weights_only=False)
    cs = build(ck["charset"][1:])
    PATH.write_text(json.dumps(cs, ensure_ascii=False), encoding="utf-8")
    print(f"v8 {len(ck['charset']) - 1} → {len(cs) - 1} 字（另加空白）；汉字 {sum(map(is_han, cs[1:]))}")
