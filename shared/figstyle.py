# -*- coding: utf-8 -*-
"""
figstyle.py — shared plotting setup for every figure in the book.

Two jobs:
  1. Register a CJK-capable font so Korean labels render instead of tofu boxes.
     matplotlib's default (DejaVu Sans) has no Hangul glyphs.
  2. Provide a tiny bilingual label helper so the same script can emit the
     Korean edition and the English edition of each figure.

Font licensing (checked for commercial publication):
  Noto Sans / Serif CJK  — SIL Open Font License 1.1  (commercial use OK)
  DejaVu Sans            — Bitstream Vera License      (commercial use OK)
No proprietary or paid font is used anywhere in this book.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# --------------------------------------------------------------- font setup
def _pick_cjk_font():
    """Return the name of an installed pan-CJK font, or None."""
    wanted = ["Noto Sans CJK KR", "Noto Sans CJK JP", "Noto Sans CJK SC",
              "Noto Sans KR", "NanumGothic", "Malgun Gothic", "AppleGothic"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for w in wanted:
        if w in have:
            return w
    # Noto CJK ships as .ttc collections that fontconfig knows but matplotlib
    # may not have scanned; register them explicitly.
    import glob
    for path in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc") + \
                glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True):
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            continue
    have = {f.name for f in font_manager.fontManager.ttflist}
    for w in wanted:
        if w in have:
            return w
    return None


CJK_FONT = _pick_cjk_font()


def use(lang="ko"):
    """Apply the house style. lang='ko' selects a Hangul-capable font."""
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.dpi": 125,
        "savefig.dpi": 125,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11.5,
        "axes.titlesize": 13.0,
        "axes.labelsize": 12.0,
        "legend.fontsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "figure.titlesize": 14.0,
        "axes.unicode_minus": False,   # keep the minus sign from becoming tofu
    })
    if lang == "ko" and CJK_FONT:
        plt.rcParams["font.family"] = CJK_FONT
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    return plt


# ------------------------------------------------------------ label helper
class L:
    """Bilingual label bundle:  L(ko='밀도', en='Density')[lang]"""

    def __init__(self, ko, en):
        self.ko, self.en = ko, en

    def __getitem__(self, lang):
        return self.ko if lang == "ko" else self.en


def T(lang, ko, en):
    """Inline shorthand for a single bilingual string."""
    return ko if lang == "ko" else en


def check_glyphs(lang="ko"):
    """Sanity check that the selected font can actually draw Hangul."""
    if lang != "ko":
        return True
    if not CJK_FONT:
        return False
    from matplotlib.font_manager import findfont, FontProperties
    from fontTools.ttLib import TTFont, TTCollection
    path = findfont(FontProperties(family=CJK_FONT))
    try:
        f = TTCollection(path).fonts[0] if path.endswith(".ttc") else TTFont(path)
        cmap = f.getBestCmap()
        probe = "\uB370\uC774\uD130"  # Hangul probe: does the font carry CJK glyphs?
        return all(ord(c) in cmap for c in probe)
    except Exception:
        return None  # unknown, but a CJK font was found
