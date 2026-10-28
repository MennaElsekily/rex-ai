# Fonts for curriculum PDFs

Bundled so the PDFs look the same on every machine (no Arial or other
system fonts). Both families are licensed under the SIL Open Font
License 1.1, which allows bundling and embedding; the licence texts are
in `OFL-NotoNaskhArabic.txt` and `OFL-NotoSans.txt`.

| File | Use | Source (downloaded 2026-10-27) |
|---|---|---|
| `NotoNaskhArabic-Regular.ttf`, `NotoNaskhArabic-Bold.ttf` | Arabic text | https://github.com/notofonts/notofonts.github.io/tree/main/fonts/NotoNaskhArabic/hinted/ttf |
| `NotoSans-Regular.ttf`, `NotoSans-Bold.ttf` | Latin text, digits, punctuation | https://github.com/notofonts/notofonts.github.io/tree/main/fonts/NotoSans/hinted/ttf |

Noto Naskh Arabic contains every Arabic presentation form that
`arabic-reshaper` produces, but **no Latin letters**; Noto Sans has no
Arabic. `pdf.py` therefore switches font per run of text.
