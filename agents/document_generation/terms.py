"""
Word lists for the language checks of generated text (validation.py).

TECHNICAL_TERMS: lowercase English terms (tools, libraries, standards, abbreviations) that may appear as they are inside
Arabic text. Capitalised, upper-case, hyphenated or digit-containing words (Python, SQL, scikit-learn, F1, LO1) are never
checked, and words that occur in the approved curriculum are always allowed, so this list only needs the lowercase names
a model might add on its own. To allow another term, add it here (lowercase, one per entry).
"""

from __future__ import annotations

TECHNICAL_TERMS = frozenset("""
python java javascript typescript sql nosql html css json xml yaml csv tsv api rest graphql git github gitlab docker
kubernetes linux ubuntu windows macos android ios excel powerpoint word tableau powerbi spark hadoop hive kafka airflow
tensorflow pytorch keras scikit sklearn pandas numpy scipy matplotlib seaborn plotly jupyter colab anaconda conda pip
aws azure gcp cloud ai ml nlp gpu cpu ram http https url uri ide vscode react node nodejs django flask fastapi streamlit
mongodb mysql postgresql sqlite redis etl bi kpi roi ui ux saas devops ssh vpn dns tcp lan wan ocr pdf xlsx docx
""".split())

# Words an Arabic or English text cannot end on: a sentence that does is cut off (compared after normalisation).
DANGLING_ENGLISH = frozenset({"and", "or", "but", "the", "a", "an", "which", "such", "like"})
DANGLING_ARABIC = frozenset({"و", "او", "ثم", "في", "علي", "على", "من", "الي", "الى", "عن", "مع", "ان", "التي", "الذي", "مثل",
                             "لكن", "حتي", "حتى", "كما"})
