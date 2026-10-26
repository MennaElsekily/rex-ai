
import json
import re
from pathlib import Path
from typing import Any

from agents.text_utils import normalize_for_matching


class CurriculumData:
    """Loads and provides access to curriculum example data."""

    def __init__(self, data_path: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]

        self.data_path = (
            Path(data_path)
            if data_path is not None
            else project_root / "data" / "curriculum_examples.json"
        )

        self.records = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        """Load curriculum records from the JSON file."""

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Curriculum data file not found: {self.data_path}"
            )

        with self.data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = data.get("curricula", data.get("examples", []))
        else:
            raise ValueError(
                "Curriculum JSON must contain a list or an object."
            )

        if not isinstance(records, list):
            raise ValueError(
                "Expected curriculum records to be stored in a list."
            )

        if not all(isinstance(record, dict) for record in records):
            raise ValueError(
                "Every curriculum record must be a JSON object."
            )

        return records

    def get_all(self) -> list[dict[str, Any]]:
        """Return all loaded curriculum examples."""

        return self.records.copy()

    @staticmethod
    def _title_text(record: dict[str, Any]) -> str:
        return normalize_for_matching(str(record.get("title", "")))

    @staticmethod
    def _searchable_text(record: dict[str, Any]) -> str:
        """Title, audience, outcomes and module titles, normalized."""

        title = str(record.get("title", ""))
        audience = str(record.get("target_audience", ""))

        outcomes = " ".join(
            str(item) for item in (record.get("learning_outcomes") or [])
        )

        modules = " ".join(
            str(module.get("title", ""))
            for module in (record.get("modules") or [])
            if isinstance(module, dict)
        )

        return normalize_for_matching(
            f"{title} {audience} {outcomes} {modules}"
        )

    def search(self, keyword: str) -> list[dict[str, Any]]:
        """Search relevant curriculum fields by keyword."""

        return self.search_with_match_type(keyword)[0]

    def search_with_match_type(
        self, keyword: str
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Search and say how good the match is.

        match_type is "all" (empty query), "exact" (the whole phrase,
        or every term, was found), or "partial" (only some of the
        requested topics matched -- the closest examples).
        """

        keyword = keyword.strip().casefold()

        if not keyword:
            return self.get_all(), "all"

        texts = [
            (record, self._searchable_text(record))
            for record in self.records
        ]
        normalized_keyword = normalize_for_matching(keyword)

        # 1. The whole phrase appears in a record (original behaviour),
        #    but starting at a word boundary -- and as a whole word when
        #    it is short -- so "ai" no longer matches "trainers".
        exact = [
            record for record, text in texts
            if phrase_in_text(normalized_keyword, text)
        ]
        if exact:
            return exact, "exact"

        # 2. Match the topics and words inside a longer request.
        concepts, tokens = extract_search_terms(normalized_keyword)

        if not concepts and not tokens:
            # Only generic words. Listing every example is right for a
            # browse request ("show me a curriculum"), not for a message
            # that merely sounds like one ("make it 40 hours").
            if _BROWSE_RE.search(normalized_keyword):
                return self.get_all(), "all"
            return [], "exact"

        scored: list[tuple[int, dict[str, Any]]] = []
        for record, text in texts:
            if all(concept_matches(c, text) for c in concepts) and all(
                token_matches(t, text) for t in tokens
            ):
                title = self._title_text(record)
                score = sum(
                    3 if concept_matches(c, title) else 1 for c in concepts
                ) + sum(
                    3 if token_matches(t, title) else 1 for t in tokens
                )
                scored.append((score, record))

        if scored:
            scored.sort(key=lambda item: -item[0])
            return [record for _, record in scored], "exact"

        # 3. Not everything matched: fall back to the records matching
        #    the most requested *topics* (never loose single words).
        counts = [
            (sum(concept_matches(c, text) for c in concepts), record)
            for record, text in texts
        ]
        best = max((count for count, _ in counts), default=0)

        if best == 0:
            return [], "exact"

        return [record for count, record in counts if count == best], "partial"


# --------------------------------------------------------------------
# Query understanding for search
# --------------------------------------------------------------------

# A "concept" is a topic users name in several ways (English and
# Arabic). `query` forms are looked for in the request; `record` forms
# are looked for in the curriculum text. Short forms such as "ai" are
# matched as whole words only, so "ai" never matches "main" or "email".
_CONCEPTS: dict[str, dict[str, tuple[str, ...]]] = {
    "ai": {
        "query": (
            "artificial intelligence", "ai", "a.i.",
            "الذكاء الاصطناعي", "ذكاء اصطناعي", "الذكاء الصناعي",
        ),
        "record": ("ai", "artificial intelligence"),
    },
    "data_analysis": {
        "query": (
            "data analysis", "data analytics", "analytics",
            "تحليل البيانات", "تحليل بيانات",
        ),
        "record": ("data analysis", "data analytics"),
    },
    "cybersecurity": {
        "query": (
            "cybersecurity", "cyber security", "cyber-security",
            "information security", "الأمن السيبراني", "امن سيبراني",
            "الامن السيبراني",
        ),
        "record": ("cybersecurity", "cyber security"),
    },
    "project_management": {
        "query": (
            "project management", "ادارة المشاريع", "إدارة المشاريع",
            "ادارة المشروعات", "إدارة المشروعات", "ادارة مشاريع",
        ),
        "record": ("project management",),
    },
    "instructional_design": {
        "query": (
            "instructional design", "التصميم التعليمي", "تصميم تعليمي",
        ),
        "record": ("instructional design",),
    },
    "programming": {
        "query": (
            "programming", "python", "coding", "javascript",
            "البرمجة", "برمجة", "بايثون",
        ),
        "record": ("programming", "python", "coding", "javascript"),
    },
    "data": {
        "query": ("البيانات", "بيانات"),
        "record": ("data",),
    },
}

# Words that carry no topic (English and Arabic, normalized).
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at",
    "by", "with", "about", "from", "into", "over", "is", "are", "be",
    "it", "its", "this", "that", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "us", "can", "could", "would", "should", "will",
    "do", "does", "need", "needs", "want", "wants", "like", "looking",
    "look", "find", "show", "give", "get", "search", "see", "list",
    "tell", "please", "any", "some", "all", "more", "new", "there",
    "what", "which", "who", "how", "have", "has", "help", "curriculum",
    "curricula", "curriculums", "course", "courses", "syllabus",
    "syllabi", "training", "trainings", "program", "programs",
    "programme", "programmes", "outline", "outlines", "example",
    "examples", "sample", "samples", "available", "existing", "related",
    "relevant", "learning", "outcomes", "outcome", "objectives", "hour",
    "hours", "hr", "hrs", "create", "generate", "build", "make", "skills",
    "skill", "knowledge", "basics", "basic", "introduction", "intro",
    "introductory", "fundamentals", "essentials", "topic", "topics",
    "subject", "subjects", "material", "materials", "plan", "plans",
    "one", "ones", "something", "anything",
    # Arabic
    "اريد", "ابحث", "عن", "منهج", "مناهج", "منهاج", "تدريبي", "تدريبيه",
    "تدريبيا", "لي", "ما", "هل", "اعطني", "اعطيني", "اظهر", "اعرض",
    "دوره", "دورات", "في", "من", "الي", "علي", "هو", "هي", "و", "ثم",
    "مع", "بعض", "كل", "هذا", "هذه", "ذلك", "يمكن", "ممكن", "لو",
    "سمحت", "محتاج", "ابغي", "احتاج", "هناك", "تعليمي", "دراسي",
})


def _form_pattern(form: str) -> re.Pattern[str]:
    form = normalize_for_matching(form)
    if form.isascii():
        words = r"\s+".join(re.escape(part) for part in form.split())
        return re.compile(rf"(?<!\w){words}(?!\w)")
    return re.compile(re.escape(form))


_QUERY_FORMS = sorted(
    (
        (concept, _form_pattern(form), len(form))
        for concept, spec in _CONCEPTS.items()
        for form in spec["query"]
    ),
    key=lambda item: -item[2],
)
_RECORD_PATTERNS = {
    concept: [_form_pattern(form) for form in spec["record"]]
    for concept, spec in _CONCEPTS.items()
}
_WORD_RE = re.compile(r"[a-z][a-z0-9+#-]*|[؀-ۿ]+")


def extract_search_terms(normalized_query: str) -> tuple[list[str], list[str]]:
    """Split a request into known topic concepts and remaining content words."""

    working = normalized_query
    concepts: list[str] = []

    # Longest forms first, and consume what they match, so that
    # "تحليل البيانات" is one topic, not "analysis" plus "data".
    for concept, pattern, _ in _QUERY_FORMS:
        if pattern.search(working):
            working = pattern.sub(" ", working)
            if concept not in concepts:
                concepts.append(concept)

    tokens: list[str] = []
    for word in _WORD_RE.findall(working):
        if len(word) < 2 or word in _STOPWORDS or word in tokens:
            continue
        tokens.append(word)

    return concepts, tokens


_BROWSE_RE = re.compile(
    r"curricul|syllab|course|program|example|sample|"
    r"منهج|مناهج|منهاج|دوره|دورات"
)


def phrase_in_text(phrase: str, normalized_text: str) -> bool:
    """Phrase found starting at a word boundary (whole word if short)."""

    pattern = rf"(?<!\w){re.escape(phrase)}"
    if len(phrase) <= 3:
        pattern += r"(?!\w)"

    return re.search(pattern, normalized_text) is not None


def concept_matches(concept: str, normalized_text: str) -> bool:
    return any(
        pattern.search(normalized_text)
        for pattern in _RECORD_PATTERNS[concept]
    )


def token_matches(token: str, normalized_text: str) -> bool:
    if not token.isascii():
        return token in normalized_text

    stem = token[:-1] if token.endswith("s") and len(token) > 3 else token

    if len(stem) >= 5:
        # Prefix match, so "educat..." style variants and plurals work.
        return re.search(rf"(?<!\w){re.escape(stem)}", normalized_text) is not None

    return (
        re.search(rf"(?<!\w){re.escape(stem)}s?(?!\w)", normalized_text)
        is not None
    )
