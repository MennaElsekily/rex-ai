import json
from pathlib import Path

from agents.orchestrator import Orchestrator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "datasets"
    / "andalusia_catalogue"
)


# Arabic Unicode character range
def contains_arabic(value):
    if isinstance(value, dict):
        return any(contains_arabic(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_arabic(v) for v in value)
    if isinstance(value, str):
        return any("\u0600" <= char <= "\u06FF" for char in value)
    return False


def test_arabic_in_data_files():
    print("\n--- Checking Arabic text in JSON data files ---")

    files = sorted(DATA_DIR.glob("*.json"))
    assert files, f"No JSON files found in {DATA_DIR}"

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        has_arabic = contains_arabic(data)
        print(f"{file_path.name}: Arabic text found = {has_arabic}")

        if file_path.name == "eval_set.json":
            assert has_arabic, f"No Arabic text found in {file_path}"


def test_arabic_request_routing():
    print("\n--- Testing Arabic request routing ---")

    orchestrator = Orchestrator()

    examples = [
        ("أريد دورة في البرمجة للمبتدئين", ["rex_assistant"]),
        ("ابحث عن مناقصات تعليمية", ["rfp_research"]),
        ("أنشئ منهج تدريبي", ["curriculum_research"]),
        ("اكتب حملة تسويقية", ["marketing"]),
        (
            "أنشئ منهج تدريبي ثم حملة تسويقية",
            ["curriculum_research", "marketing"],
        ),
    ]

    for request, expected in examples:
        language = orchestrator.detect_language(request)
        actual = orchestrator._make_plan(request)

        print(f"\nRequest: {request}")
        print(f"Detected language: {language}")
        print(f"Expected agents: {expected}")
        print(f"Actual agents:   {actual}")

        assert language == "ar", f"Arabic detection failed: {request}"
        assert actual == expected, (
            f"Routing mismatch for '{request}': "
            f"expected {expected}, got {actual}"
        )