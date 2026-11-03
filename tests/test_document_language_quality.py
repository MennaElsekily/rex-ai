"""
Language-quality checks found by inspecting the real-model PDFs (Arabic Question Bank GDOC-865b6e1f785a, Arabic
Instructor Guide GDOC-81fb190585e4, Arabic Presentation Outline GDOC-5e81f4d59f88, Arabic and English Assessment
Rubrics GDOC-ff5dec0a5864 / GDOC-2894bbfe153b): Arabic "all of the above", options reused across questions, English
words in Arabic text, cut-off text, rambling answers, near-identical rubric levels, instructions that repeat the purpose.
The texts below are the model's own output copied from those documents.
"""

import copy

import pytest

from agents.document_generation import templates, units
from agents.document_generation.validation import (
    MAX_ANSWER_WORDS,
    MAX_EXPLANATION_WORDS,
    _distinct_levels,
    check_unit,
    curriculum_words,
    foreign_word_problems,
    incomplete_problems,
    rambling,
)
from tests.curriculum_fixtures import valid_curriculum
from tests.document_fakes import container_answer

QB = templates.TEMPLATES["question_bank"]
RUBRIC = templates.TEMPLATES["assessment_rubric"]


def codes(problems):
    return [p["code"] for p in problems]


def unit_for(template, language, key, unit_id, mutate=None):
    content = valid_curriculum(language)
    data = next(u for u in container_answer(template.doc_type, language, content)[key] if u.get("module_id", u.get("assessment_id")) == unit_id)
    data = copy.deepcopy(data)
    if mutate:
        mutate(data)
    return template.unit_schema.model_validate(data), content


# ---- the real Arabic Question Bank (GDOC-865b6e1f785a, original model text) ----------------------------------------

Q1 = ["تحديد البيانات التي يجب استبعادها", "استخدام الفلاتر لاستبعاد البيانات غير المرغوب فيها", "تحليل البيانات لتحديد الأنماط والعيوب",
      "تحديد العناصر غير الصحيحة في البيانات"]
Q2 = [Q1[1], Q1[2], Q1[3], Q1[0]]                                  # the same four options, reordered
Q5 = ["تحسين دقة التصنيف", "تقليل كمية البيانات المطلوبة", "زيادة سرعة التدريب", "جميع الأطراف أعلاه"]
Q6 = ["تقنيات تحليل البيانات", "تقنيات تدريب النماذج", "تقنيات تقييم النماذج", "جميع الأطراف أعلاه"]
Q4_ANSWER = (
    "الخطوات الأساسية التي يجب اتباعها في عملية تنظيف البيانات هي تحديد البيانات التي يجب استبعادها، استخدام الفلاتر لاستبعاد "
    "البيانات غير المرغوب فيها، وتحليل البيانات لتحديد الأنماط والعيوب. يجب تجنب البيانات التي تحتوي على العناصر غير الصحيحة في "
    "بداية عملية تنظيف البيانات، وتحديد العناصر غير الصحيحة في البيانات بعد تحديد البيانات التي يجب استبعادها. يجب أيضًا معالجة "
    "البيانات المكررة والقيم الفارغة والقيم غير الصحيحة في نهاية عملية تنظيف البيانات. يجب أيضًا تجنب البيانات التي تحتوي على "
    "العناصر غير المرغوب فيها أو التي لا تتوافق مع المتطلبات المحددة للتحليل. يجب أيضًا تجنب البيانات التي تحتوي على العناصر "
    "التي قد تؤثر على النتائج أو تؤثر على النموذج الذي سيتم بناؤه من البيانات المتبقي. يجب أيضًا،")
Q4_EXPLANATION = (
    "يجب توضيح الخطوات الأساسية التي يجب اتباعها في عملية تنظيف البيانات، بما في ذلك تحديد البيانات التي يجب استبعادها، استخدام "
    "الفلاتر لاستبعاد البيانات غير المرغوب فيها، وتحليل البيانات لتحديد الأنماط والعيوب. يجب أيضًا توضيح الخطوات التي يجب اتباعها "
    "في نهاية عملية تنظيف البيانات، بما في ذلك معالجة البيانات المكررة والقيم الفارغة والقيم غير الصحيحة، وتجنب البيانات التي "
    "تحتوي على العناصر غير المرغوب فيها أو التي لا تتوافق مع المتطلبات المحددة للتحليل.")
GLUED = "تقنيات التصنيف العكسي أو التصنيف المع تهدف إلى تحسين دقة التصنيف، وهي متعلقة بموضوع هذا المodule."


def mc(question, options, correct=0, rationale="لأنه الخيار الصحيح."):
    return {"question": question, "options": list(options), "correct_index": correct, "rationale": rationale}   # (a copy: tests change options)


def arabic_unit(mutate=None, module="M1"):
    def build(data):
        data["multiple_choice"] = [mc("ما هي الخطوة الأولى في تنظيف البيانات؟", Q1),
                                   mc("ما هي الخطوة التالية بعد تحديد البيانات التي يجب استبعادها؟", Q2, 3)]
        data["true_false"] = [{"statement": "يجب تجنب البيانات التي تحتوي على العناصر غير الصحيحة في بداية عملية تنظيف البيانات.",
                               "answer": True, "rationale": "هذا صحيح."}]
        data["short_answer"] = [{"question": "ما هي الخطوات الأساسية التي يجب اتباعها في عملية تنظيف البيانات؟",
                                 "model_answer": "تحديد البيانات التي يجب استبعادها ثم تحليلها.", "rationale": "خطوات التنظيف."}]
        if mutate:
            mutate(data)
    return unit_for(QB, "ar", "modules", module, build)


def test_the_baseline_arabic_unit_used_below_is_otherwise_valid_except_for_the_reordered_options():
    unit, content = arabic_unit()
    assert codes(check_unit(unit, QB, content)) == ["question_options"]                 # only the reused option set


# ---- "all / none of the above" in Arabic and English ---------------------------------------------------------------

@pytest.mark.parametrize("option", [
    "جميع الأطراف أعلاه",                      # the inspected PDFs (question 5 and 6)
    "جميع ما سبق", "كل ما سبق", "جميع الإجابات", "كل الخيارات", "جميع الخيارات أعلاه", "جميع الإجابات السابقة".replace("السابقة", "أعلاه"),
    "الخيارات جميعها", "الإجابات كلها", "جميع ما ذكر", "كل ما ذُكر أعلاه", "لا شيء مما سبق", "لا شيء مما ذكر", "لا أحد مما سبق",
    "كلاهما", "كلا الخيارين",
    "All of the above", "None of the above", "all of these", "None of these", "All the above", "Both of the above",
    "Neither of the above", "All options", "All answers are correct", "None of the answers", "All choices",
])
def test_an_option_that_stands_for_the_other_options_is_rejected(option):
    unit, content = arabic_unit(lambda d: d["multiple_choice"][0]["options"].__setitem__(3, option)) if any("؀" <= c <= "ۿ" for c in option) \
        else unit_for(QB, "en", "modules", "M1", lambda d: d["multiple_choice"][0]["options"].__setitem__(3, option))
    messages = [p["message"] for p in check_unit(unit, QB, content) if "all/none" in p["message"]]
    assert messages and "option 4" in messages[0]                                         # names the option


@pytest.mark.parametrize("option", [
    "تحسين دقة التصنيف", "كل شيء عن البيانات", "جميع الطلاب", "الخيارات متعددة الأنواع", "لا يوجد مما سبق ذكره في الدرس".replace("لا يوجد", "يوجد"),
    "All students", "None of the rows", "Remove all outliers", "Both numeric and categorical columns", "Neither of the two classes",
    "Select all features", "Use every option listed in the data dictionary".replace("every option", "each column"),
])
def test_ordinary_options_are_not_mistaken_for_all_or_none_of_the_above(option):
    unit, content = arabic_unit(lambda d: d["multiple_choice"][0]["options"].__setitem__(2, option)) if any("؀" <= c <= "ۿ" for c in option) \
        else unit_for(QB, "en", "modules", "M1", lambda d: d["multiple_choice"][0]["options"].__setitem__(2, option))
    assert not any("all/none" in p["message"] for p in check_unit(unit, QB, content))


# ---- the same options used by different questions ----------------------------------------------------------------------

def test_the_inspected_questions_one_and_two_use_the_same_options_in_a_different_order():
    unit, content = arabic_unit()
    problems = [p for p in check_unit(unit, QB, content) if p["code"] == "question_options"]

    assert len(problems) == 1 and problems[0]["target"] == "multiple_choice:1"           # the later question is the one to rewrite
    assert "Multiple-choice question 2" in problems[0]["message"] and "multiple-choice question 1" in problems[0]["message"]
    assert "different order" in problems[0]["message"]


def test_three_of_four_options_in_common_is_also_reused():
    unit, content = unit_for(QB, "en", "modules", "M1", lambda d: d["multiple_choice"][1].update(
        options=[*d["multiple_choice"][0]["options"][:3], "A brand new fourth option"]))
    assert any("uses (nearly) the same options" in p["message"] for p in check_unit(unit, QB, content))


def test_two_in_common_and_all_different_options_are_fine():
    unit, content = unit_for(QB, "en", "modules", "M1", lambda d: d["multiple_choice"][1].update(
        options=[*d["multiple_choice"][0]["options"][:2], "A new third option", "A new fourth option"]))
    assert not any("same options as" in p["message"] for p in check_unit(unit, QB, content))
    good, content = unit_for(QB, "en", "modules", "M1")
    assert check_unit(good, QB, content) == []


def test_options_reused_from_an_earlier_module_are_found():
    first, content = unit_for(QB, "en", "modules", "M1")
    second, _ = unit_for(QB, "en", "modules", "M2", lambda d: d["multiple_choice"][0].update(options=list(reversed(first.multiple_choice[0].options))))
    problems = check_unit(second, QB, content, (first,))
    assert any("of module M1" in p["message"] and p["target"] == "multiple_choice:0" for p in problems)


# ---- English words in Arabic text ----------------------------------------------------------------------------------------

def test_the_inspected_instructor_guide_bullets_with_uso_are_rejected():
    allowed = curriculum_words(valid_curriculum("ar"))
    problems = foreign_word_problems("uso من الألعاب التفاعلية لشرح مفاهيم التصنيف", allowed)
    assert codes(problems) == ["foreign_word"] and '"uso"' in problems[0]["message"]


@pytest.mark.parametrize("text, shown", [
    ("يستخدم الطلاب أدوات مثل بايثون لتنفيذ التدريب والevaluating.", "والevaluating"),          # the inspected presentation outline
    (GLUED, "المodule"),                                                                           # the inspected question bank
])
def test_latin_letters_glued_to_arabic_letters_are_rejected(text, shown):
    problems = foreign_word_problems(text, curriculum_words(valid_curriculum("ar")))
    assert codes(problems) == ["foreign_word"] and shown in problems[0]["message"] and "attached to Arabic letters" in problems[0]["message"]


@pytest.mark.parametrize("text", [
    "يستخدم الطلاب Python و SQL لتنظيف البيانات.", "استخدم مكتبة scikit-learn لتدريب النموذج.", "درجة F1 مقياس للتقييم.",
    "تحدد LO1 مخرج التعلم.", "نستخدم pandas و numpy لقراءة البيانات.", "يعرض الدفتر Jupyter الناتج.", "بـPython يمكن ذلك.",
    "استخدم أداة (Python)، ثم SQL.", "الوحدة M1 تغطي ذلك.", "صيغة الملف CSV شائعة.", "تدريب النموذج باستخدام بايثون.", ""])
def test_technical_terms_names_and_arabic_text_are_accepted(text):
    assert foreign_word_problems(text, curriculum_words(valid_curriculum("ar"))) == []


def test_an_english_term_the_curriculum_uses_is_allowed_even_when_it_is_lowercase():
    content = valid_curriculum("ar")
    content["modules"][0]["topics"].append("dataframes")
    assert "dataframes" in curriculum_words(content)
    assert foreign_word_problems("نستخدم dataframes لتنظيم الجدول.", curriculum_words(content)) == []
    assert codes(foreign_word_problems("نستخدم dataframes لتنظيم الجدول.", curriculum_words(valid_curriculum("ar")))) == ["foreign_word"]


def test_the_check_runs_in_arabic_documents_only():
    english, content = unit_for(QB, "en", "modules", "M1", lambda d: d["short_answer"][0].update(model_answer="Clean the data then check it carefully."))
    assert not any(p["code"] == "foreign_word" for p in check_unit(english, QB, content))
    arabic, content = arabic_unit(lambda d: d["short_answer"][0].update(model_answer="uso من الأدوات لتنظيف البيانات."))
    assert any(p["code"] == "foreign_word" and p.get("target") == "short_answer:0" for p in check_unit(arabic, QB, content))


# ---- cut-off text ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["يجب أيضًا،", "Check the data first,", "Steps: cleaning;", "Check the data and", "Remove the", "Fix errors like",
                                  "نظف البيانات و", "نظف البيانات ثم", "استخدم الأدوات في", "الخطوة هي التحديد من", "Prepare the data, which"])
def test_text_that_ends_in_the_middle_of_a_sentence_is_flagged(text):
    problems = incomplete_problems(text)
    assert codes(problems) == ["incomplete_text"]


@pytest.mark.parametrize("text", ["Identify and handle missing values", "Cleaning removes errors.", "ما هي الخطوة الأولى؟", "يجب هذا أيضًا.",
                                  "هذا مهم أيضا", "Remove outliers", "What does the model learn from?", "Is data cleaning optional?", ""])
def test_complete_text_is_not_flagged_even_without_a_full_stop(text):
    assert incomplete_problems(text) == []


@pytest.mark.parametrize("stem, flagged", [("ما هي الخطوة الأولى في تنظيف البيانات", True), ("What is the first step", True),
                                          ("ما هي الخطوة الأولى في تنظيف البيانات؟", False), ("Why is data cleaned?", False),
                                          ("Explain why data is cleaned.", False), ("Data cleaning is optional!", False)])
def test_a_question_or_statement_must_end_with_terminal_punctuation(stem, flagged):
    assert bool(incomplete_problems(stem, stem=True)) is flagged


def test_the_inspected_dangling_model_answer_is_flagged_inside_a_unit():
    unit, content = arabic_unit(lambda d: (d.update(multiple_choice=[mc("ما الخطوة الأولى؟ تنظيف البيانات؟", Q1), mc("ما الخطوة الثانية لتنظيف البيانات؟", [*Q5[:3], "فحص البيانات"])]),
                                           d["short_answer"][0].update(model_answer="تحديد البيانات ثم تحليلها. يجب أيضًا،")))
    problems = check_unit(unit, QB, content)
    assert any(p["code"] == "incomplete_text" and p["target"] == "short_answer:0" for p in problems)


# ---- rambling answers and explanations ------------------------------------------------------------------------------------------

def test_the_inspected_rambling_model_answer_is_rejected_for_length_and_repetition():
    unit, content = arabic_unit(lambda d: (d["multiple_choice"][1].update(options=[*Q5[:3], "فحص البيانات"], question="ما الخطوة الثانية لتنظيف البيانات؟"),
                                           d["short_answer"][0].update(model_answer=Q4_ANSWER, rationale=Q4_EXPLANATION)))
    problems = [p for p in check_unit(unit, QB, content) if p.get("target") == "short_answer:0"]

    assert {"answer_too_long", "repeated_text", "incomplete_text"} <= set(codes(problems))
    assert len(Q4_ANSWER.split()) > MAX_ANSWER_WORDS and len(Q4_EXPLANATION.split()) > MAX_EXPLANATION_WORDS - 15
    assert "too long" in next(p for p in problems if p["code"] == "answer_too_long")["message"]


def test_a_loop_of_sentences_that_all_start_alike_is_rambling():
    # different sentences that all begin "يجب أيضًا" (the inspected answer did this six times)
    loop = ("يجب أيضًا مراجعة الجداول. يجب أيضًا حذف الصفوف الفارغة. يجب أيضًا توحيد صيغ التواريخ. "
            "يجب أيضًا فحص القيم الشاذة. يجب أيضًا حفظ نسخة احتياطية.")
    assert rambling(loop) == "يجب ايضا"
    assert rambling("يجب أيضًا مراجعة الجداول. يجب أيضًا حذف الصفوف الفارغة. يجب أيضًا توحيد صيغ التواريخ.") is None   # three is fine


@pytest.mark.parametrize("text", [
    "Precision measures exactness. Recall measures completeness. F1 balances both.",
    "Cleaning removes errors. It also fixes types. Finally it checks the result.",
    "Data cleaning is important for numeric data. It matters for categorical data too.",
    "يحدد التنظيف الأخطاء. ثم يصحح الأنواع. وأخيرا يتحقق من النتيجة.", "Short answer.", ""])
def test_ordinary_answers_are_not_rambling(text):
    assert rambling(text) is None


def test_the_same_sentence_nearly_repeated_is_rambling():
    assert rambling("Check that every column has the right type of data. Check that every column has the right types of data.") is not None


def _words(n):
    return " ".join(f"w{i}x" for i in range(n)) + "."


def test_word_limits_apply_to_question_bank_answers_and_explanations_only_at_the_limit():
    exactly, content = unit_for(QB, "en", "modules", "M1", lambda d: (
        d["short_answer"][0].update(model_answer=" ".join(["Cleaning"] + [f"step{i}" for i in range(MAX_ANSWER_WORDS - 1)])),
        d["multiple_choice"][0].update(rationale=" ".join(["Cleaning"] + [f"reason{i}" for i in range(MAX_EXPLANATION_WORDS - 1)]))))
    assert not any(p["code"] in ("answer_too_long", "explanation_too_long") for p in check_unit(exactly, QB, content))

    over, content = unit_for(QB, "en", "modules", "M1", lambda d: (
        d["short_answer"][0].update(model_answer=" ".join(["Cleaning"] + [f"step{i}" for i in range(MAX_ANSWER_WORDS)])),
        d["multiple_choice"][0].update(rationale=" ".join(["Cleaning"] + [f"reason{i}" for i in range(MAX_EXPLANATION_WORDS)])),
        d["true_false"][0].update(rationale=" ".join(["Cleaning"] + [f"why{i}" for i in range(MAX_EXPLANATION_WORDS)])),
        d["short_answer"][0].update(rationale=" ".join(["Cleaning"] + [f"why{i}" for i in range(MAX_EXPLANATION_WORDS)]))))
    found = {(p["code"], p["target"]) for p in check_unit(over, QB, content)}
    assert found >= {("answer_too_long", "short_answer:0"), ("explanation_too_long", "multiple_choice:0"),
                     ("explanation_too_long", "true_false:0"), ("explanation_too_long", "short_answer:0")}


def test_the_word_limits_do_not_apply_to_other_document_types():
    long_text = " ".join(f"word{i}" for i in range(150))          # (a 700-character field would still be refused by the schema)
    teaching = templates.TEMPLATES["instructor_guide"]
    unit, content = unit_for(teaching, "en", "modules", "M1", lambda d: d.update(instructor_notes=long_text[:690]))
    assert not any(p["code"] in ("answer_too_long", "explanation_too_long") for p in check_unit(unit, teaching, content))


# ---- rubric performance levels ---------------------------------------------------------------------------------------------------

# the four levels of each criterion of the inspected Arabic rubrics (GDOC-ff5dec0a5864, original model text)
AR_RUBRIC_GROUPS = {
    "A1/LO1": [
        "يبدأ الطلاب في تنظيف البيانات ببساطة، ويقومون بحذف البيانات غير الصحيحة أو غير المفيدة بشكل محدود.",
        "يستخدم الطلاب أدوات بسيطة لتنظيف البيانات، ويقومون بحذف البيانات غير الصحيحة بشكل محدود، ويقومون بتصحيح الأخطاء الصغيرة.",
        "يستخدم الطلاب أدوات متقدمة لتنظيف البيانات، ويقومون بحذف البيانات غير الصحيحة بشكل كبير، ويقومون بتصحيح الأخطاء الكبيرة.",
        "يستخدم الطلاب أدوات متقدمة لتنظيف البيانات، ويقومون بحذف البيانات غير الصحيحة بشكل كبير، ويقومون بتصحيح الأخطاء الكبيرة ويحلون المشكلات المعقدة."],
    "A1/LO2": [
        "يقوم الطلاب بتحليل البيانات وتقييم أداء النموذج بطرق بسيطة، ويستخدمون معايير بسيطة لتحديد الأخطاء.",
        "يستخدم الطلاب أدوات متقدمة لتحليل البيانات وتقييم أداء النموذج، ويستخدمون معايير متقدمة لتحديد الأخطاء.",
        "يستخدم الطلاب أدوات متقدمة لتحليل البيانات وتقييم أداء النموذج، ويستخدمون معايير متقدمة لتحديد الأخطاء ويحلون المشكلات المعقدة.",
        "يستخدم الطلاب أدوات متقدمة لتحليل البيانات وتقييم أداء النموذج، ويستخدمون معايير متقدمة لتحديد الأخطاء ويحلون المشكلات المعقدة ويقترحون تحسينات لتحسين أداء النموذج."],
    "CAP/LO2": [
        "يقوم بتدريب نموذج تصنيف بسيط باستخدام بيانات مجهزة بشكل بسيط ويقيم النتائج بشكل عام.",
        "يستخدم نماذج تصنيف متقدمة ويقيم النتائج ويقوم بتحليل النتائج ويقدم توصيات.",
        "يستخدم نماذج تصنيف متقدمة ويقيم النتائج ويقوم بتحليل النتائج ويقدم توصيات ويقوم بتحليل البيانات بشكل دقيق.",
        "يستخدم نماذج تصنيف متقدمة ويقيم النتائج ويقوم بتحليل النتائج ويقدم توصيات ويقوم بتحليل البيانات بشكل دقيق ويقدم توصيات عملية وفعالة."],
}
AR_RUBRIC_DISTINCT = [      # the one inspected Arabic criterion whose levels do differ in substance (CAP/LO1)
    "يبدأ بالتحضير الأولي للبيانات دون تدقيق دقيق للبيانات المقدمة.",
    "يستخدم أدوات بسيطة لتنظيف البيانات ويقوم بتصحيح الأخطاء البسيطة.",
    "يستخدم أدوات متخصصة لتنظيف البيانات ويقوم بتصحيح الأخطاء المعقدة ويتحقق من دقة البيانات.",
    "يستخدم أدوات متقدمة لتنظيف البيانات ويقوم بتصحيح الأخطاء المعقدة ويحقق من دقة البيانات ويقوم بتحليل البيانات بشكل دقيق."]
# the English criterion A1/LO2 of the inspected rubrics (GDOC-2894bbfe153b, page 3)
EN_RUBRIC_GOOD = [
    "The candidate describes a simple classification model but does not provide a clear rationale for its choice or how it will be evaluated.",
    "The candidate describes a simple classification model with a rationale for its choice and a basic plan for evaluation.",
    "The candidate describes a simple classification model with a clear rationale for its choice and a detailed plan for evaluation.",
    "The candidate describes a sophisticated classification model with a compelling rationale for its choice and a comprehensive plan for evaluation."]


@pytest.mark.parametrize("name", sorted(AR_RUBRIC_GROUPS))
def test_levels_that_are_the_previous_level_plus_a_clause_are_rejected(name):
    problems = _distinct_levels(AR_RUBRIC_GROUPS[name], name)
    assert codes(problems) == ["descriptors_similar"] and problems[0]["where"] == name


def test_the_inspected_arabic_levels_that_do_differ_and_the_english_levels_are_accepted():
    assert _distinct_levels(AR_RUBRIC_DISTINCT, "CAP/LO1") == []
    assert _distinct_levels(EN_RUBRIC_GOOD, "A1/LO2") == []


@pytest.mark.parametrize("a, b", [("Meets the standard in full.", "meets the standard in full"),
                                  ("Works with help from the tutor.", "Works with help from the tutor and with the group."),
                                  ("يعمل بمساعدة المدرب.", "يعمل بمساعدة المدرب وبمساعدة المجموعة.")])
def test_identical_or_contained_descriptions_are_the_same_description(a, b):
    assert _distinct_levels([a, b, "Completely different text that says other things.", "Another unrelated sentence here today."], "x")


def test_a_pair_that_differs_by_two_words_is_still_too_close():
    assert _distinct_levels(["Prepares the data with care.", "Prepares the data with skill.", "Does little at all here.", "Excels well beyond the rest."], "x")


def test_the_levels_of_the_arabic_rubric_are_checked_inside_a_unit():
    content = valid_curriculum("ar")
    unit, content = unit_for(RUBRIC, "ar", "rubrics", "A1", lambda d: d["criteria"][0].update(
        zip(("beginning", "developing", "proficient", "advanced"), AR_RUBRIC_GROUPS["A1/LO1"])))
    assert "descriptors_similar" in codes(check_unit(unit, RUBRIC, content))


# ---- assessor instructions that repeat the purpose ----------------------------------------------------------------------------------

@pytest.mark.parametrize("language, description, instructions", [
    ("en", "Prepare data and evaluate a model", "Prepare data and evaluate a model"),               # the inspected English rubric
    ("en", "Prepare data and evaluate a model", "prepare data and evaluate a model."),
    ("ar", "تجهيز بيانات وتقييم نموذج", "تجهيز بيانات وتقييم نموذج"),
    ("en", "Prepare data and evaluate a model", "Evaluate a model"),                                # nothing but words of the purpose
])
def test_instructions_that_repeat_the_purpose_are_rejected(language, description, instructions):
    unit, content = unit_for(RUBRIC, language, "rubrics", "A1", lambda d: d.update(instructions=instructions))
    assert content["assessments"][0]["description"] == description
    problems = [p for p in check_unit(unit, RUBRIC, content) if p["code"] == "instructions_repeat_purpose"]
    assert len(problems) == 1 and "how to assess" in problems[0]["message"]


@pytest.mark.parametrize("language, instructions", [
    ("en", "Review the notebook, then score each criterion by the evidence it shows and add the level points."),
    ("ar", "راجع الدفتر ثم قيّم كل بند بحسب الدليل الذي يظهره واجمع نقاط المستويات.")])
def test_instructions_that_explain_how_to_assess_are_accepted(language, instructions):
    unit, content = unit_for(RUBRIC, language, "rubrics", "A1", lambda d: d.update(instructions=instructions))
    assert not any(p["code"] == "instructions_repeat_purpose" for p in check_unit(unit, RUBRIC, content))


def test_the_inspected_arabic_bank_still_fails_for_its_other_problems():
    """The Arabic question 5 and 6 options and the glued word of the inspected bank (module M2) are each reported."""
    def build(data):
        data["multiple_choice"] = [mc("ما هو الهدف من تقنيات مثل التصنيف العكسي أو التصنيف المع؟", Q5),
                                   mc("ما هو التصنيف العكسي؟", Q6, 2, GLUED)]
        data["true_false"] = [{"statement": "التصنيف العكسي هو تقنية تقييم النماذج.", "answer": False, "rationale": "ليس كذلك."}]
        data["short_answer"] = [{"question": "كيف يمكن استخدام تقنية التصنيف العكسي لتحسين أداء نموذج التصنيف؟",
                                 "model_answer": "يستخدم التصنيف العكسي لتحسين الدقة.", "rationale": "تحسين الأداء."}]

    unit, content = unit_for(QB, "ar", "modules", "M2", build)
    problems = check_unit(unit, QB, content)

    assert [p["target"] for p in problems if "all/none" in p["message"]] == ["multiple_choice:0", "multiple_choice:1"]
    assert any(p["code"] == "foreign_word" and p["target"] == "multiple_choice:1" for p in problems)


# ---- the prompts tell the model what the checks will reject --------------------------------------------------------------------------

def _system_text(prompt):
    return prompt.format_messages(language_name="Arabic (Modern Standard Arabic)", curriculum_block="c", unit_block="u", problems_block="")[0].content


def test_the_question_prompt_states_the_new_rules():
    from agents.document_generation import prompts

    text = _system_text(prompts.MODULE_QUESTIONS)
    assert "never reuse the options of another question, even in a different order" in text
    assert "question mark or a full stop" in text and "at most 80 words" in text and "at most 60 words" in text
    assert "Never put English letters inside an Arabic word" in text
    assert 'Never use "all of the above" or "none of the above"' in text and "do not repeat a sentence" in text      # the earlier rules stay


def test_the_rubric_prompt_asks_for_different_levels_and_instructions_that_do_not_repeat_the_purpose():
    from agents.document_generation import prompts

    text = _system_text(prompts.ASSESSMENT_RUBRIC)
    assert "never write a level as the previous level with an extra clause" in text
    assert "explain HOW to assess" in text and "Do not repeat the assessment's description or purpose" in text
    assert "clearly differ from one another, from weakest to strongest" in text                                      # the earlier rule stays
