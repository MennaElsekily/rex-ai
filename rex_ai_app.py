"""
REX AI -- one Streamlit app, two doors.

    User  -> REX Assistant directly (agents/user_service.py)
    Admin -> passcode checked on the server (agents/admin_service.py)
          -> Admin orchestrator -> RFP Research, Curriculum Research
             (REX Assistant excluded)

The User/Admin choice on the welcome screen is navigation only. Admin
access is granted by AdminService.login() against REX_ADMIN_PASSCODE, and
every admin action re-checks the session token on the server.

Admin dashboard: in-app alerts for new tenders, a tender search form (or
chat requests), the tender review queue (approve / reject, Egypt tenders
flagged for immediate review), the searchable dashboard of approved
tenders, and the audit trail of decisions.

Run:  streamlit run rex_ai_app.py
"""

from __future__ import annotations

import re
import uuid

import streamlit as st

from agents import admin_service as admin_module
from agents import register_agents
from agents import user_service as user_module
from agents.rfp_research.state import ALLOWED_COUNTRIES, TRAINING_CATEGORIES

# --------------------------------------------------------------------
# Interface text (English / Arabic)
# --------------------------------------------------------------------

T = {
    "en": {
        "tagline": "Find the right course, and the right opportunity.",
        "user_title": "I'm a learner",
        "user_desc": "Chat with REX: get course recommendations that fit your goals, "
                     "or ask about any course in our catalogue.",
        "user_button": "Start chatting",
        "admin_title": "I'm an admin",
        "admin_desc": "Find education and training tenders in Egypt and Saudi Arabia, "
                      "approve the right ones, and draft curricula.",
        "admin_button": "Admin access",
        "admin_disabled": "Admin access is disabled: set REX_ADMIN_PASSCODE in the .env file.",
        "passcode_title": "Admin access",
        "passcode": "Passcode",
        "unlock": "Unlock",
        "wrong_passcode": "That passcode is not correct.",
        "locked": "Too many attempts. Please wait a minute and try again.",
        "back": "Home",
        "new_chat": "New conversation",
        "chat_placeholder": "Ask REX about courses...",
        "rex_intro": "Hi, I'm **REX**. I can recommend courses that match your goals, "
                     "or answer questions about a specific course. You can write in "
                     "English or Arabic.",
        "examples": ["Help me choose a course", "What does the data analysis course cover?"],
        "sources": "Sources",
        "page": "p.",
        "thinking": "Thinking...",
        "rex_error": "Sorry, something went wrong. Please try again.",
        "dashboard": "Admin dashboard",
        "sign_out": "Sign out",
        "session_expired": "Your admin session has ended. Please unlock again.",
        "tab_review": "Tender review",
        "tab_search": "New tender search",
        "tab_agents": "Ask agents",
        "tab_dashboard": "Tender dashboard",
        "tab_audit": "Audit trail",
        "pending": "Pending",
        "approved": "Approved",
        "rejected": "Rejected",
        "all": "All",
        "flagged_pending": "Egypt: review now",
        "flagged": "Egypt · immediate review",
        "monitored": "Saudi Arabia · monitored",
        "priority": "Priority match",
        "alerts": "{n} new tender(s) found since you last checked "
                  "({f} in Egypt flagged for immediate review).",
        "mark_seen": "Mark as seen",
        "review_help": "Every tender RFP Research finds, from tender APIs and the web, waits "
                       "here for your decision. Approved tenders appear in the Tender dashboard.",
        "approve": "Approve",
        "reject": "Reject",
        "to_pending": "Back to pending",
        "open_source": "Source page",
        "open_submission": "Submission page",
        "no_items": "Nothing here. Run a New tender search to fill the queue.",
        "no_dashboard": "No approved tenders match.",
        "decided": "Marked as {status}.",
        "search_intro": "RFP Research searches the tender APIs, the official portals, tender "
                        "sites and the open web. Every tender found waits for your review.",
        "countries": "Countries",
        "categories": "Training categories",
        "categories_help": "Leave empty for all categories.",
        "priority_label": "Priority organisations or sectors (optional)",
        "priority_help": "Comma-separated, e.g. Ministry of Education, banking.",
        "eligibility_label": "What the academy can show for eligibility (optional)",
        "eligibility_help": "Used to judge qualification, e.g. registered training provider "
                            "in Egypt, ISO 9001.",
        "deadline_from": "Deadline from",
        "deadline_to": "Deadline to",
        "run_search": "Search tenders",
        "need_country": "Choose at least one country.",
        "searching": "Searching tender APIs and the web. This can take a few minutes...",
        "meta_country": "Country",
        "meta_org": "Organisation",
        "meta_category": "Category",
        "meta_reference": "Reference",
        "meta_published": "Published",
        "meta_source": "Source",
        "meta_reviewer": "Decided by",
        "deadline": "Deadline",
        "qualification": "Qualification",
        "eligibility": "Eligibility",
        "documents": "Documents",
        "missing": "Not stated by the source",
        "summary": "AI summary",
        "q_qualified": "Qualified",
        "q_not_qualified": "Not qualified",
        "q_unknown": "Unknown",
        "send": "Send",
        "agents_placeholder": "e.g. Find education tenders in Egypt",
        "agents_intro": "Requests are routed by the orchestrator to RFP Research or "
                        "Curriculum Research. Learner questions belong to REX in the "
                        "User interface.",
        "agents_examples": ["Find education and training tenders in Egypt and Saudi Arabia",
                            "Create a curriculum about data analysis"],
        "country": "Country",
        "category": "Category",
        "keyword": "Keyword (title, organisation, summary)",
        "no_audit": "No review decisions yet.",
        "details": "Details",
        # Approval and applications
        "approve_title": "Approve tender",
        "route_label": "How to apply",
        "automatic_yes": "REX AI can prepare this application (dry run: it is never sent; "
                         "you review and send it yourself).",
        "automatic_no": "Manual only: REX AI records the official steps for you.",
        "authorise_label": "I authorise REX AI to submit on the academy's behalf where supported",
        "authorise_help": "Without this, nothing is ever submitted for you. In this version "
                          "REX AI only prepares (dry run) and never sends anything.",
        "confirm_approve": "Approve",
        "cancel": "Cancel",
        "approved_flash": "Approved. Application: {status}.",
        "sub_status": {"not_started": "Not started", "in_progress": "In progress",
                       "submitted": "Submitted", "submission_failed": "Failed",
                       "requires_manual_action": "Needs manual action",
                       "outcome_unknown": "Result unknown"},
        "app_status": {"not_applied": "Not applied", "applied": "Applied", "awarded": "Awarded",
                       "not_awarded": "Not awarded", "withdrawn": "Withdrawn"},
        "filter_submission": "Submission",
        "filter_application": "Application",
        "submission": "Submission",
        "application": "Application",
        "authorised": "Submission authorised",
        "not_authorised": "Submission not authorised",
        "authorise_now": "Authorise submission",
        "prepare_again": "Prepare again",
        "missing_items": "Missing",
        "next_steps": "What to do",
        "prepared_email": "Prepared email (dry run: NOT sent)",
        "to": "To",
        "from": "From",
        "subject": "Subject",
        "attachments": "Attachments",
        "recommended_missing": "Recommended but not uploaded",
        "attempt_error": "Error",
        "record_submission": "Record manual submission",
        "submitted_on": "Submission date",
        "reference": "Reference or confirmation number",
        "submission_link": "Link (optional)",
        "notes": "Notes (optional)",
        "confirm_submitted": "I confirm the academy actually submitted this application",
        "need_confirm": "Tick the confirmation first.",
        "record": "Record",
        "unknown_help": "The last attempt's result is unknown. Check whether it was actually "
                        "sent before doing anything else.",
        "was_sent": "It was sent",
        "not_sent": "It was not sent",
        "outcome": "Outcome",
        "record_outcome": "Record outcome",
        "correct_applied": "Correct back to Applied",
        "history": "History",
        "metric_applied": "Applied",
        "metric_manual": "Needs manual action",
        "done": "Saved.",
        # Curricula (Agent 3)
        "tab_curricula": "Curricula",
        "curricula_intro": "Curricula drafted by the Curriculum Agent. Review a version, then approve it or "
                           "request changes. The agent can never approve; catalogue facts come from "
                           "demonstration data unless stated otherwise.",
        "cur_filter": "Status",
        "cur_all": "All",
        "cur_status": {"draft": "Draft", "pending_review": "Pending review",
                       "changes_requested": "Changes requested", "approved": "Approved"},
        "cur_none": "No curricula yet. Ask the agents, for example: \"Create a beginner curriculum about "
                    "data analysis, 40 hours\".",
        "cur_version": "Version",
        "cur_current": "current",
        "cur_meta": {"level": "Level", "target_audience": "Audience", "total_hours": "Total hours",
                     "language": "Language", "origin": "Origin", "change_summary": "Change in this version"},
        "cur_outcomes": "Learning outcomes",
        "cur_modules": "Modules",
        "cur_assessments": "Assessments",
        "cur_capstone": "Capstone project",
        "cur_no_capstone": "No capstone",
        "cur_validation_ok": "All validation rules passed.",
        "cur_validation_bad": "Validation problems (this version is a draft):",
        "cur_alignment": "Framework alignment",
        "cur_no_alignment": "No framework comparison (unable to determine without an approved framework).",
        "cur_references": "References",
        "cur_no_references": "No external research sources were used.",
        "cur_assumptions": "Assumptions and notes",
        "cur_pdf_download": "Download PDF",
        "cur_pdf_create": "Create PDF",
        "cur_review": "Review this version",
        "cur_notes": "Notes (required to request changes)",
        "cur_approve": "Approve",
        "cur_request_changes": "Request changes",
        "cur_not_reviewable": "Only the current version, while pending review, can be approved or sent back.",
        "cur_audit": "Review history",
        "cur_approved_msg": "Approved.",
        "cur_changes_msg": "Changes requested.",
        # Application profile
        "tab_profile": "Application profile",
        "profile_intro": "The academy's real details and documents, used only to prepare "
                         "applications. Nothing is invented: empty fields stay empty.",
        "save_profile": "Save details",
        "documents_title": "Documents",
        "doc_type": "Document type",
        "upload_file": "File (PDF, DOCX, PNG or JPG, up to 10 MB)",
        "upload": "Upload",
        "remove": "Remove",
        "no_documents": "No documents uploaded yet.",
        "field_label": {"legal_name": "Legal name",
                        "registration_number": "Commercial registration number",
                        "tax_number": "Tax registration number", "address": "Address",
                        "country": "Country of registration", "contact_person": "Contact person",
                        "contact_title": "Contact person's title", "contact_email": "Contact email",
                        "contact_phone": "Contact phone", "website": "Website"},
        "doc_label": {"company_profile": "Company profile",
                      "registration_certificate": "Commercial registration certificate",
                      "tax_card": "Tax card", "experience_references": "Experience / reference letters",
                      "cvs": "CVs of key staff", "other": "Other"},
    },
    "ar": {
        "tagline": "اعثر على الدورة المناسبة، والفرصة المناسبة.",
        "user_title": "أنا متعلم",
        "user_desc": "تحدث مع ركس: احصل على ترشيحات دورات تناسب أهدافك، "
                     "أو اسأل عن أي دورة في الكتالوج.",
        "user_button": "ابدأ المحادثة",
        "admin_title": "أنا مسؤول",
        "admin_desc": "ابحث عن مناقصات التعليم والتدريب في مصر والسعودية، واعتمد المناسب "
                      "منها، وأعدّ المناهج.",
        "admin_button": "دخول المسؤول",
        "admin_disabled": "دخول المسؤول غير مفعّل: اضبط REX_ADMIN_PASSCODE في ملف ‎.env.",
        "passcode_title": "دخول المسؤول",
        "passcode": "رمز الدخول",
        "unlock": "دخول",
        "wrong_passcode": "رمز الدخول غير صحيح.",
        "locked": "محاولات كثيرة. انتظر دقيقة ثم حاول مرة أخرى.",
        "back": "الرئيسية",
        "new_chat": "محادثة جديدة",
        "chat_placeholder": "اسأل ركس عن الدورات...",
        "rex_intro": "مرحبًا، أنا **ركس**. أستطيع ترشيح دورات تناسب أهدافك، أو الإجابة "
                     "عن أسئلتك حول دورة معينة. يمكنك الكتابة بالعربية أو الإنجليزية.",
        "examples": ["ساعدني أختار دورة", "إيه اللي بيغطيه كورس تحليل البيانات؟"],
        "sources": "المصادر",
        "page": "ص",
        "thinking": "جارٍ التفكير...",
        "rex_error": "عذرًا، حدث خطأ. حاول مرة أخرى.",
        "dashboard": "لوحة تحكم المسؤول",
        "sign_out": "تسجيل الخروج",
        "session_expired": "انتهت جلسة المسؤول. أدخل رمز الدخول مرة أخرى.",
        "tab_review": "مراجعة المناقصات",
        "tab_search": "بحث جديد عن مناقصات",
        "tab_agents": "الوكلاء",
        "tab_dashboard": "لوحة المناقصات",
        "tab_audit": "سجل المراجعة",
        "pending": "قيد المراجعة",
        "approved": "مقبولة",
        "rejected": "مرفوضة",
        "all": "الكل",
        "flagged_pending": "مصر: مراجعة فورية",
        "flagged": "مصر · مراجعة فورية",
        "monitored": "السعودية · متابعة",
        "priority": "ضمن الأولويات",
        "alerts": "تم العثور على {n} مناقصة جديدة منذ آخر مراجعة "
                  "({f} في مصر تحتاج مراجعة فورية).",
        "mark_seen": "تمت المشاهدة",
        "review_help": "كل مناقصة يجدها وكيل المناقصات، من واجهات المناقصات أو من الويب، "
                       "تنتظر قرارك هنا. المناقصات المقبولة تظهر في لوحة المناقصات.",
        "approve": "قبول",
        "reject": "رفض",
        "to_pending": "إعادة للمراجعة",
        "open_source": "صفحة المصدر",
        "open_submission": "صفحة التقديم",
        "no_items": "لا يوجد شيء هنا. شغّل بحثًا جديدًا عن مناقصات لملء القائمة.",
        "no_dashboard": "لا توجد مناقصات مقبولة مطابقة.",
        "decided": "تم التعيين: {status}.",
        "search_intro": "يبحث وكيل المناقصات في واجهات المناقصات والبوابات الرسمية ومواقع "
                        "المناقصات والويب. كل مناقصة يجدها تنتظر مراجعتك.",
        "countries": "الدول",
        "categories": "مجالات التدريب",
        "categories_help": "اتركها فارغة لكل المجالات.",
        "priority_label": "جهات أو قطاعات ذات أولوية (اختياري)",
        "priority_help": "افصل بينها بفواصل، مثل: وزارة التعليم، البنوك.",
        "eligibility_label": "ما تستطيع الأكاديمية إثباته للأهلية (اختياري)",
        "eligibility_help": "يُستخدم لتقييم الأهلية، مثل: مقدم تدريب مسجل في مصر، ISO 9001.",
        "deadline_from": "الموعد النهائي من",
        "deadline_to": "الموعد النهائي إلى",
        "run_search": "ابحث عن مناقصات",
        "need_country": "اختر دولة واحدة على الأقل.",
        "searching": "جارٍ البحث في واجهات المناقصات والويب. قد يستغرق ذلك بضع دقائق...",
        "meta_country": "الدولة",
        "meta_org": "الجهة",
        "meta_category": "المجال",
        "meta_reference": "الرقم المرجعي",
        "meta_published": "تاريخ النشر",
        "meta_source": "المصدر",
        "meta_reviewer": "صاحب القرار",
        "deadline": "الموعد النهائي",
        "qualification": "الأهلية",
        "eligibility": "شروط الأهلية",
        "documents": "المستندات",
        "missing": "لم يذكرها المصدر",
        "summary": "ملخص الذكاء الاصطناعي",
        "q_qualified": "مؤهلة",
        "q_not_qualified": "غير مؤهلة",
        "q_unknown": "غير معروف",
        "send": "إرسال",
        "agents_placeholder": "مثال: ابحث عن مناقصات تعليمية في مصر",
        "agents_intro": "يوجّه المنسّق الطلبات إلى وكيل المناقصات أو وكيل المناهج. "
                        "أسئلة المتعلمين يجيب عنها ركس في واجهة المستخدم.",
        "agents_examples": ["ابحث عن مناقصات تعليم وتدريب في مصر والسعودية",
                            "أنشئ منهجًا عن تحليل البيانات"],
        "country": "الدولة",
        "category": "المجال",
        "keyword": "كلمة (العنوان، الجهة، الملخص)",
        "no_audit": "لا توجد قرارات مراجعة بعد.",
        "details": "التفاصيل",
        # Approval and applications
        "approve_title": "قبول المناقصة",
        "route_label": "طريقة التقديم",
        "automatic_yes": "يستطيع ركس تجهيز هذا الطلب (تجربة فقط: لا يُرسل أبدًا؛ "
                         "تراجعه وترسله بنفسك).",
        "automatic_no": "يدوي فقط: يسجّل ركس الخطوات الرسمية لك.",
        "authorise_label": "أفوّض ركس بالتقديم نيابةً عن الأكاديمية حيثما أمكن",
        "authorise_help": "بدون هذا التفويض لا يُقدَّم أي شيء نيابةً عنك. في هذا الإصدار "
                          "يجهّز ركس الطلب فقط (تجربة) ولا يرسل أي شيء.",
        "confirm_approve": "قبول",
        "cancel": "إلغاء",
        "approved_flash": "تم القبول. حالة الطلب: {status}.",
        "sub_status": {"not_started": "لم يبدأ", "in_progress": "قيد التنفيذ",
                       "submitted": "تم التقديم", "submission_failed": "فشل",
                       "requires_manual_action": "يحتاج إجراءً يدويًا",
                       "outcome_unknown": "النتيجة غير معروفة"},
        "app_status": {"not_applied": "لم يتم التقدم", "applied": "تم التقدم", "awarded": "تمت الترسية",
                       "not_awarded": "لم تتم الترسية", "withdrawn": "تم الانسحاب"},
        "filter_submission": "التقديم",
        "filter_application": "الطلب",
        "submission": "التقديم",
        "application": "الطلب",
        "authorised": "التقديم مفوَّض",
        "not_authorised": "التقديم غير مفوَّض",
        "authorise_now": "تفويض التقديم",
        "prepare_again": "إعادة التجهيز",
        "missing_items": "الناقص",
        "next_steps": "المطلوب",
        "prepared_email": "البريد المجهَّز (تجربة: لم يُرسَل)",
        "to": "إلى",
        "from": "من",
        "subject": "الموضوع",
        "attachments": "المرفقات",
        "recommended_missing": "مستحسن لكن لم يُرفع",
        "attempt_error": "خطأ",
        "record_submission": "تسجيل تقديم يدوي",
        "submitted_on": "تاريخ التقديم",
        "reference": "الرقم المرجعي أو رقم التأكيد",
        "submission_link": "الرابط (اختياري)",
        "notes": "ملاحظات (اختياري)",
        "confirm_submitted": "أؤكد أن الأكاديمية قدّمت هذا الطلب فعلًا",
        "need_confirm": "ضع علامة التأكيد أولًا.",
        "record": "تسجيل",
        "unknown_help": "نتيجة المحاولة الأخيرة غير معروفة. تحقّق هل أُرسل الطلب فعلًا قبل أي "
                        "إجراء آخر.",
        "was_sent": "تم الإرسال",
        "not_sent": "لم يُرسَل",
        "outcome": "النتيجة",
        "record_outcome": "تسجيل النتيجة",
        "correct_applied": "تصحيح إلى «تم التقدم»",
        "history": "السجل",
        "metric_applied": "تم التقدم",
        "metric_manual": "يحتاج إجراءً يدويًا",
        "done": "تم الحفظ.",
        # Curricula (Agent 3)
        "tab_curricula": "المناهج",
        "curricula_intro": "مناهج أعدّها وكيل المناهج. راجع النسخة ثم اعتمدها أو اطلب تعديلات. لا يستطيع "
                           "الوكيل الاعتماد أبدًا، وبيانات الكتالوج توضيحية ما لم يُذكر غير ذلك.",
        "cur_filter": "الحالة",
        "cur_all": "الكل",
        "cur_status": {"draft": "مسودة", "pending_review": "بانتظار المراجعة",
                       "changes_requested": "مطلوب تعديلات", "approved": "معتمد"},
        "cur_none": "لا توجد مناهج بعد. اطلب من الوكلاء مثلًا: «أنشئ منهج عن تحليل البيانات للمبتدئين لمدة 40 ساعة».",
        "cur_version": "النسخة",
        "cur_current": "الحالية",
        "cur_meta": {"level": "المستوى", "target_audience": "الجمهور", "total_hours": "إجمالي الساعات",
                     "language": "اللغة", "origin": "المصدر", "change_summary": "التعديل في هذه النسخة"},
        "cur_outcomes": "مخرجات التعلم",
        "cur_modules": "الوحدات",
        "cur_assessments": "التقييمات",
        "cur_capstone": "المشروع الختامي",
        "cur_no_capstone": "لا يوجد مشروع ختامي",
        "cur_validation_ok": "اجتاز المنهج جميع قواعد التحقق.",
        "cur_validation_bad": "مشكلات التحقق (هذه النسخة مسودة):",
        "cur_alignment": "المواءمة مع الإطار",
        "cur_no_alignment": "لا توجد مقارنة مع إطار (يتعذر التحديد دون إطار معتمد).",
        "cur_references": "المراجع",
        "cur_no_references": "لم تُستخدم مصادر بحث خارجية.",
        "cur_assumptions": "الافتراضات والملاحظات",
        "cur_pdf_download": "تنزيل PDF",
        "cur_pdf_create": "إنشاء PDF",
        "cur_review": "مراجعة هذه النسخة",
        "cur_notes": "ملاحظات (مطلوبة عند طلب التعديلات)",
        "cur_approve": "اعتماد",
        "cur_request_changes": "طلب تعديلات",
        "cur_not_reviewable": "يمكن اعتماد النسخة الحالية أو إعادتها فقط وهي بانتظار المراجعة.",
        "cur_audit": "سجل المراجعة",
        "cur_approved_msg": "تم الاعتماد.",
        "cur_changes_msg": "تم طلب التعديلات.",
        # Application profile
        "tab_profile": "ملف التقديم",
        "profile_intro": "بيانات الأكاديمية ومستنداتها الحقيقية، تُستخدم فقط لتجهيز الطلبات. "
                         "لا يُخترع شيء: الحقول الفارغة تبقى فارغة.",
        "save_profile": "حفظ البيانات",
        "documents_title": "المستندات",
        "doc_type": "نوع المستند",
        "upload_file": "الملف (PDF أو DOCX أو PNG أو JPG، حتى 10 ميجابايت)",
        "upload": "رفع",
        "remove": "حذف",
        "no_documents": "لم تُرفع مستندات بعد.",
        "field_label": {"legal_name": "الاسم القانوني", "registration_number": "رقم السجل التجاري",
                        "tax_number": "رقم التسجيل الضريبي", "address": "العنوان",
                        "country": "بلد التسجيل", "contact_person": "مسؤول التواصل",
                        "contact_title": "صفة مسؤول التواصل", "contact_email": "بريد التواصل",
                        "contact_phone": "هاتف التواصل", "website": "الموقع الإلكتروني"},
        "doc_label": {"company_profile": "ملف تعريف الشركة",
                      "registration_certificate": "شهادة السجل التجاري",
                      "tax_card": "البطاقة الضريبية", "experience_references": "خطابات الخبرة والمراجع",
                      "cvs": "السير الذاتية للفريق", "other": "أخرى"},
    },
}

AGENT_LABELS = {
    "rfp_research": {"en": "RFP Research", "ar": "وكيل المناقصات"},
    "curriculum_research": {"en": "Curriculum Research", "ar": "وكيل المناهج"},
    "orchestrator": {"en": "Orchestrator", "ar": "المنسّق"},
}

COUNTRY_LABELS = {"Egypt": {"en": "Egypt", "ar": "مصر"},
                  "Saudi Arabia": {"en": "Saudi Arabia", "ar": "السعودية"}}

STATUSES = ("pending", "approved", "rejected")


def lang() -> str:
    # Clicking the selected segment again deselects it (None).
    return "ar" if st.session_state.get("lang") == "ar" else "en"


def t(key: str):
    return T[lang()][key]


def country_label(country) -> str:
    return COUNTRY_LABELS.get(country, {}).get(lang(), str(country or ""))


# --------------------------------------------------------------------
# Services (one per server process)
# --------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_user_service():
    return user_module.RexUserService()


@st.cache_resource(show_spinner=False)
def get_admin_service():
    # Nothing admin-related (agents, stores) is built until an admin
    # action runs after a successful login.
    return admin_module.AdminService(
        orchestrator_factory=lambda: register_agents.create_admin_orchestrator(),
    )


# --------------------------------------------------------------------
# Session, styling, helpers
# --------------------------------------------------------------------

def init_session() -> None:
    defaults = {
        "lang": "en",
        "view": "welcome",
        "chat": [],
        "rex_state": None,
        "admin_token": None,
        "admin_chat": [],
        "admin_conversation_id": uuid.uuid4().hex,
        "notice": None,
        "passcode_open": False,
        "last_search": None,
        "approve_for": None,   # (opportunity_id, title) while the approve dialog is open
        "flash": None,         # a message shown once after an action
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


CSS = """
<style>
.rex-hero {padding: 2rem 1.6rem; border-radius: 22px; margin-bottom: 1.4rem;
  background: linear-gradient(135deg, #4f46e5 0%, #0ea5a4 100%); color: #fff;}
.rex-hero h1 {color: #fff; font-size: 2.6rem; margin: 0; letter-spacing: -0.02em;}
.rex-hero p {color: #fff; opacity: .92; margin: .35rem 0 0; font-size: 1.08rem;}
.rex-title {display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .6rem;}
.rex-title h2 {margin: 0;}
.rex-badge {display: inline-block; padding: .08rem .6rem; border-radius: 999px;
  font-size: .78rem; font-weight: 600; margin-inline-end: .35rem;}
.rex-pending {background: #fef3c7; color: #92400e;}
.rex-approved {background: #d1fae5; color: #065f46;}
.rex-rejected {background: #fee2e2; color: #991b1b;}
.rex-agent {background: #e0e7ff; color: #3730a3;}
.rex-flagged {background: #fee2e2; color: #b91c1c;}
.rex-monitored {background: #e0f2fe; color: #075985;}
.rex-priority {background: #ede9fe; color: #5b21b6;}
.rex-sub-submitted, .rex-app-applied, .rex-app-awarded {background: #d1fae5; color: #065f46;}
.rex-sub-requires_manual_action, .rex-sub-in_progress {background: #fef3c7; color: #92400e;}
.rex-sub-submission_failed, .rex-sub-outcome_unknown, .rex-app-not_awarded {background: #fee2e2; color: #991b1b;}
.rex-sub-not_started, .rex-app-not_applied, .rex-app-withdrawn {background: #e5e7eb; color: #374151;}
.rex-cur-approved {background: #d1fae5; color: #065f46;}
.rex-cur-pending_review {background: #fef3c7; color: #92400e;}
.rex-cur-changes_requested {background: #fee2e2; color: #991b1b;}
.rex-cur-draft {background: #e5e7eb; color: #374151;}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
.rex-auto p {unicode-bidi: plaintext; text-align: start;}
</style>
"""

RTL_CSS = """
<style>
[data-testid="stMainBlockContainer"], .main .block-container,
[data-testid="stDialog"] [role="dialog"] {direction: rtl;}
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {text-align: right;}
input, textarea {direction: rtl; text-align: right;}
</style>
"""


def apply_style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if lang() == "ar":
        st.markdown(RTL_CSS, unsafe_allow_html=True)


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def plain(text) -> str:
    """Escape untrusted text so st.markdown shows it literally."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", str(text or ""))


def badge(css_class: str, label: str) -> str:
    return f'<span class="rex-badge rex-{css_class}">{label}</span>'


def language_switcher() -> None:
    st.segmented_control(
        "Language",
        options=["en", "ar"],
        format_func=lambda code: "English" if code == "en" else "العربية",
        key="lang",
        label_visibility="collapsed",
    )


def top_bar(title: str | None = None, home: bool = True) -> None:
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        if home and st.button(t("back"), icon=":material/home:", key="home"):
            st.session_state.view = "welcome"
            st.rerun()
    with right:
        language_switcher()
    if title:
        st.markdown(f'<div class="rex-title"><h2>{title}</h2></div>', unsafe_allow_html=True)


def go_home(notice: str | None = None) -> None:
    st.session_state.view = "welcome"
    st.session_state.notice = notice
    st.rerun()


# --------------------------------------------------------------------
# Welcome
# --------------------------------------------------------------------

def _close_passcode() -> None:
    st.session_state.passcode_open = False


def passcode_dialog() -> None:
    # Kept open by a session flag (not only by the click that opened it),
    # so the form submit that follows still finds the dialog.
    @st.dialog(t("passcode_title"), on_dismiss=_close_passcode)
    def dialog():
        admin = get_admin_service()
        if not admin.enabled:
            st.warning(t("admin_disabled"), icon=":material/lock:")
            return
        with st.form("passcode_form", clear_on_submit=True, border=False):
            code = st.text_input(t("passcode"), type="password")
            submitted = st.form_submit_button(t("unlock"), type="primary",
                                              width="stretch")
        if submitted:
            token = admin.login(code)
            if token:
                st.session_state.admin_token = token
                st.session_state.passcode_open = False
                st.session_state.view = "admin"
                st.rerun()
            elif admin.is_locked():
                st.error(t("locked"))
            else:
                st.error(t("wrong_passcode"))

    dialog()


def welcome_view() -> None:
    top_bar(home=False)
    if st.session_state.notice:
        st.info(st.session_state.notice)
        st.session_state.notice = None

    st.markdown(
        f'<div class="rex-hero"><h1>REX AI</h1><p>{t("tagline")}</p></div>',
        unsafe_allow_html=True,
    )

    learner, admin = st.columns(2, gap="medium")
    with learner, st.container(border=True, height="stretch"):
        st.markdown(f"### :material/school: {t('user_title')}")
        st.write(t("user_desc"))
        if st.button(t("user_button"), type="primary", width="stretch", key="go_user"):
            st.session_state.view = "user"
            st.rerun()
    with admin, st.container(border=True, height="stretch"):
        st.markdown(f"### :material/admin_panel_settings: {t('admin_title')}")
        st.write(t("admin_desc"))
        if st.button(t("admin_button"), width="stretch", key="go_admin"):
            if st.session_state.admin_token and get_admin_service().is_valid(
                st.session_state.admin_token
            ):
                st.session_state.view = "admin"
                st.rerun()
            st.session_state.passcode_open = True
    if st.session_state.passcode_open:
        passcode_dialog()


# --------------------------------------------------------------------
# User: REX Assistant
# --------------------------------------------------------------------

def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(t("sources"), icon=":material/menu_book:"):
        for source in sources:
            page = f", {t('page')} {source['page']}" if source.get("page") else ""
            st.markdown(f"**[{source.get('reference')}]** {plain(source.get('source'))}{page}")


def send_to_rex(message: str) -> None:
    service = get_user_service()
    if st.session_state.rex_state is None:
        st.session_state.rex_state = service.new_conversation(lang())

    st.session_state.chat.append({"role": "user", "content": message})
    with st.spinner(t("thinking")):
        try:
            response = service.chat(message, state=st.session_state.rex_state)
        except Exception:
            response = {"status": "error"}

    if response.get("status") == "success" and response.get("result"):
        st.session_state.rex_state = response["state"]
        st.session_state.chat.append({
            "role": "assistant",
            "content": response["result"].get("answer") or t("rex_error"),
            "sources": response["result"].get("sources", []),
        })
    else:
        st.session_state.chat.append({"role": "assistant", "content": t("rex_error")})


def user_view() -> None:
    top_bar()
    header, action = st.columns([3, 2], vertical_alignment="center")
    with header:
        st.markdown('<div class="rex-title"><h2>REX</h2></div>', unsafe_allow_html=True)
    with action:
        if st.button(t("new_chat"), icon=":material/add_comment:", key="new_chat"):
            st.session_state.chat = []
            st.session_state.rex_state = None
            st.rerun()

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.markdown(t("rex_intro"))

    prompt = None
    if not st.session_state.chat:
        cols = st.columns(len(t("examples")))
        for i, (col, example) in enumerate(zip(cols, t("examples"))):
            if col.button(example, key=f"example_{i}", width="stretch"):
                prompt = example

    typed = st.chat_input(t("chat_placeholder"))
    prompt = typed or prompt
    if prompt and prompt.strip():
        send_to_rex(prompt.strip())

    for message in st.session_state.chat:
        avatar = ":material/smart_toy:" if message["role"] == "assistant" else ":material/person:"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))


# --------------------------------------------------------------------
# Admin dashboard
# --------------------------------------------------------------------

def admin_call(method: str, *args, **kwargs):
    """Every admin action goes through AdminService with the session token."""
    try:
        return getattr(get_admin_service(), method)(st.session_state.admin_token, *args, **kwargs)
    except admin_module.AdminAuthError:
        st.session_state.admin_token = None
        go_home(t("session_expired"))


def admin_action(method: str, *args, **kwargs) -> bool:
    """
    Run an admin action that changes data. A refusal by the rules (e.g. a
    transition that is not allowed, a bad file) is shown as an error
    instead of crashing the page. Returns True when it worked.
    """
    try:
        admin_call(method, *args, **kwargs)
        return True
    except (ValueError, KeyError) as exc:  # InvalidTransition is a ValueError
        st.error(str(exc).strip("'\""))
        return False


def _web_link(url) -> str | None:
    url = str(url or "")
    return url if url.startswith(("http://", "https://")) else None


def tender_badges(tender: dict, with_status: bool = True) -> str:
    parts = [badge(tender.get("approval_status") or "pending",
                   t(tender.get("approval_status") or "pending"))] if with_status else []
    if tender.get("review_status") == "flagged_for_immediate_review":
        parts.append(badge("flagged", t("flagged")))
    elif tender.get("review_status") == "monitored_pipeline":
        parts.append(badge("monitored", t("monitored")))
    if tender.get("priority_match"):
        parts.append(badge("priority", t("priority")))
    return "".join(parts)


def tender_details(tender: dict) -> None:
    """All extracted fields; web text is escaped and shown literally."""
    qualification = tender.get("qualification_status")
    meta = [
        ("meta_org", tender.get("organization")),
        ("meta_country", country_label(tender.get("country"))),
        ("meta_category", tender.get("training_category") or tender.get("category")),
        ("deadline", tender.get("deadline")),
        ("meta_published", tender.get("publication_date")),
        ("meta_reference", tender.get("reference_number")),
        ("qualification", qualification and T[lang()].get(f"q_{qualification}", qualification)),
        ("meta_source", tender.get("source_name")),
        ("meta_reviewer", tender.get("decided_by")),
    ]
    st.markdown(" · ".join(f"**{t(key)}:** {plain(value)}" for key, value in meta if value))
    if tender.get("ai_summary"):
        st.markdown(f"**{t('summary')}:** {plain(tender['ai_summary'])}")
    if tender.get("qualification_reason"):
        st.caption(plain(tender["qualification_reason"]))
    if tender.get("eligibility"):
        st.markdown(f"**{t('eligibility')}:**\n" + "\n".join(
            f"- {plain(item)}" for item in tender["eligibility"]))
    if tender.get("available_documents"):
        st.markdown(f"**{t('documents')}:**\n" + "\n".join(
            f"- {plain(doc)}" for doc in tender["available_documents"]))
    if tender.get("missing_information"):
        st.caption(f"{t('missing')}: " + ", ".join(plain(f) for f in tender["missing_information"]))


def tender_links(tender: dict, columns) -> int:
    """Link buttons for the source and submission pages; returns columns used."""
    used = 0
    source = _web_link(tender.get("source_url") or tender.get("url"))
    submission = _web_link(tender.get("submission_url"))
    if source:
        columns[used].link_button(t("open_source"), source, icon=":material/open_in_new:")
        used += 1
    if submission and submission != source:
        columns[used].link_button(t("open_submission"), submission, icon=":material/send:")
        used += 1
    return used


def review_item(tender: dict) -> None:
    opp_id = tender["opportunity_id"]
    status = tender.get("approval_status") or "pending"
    with st.container(border=True):
        st.markdown(tender_badges(tender), unsafe_allow_html=True)
        st.markdown(f"**{plain(tender.get('title'))}**")
        tender_details(tender)

        buttons = st.columns(5)
        column = tender_links(tender, buttons)
        actions = [("approved", "approve", "primary"), ("rejected", "reject", "secondary"),
                   ("pending", "to_pending", "tertiary")]
        for decision, label, kind in actions:
            if decision == status:
                continue
            if buttons[column].button(t(label), key=f"{decision}_{opp_id}", type=kind):
                if decision == "approved":
                    # Approving starts the application workflow, so it
                    # goes through a dialog that shows the route first.
                    st.session_state.approve_for = (opp_id, tender.get("title") or "")
                    st.rerun()
                if admin_action("decide", opp_id, decision):
                    st.session_state.flash = t("decided").format(status=t(decision))
                    st.rerun()
            column += 1


def _close_approve() -> None:
    st.session_state.approve_for = None


def approve_dialog() -> None:
    opp_id, title = st.session_state.approve_for

    @st.dialog(t("approve_title"), on_dismiss=_close_approve)
    def dialog():
        route = admin_call("submission_route", opp_id)
        st.markdown(f"**{plain(title)}**")
        st.markdown(f"**{t('route_label')}:** {plain(route['label'])}")
        st.caption(t("automatic_yes") if route["automatic"] else t("automatic_no"))
        st.markdown("\n".join(f"- {plain(step)}" for step in route["steps"]))
        authorise = st.checkbox(t("authorise_label"), help=t("authorise_help"), key="authorise_submission")
        confirm, cancel = st.columns(2)
        if confirm.button(t("confirm_approve"), type="primary", key="confirm_approve", width="stretch"):
            result = admin_call("decide", opp_id, "approved", authorize_submission=authorise)
            st.session_state.approve_for = None
            st.session_state.flash = t("approved_flash").format(
                status=t("sub_status")[result["submission_status"]])
            st.rerun()
        if cancel.button(t("cancel"), key="cancel_approve", width="stretch"):
            st.session_state.approve_for = None
            st.rerun()

    dialog()


def alerts_banner() -> None:
    alerts = admin_call("alerts")
    if not alerts:
        return
    flagged = sum(1 for a in alerts if a.get("review_status") == "flagged_for_immediate_review")
    with st.container(border=True):
        text, action = st.columns([5, 1], vertical_alignment="center")
        text.warning(t("alerts").format(n=len(alerts), f=flagged), icon=":material/notifications:")
        if action.button(t("mark_seen"), key="mark_seen"):
            admin_call("mark_alerts_seen")
            st.rerun()


def tender_filters(prefix: str):
    country_col, category_col, keyword_col = st.columns([1, 1, 2])
    country = country_col.selectbox(
        t("country"), [None, *ALLOWED_COUNTRIES], key=f"{prefix}_country",
        format_func=lambda c: t("all") if c is None else country_label(c))
    category = category_col.selectbox(
        t("category"), [None, *TRAINING_CATEGORIES], key=f"{prefix}_category",
        format_func=lambda c: t("all") if c is None else c)
    keyword = keyword_col.text_input(t("keyword"), key=f"{prefix}_keyword")
    return {"country": country, "category": category, "keyword": keyword.strip() or None}


def review_tab(counts: dict) -> None:
    st.caption(t("review_help"))
    choice = st.segmented_control(
        "Status", options=["pending", "approved", "rejected", "all"], default="pending",
        format_func=lambda s: f"{t(s)} ({counts[s]})" if s in counts else t(s),
        key="review_filter", label_visibility="collapsed",
    ) or "pending"
    filters = tender_filters("review")

    tenders = admin_call("list_tenders", None if choice == "all" else choice, **filters)
    if not tenders:
        st.info(t("no_items"))
    for tender in tenders:
        review_item(tender)
    if st.session_state.approve_for:
        approve_dialog()


def search_tab() -> None:
    st.caption(t("search_intro"))
    with st.form("tender_search_form", border=True):
        countries = st.multiselect(t("countries"), list(ALLOWED_COUNTRIES),
                                   default=list(ALLOWED_COUNTRIES), format_func=country_label)
        categories = st.multiselect(t("categories"), list(TRAINING_CATEGORIES),
                                    help=t("categories_help"))
        priority = st.text_input(t("priority_label"), help=t("priority_help"))
        eligibility = st.text_area(t("eligibility_label"), help=t("eligibility_help"), height=80)
        start, end = st.columns(2)
        deadline_from = start.date_input(t("deadline_from"), value=None)
        deadline_to = end.date_input(t("deadline_to"), value=None)
        submitted = st.form_submit_button(t("run_search"), icon=":material/travel_explore:",
                                          type="primary")

    if submitted:
        if not countries:
            st.error(t("need_country"))
        else:
            options = {
                "countries": countries,
                "categories": categories,
                "priority": priority.strip(),
                "eligibility_notes": eligibility.strip(),
                "deadline_from": deadline_from.isoformat() if deadline_from else None,
                "deadline_to": deadline_to.isoformat() if deadline_to else None,
            }
            with st.spinner(t("searching")):
                st.session_state.last_search = admin_call("search_tenders", options, context={
                    "conversation_id": st.session_state.admin_conversation_id,
                    # The request text is fixed English; answer in the UI language.
                    "preferred_language": lang(),
                })

    if st.session_state.last_search:
        for result in st.session_state.last_search.get("result") or []:
            render_agent_result(result)


def render_agent_result(result: dict) -> None:
    agent = result.get("agent_name", "")
    label = AGENT_LABELS.get(agent, {}).get(lang(), agent)
    st.markdown(f'<span class="rex-badge rex-agent">{label}</span>', unsafe_allow_html=True)

    if result.get("status") != "success":
        st.error(result.get("error_message") or t("rex_error"))
        return

    body = result.get("result") or {}
    if not isinstance(body, dict):
        st.write(body)
        return
    if body.get("message"):
        # One line per point; Markdown needs "  \n" for a line break.
        st.markdown(plain(body["message"]).replace("\n", "  \n"))
    for opportunity in body.get("opportunities") or []:
        with st.expander(plain(opportunity.get("title") or opportunity.get("url"))):
            st.markdown(tender_badges(opportunity, with_status=False), unsafe_allow_html=True)
            tender_details(opportunity)
            tender_links(opportunity, st.columns(2))
    if body.get("curriculum_id"):
        st.caption(f"{t('tab_curricula')}: `{body['curriculum_id']}`")
    # The full curriculum is read in the Curricula tab, not as raw JSON here.
    extra = {k: v for k, v in body.items() if k not in ("message", "opportunities", "curriculum")}
    if extra:
        with st.expander(t("details")):
            st.json(extra, expanded=False)


def agents_tab() -> None:
    st.caption(t("agents_intro"))
    prompt = None
    if not st.session_state.admin_chat:
        cols = st.columns(len(t("agents_examples")))
        for i, (col, example) in enumerate(zip(cols, t("agents_examples"))):
            if col.button(example, key=f"agent_example_{i}", width="stretch"):
                prompt = example

    with st.form("agent_form", clear_on_submit=True, border=False):
        typed = st.text_input(t("tab_agents"), placeholder=t("agents_placeholder"),
                              label_visibility="collapsed")
        if st.form_submit_button(t("send"), icon=":material/send:", type="primary"):
            prompt = typed or prompt

    if prompt and prompt.strip():
        with st.spinner(t("thinking")):
            response = admin_call("ask", prompt.strip(), context={
                "conversation_id": st.session_state.admin_conversation_id,
            })
        st.session_state.admin_chat.append({"role": "user", "content": prompt.strip()})
        st.session_state.admin_chat.append({"role": "assistant", "response": response})

    for message in reversed(_pairs(st.session_state.admin_chat)):
        user, reply = message
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(user["content"])
        with st.chat_message("assistant", avatar=":material/hub:"):
            response = reply["response"] or {}
            results = response.get("result") or []
            if not results and response.get("error_message"):
                st.error(response["error_message"])
            for result in results:
                render_agent_result(result)


def _pairs(messages: list[dict]) -> list[tuple[dict, dict]]:
    return [(messages[i], messages[i + 1]) for i in range(0, len(messages) - 1, 2)]


def status_badges(tender: dict) -> str:
    submission, application = tender["submission_status"], tender["application_status"]
    return (badge(f"sub-{submission}", f"{t('submission')}: {t('sub_status')[submission]}")
            + badge(f"app-{application}", f"{t('application')}: {t('app_status')[application]}"))


def dashboard_tab() -> None:
    """Approved tenders, searchable (brief requirement 16), with their applications."""
    filters = tender_filters("dashboard")
    sub_col, app_col = st.columns(2)
    submission = sub_col.selectbox(
        t("filter_submission"), [None, *t("sub_status")], key="dashboard_submission",
        format_func=lambda s: t("all") if s is None else t("sub_status")[s])
    application = app_col.selectbox(
        t("filter_application"), [None, *t("app_status")], key="dashboard_application",
        format_func=lambda s: t("all") if s is None else t("app_status")[s])
    rows = admin_call("list_tenders", "approved", submission_status=submission,
                      application_status=application, **filters)
    if not rows:
        st.info(t("no_dashboard"))
        return
    columns = ["title", "organization", "country", "training_category", "deadline",
               "submission_status", "application_status", "submitted_at", "submission_reference",
               "qualification_status", "review_status", "submission_url", "source_url"]
    st.dataframe(
        [{c: (row.get(c) if c != "source_url" else _web_link(row.get("source_url") or row.get("url")))
          for c in columns} for row in rows],
        hide_index=True, width="stretch",
        column_config={"submission_url": st.column_config.LinkColumn(t("open_submission")),
                       "source_url": st.column_config.LinkColumn(t("open_source"))},
    )
    for row in rows:
        with st.expander(plain(row.get("title"))):
            st.markdown(tender_badges(row, with_status=False) + status_badges(row), unsafe_allow_html=True)
            tender_details(row)
            application_panel(row)


def application_panel(tender: dict) -> None:
    """The latest attempt, the prepared email, and the actions the state allows."""
    opp_id = tender["opportunity_id"]
    submission = tender["submission_status"]
    history = admin_call("application_history", opp_id)
    attempts = history["attempts"]
    latest = attempts[-1] if attempts else None

    st.markdown(f"**{t('application')}**")
    st.caption(t("authorised") if tender["submission_authorized"] else t("not_authorised"))
    if latest:
        if latest["missing"]:
            st.warning(f"**{t('missing_items')}:** " + "; ".join(plain(m) for m in latest["missing"]))
        if latest["manual_actions"]:
            st.markdown(f"**{t('next_steps')}:**\n" + "\n".join(
                f"{i}. {plain(step)}" for i, step in enumerate(latest["manual_actions"], 1)))
        if latest.get("error"):
            st.error(f"{t('attempt_error')}: {plain(latest['error'])}")
        email = (latest["confirmation"] or {}).get("prepared_email")
        if email:
            with st.container(border=True):
                st.markdown(f"**{t('prepared_email')}**")
                st.markdown(f"**{t('to')}:** {plain(email['to'])} · **{t('from')}:** {plain(email['from'])}")
                st.markdown(f"**{t('subject')}:** {plain(email['subject'])}")
                st.text(email["body"])
                st.markdown(f"**{t('attachments')}:** " + "; ".join(
                    plain(a["file_name"]) for a in email["attachments"]))
                recommended = latest["confirmation"].get("recommended_documents_missing")
                if recommended:
                    st.caption(f"{t('recommended_missing')}: " + ", ".join(plain(r) for r in recommended))

    actions = st.columns(3)
    if not tender["submission_authorized"] and submission != "submitted":
        if actions[0].button(t("authorise_now"), key=f"authorise_{opp_id}"):
            if admin_action("authorize_submission", opp_id):
                st.session_state.flash = t("done")
                st.rerun()
    if submission in ("not_started", "requires_manual_action", "submission_failed"):
        if actions[1].button(t("prepare_again"), key=f"retry_{opp_id}", icon=":material/refresh:"):
            if admin_action("retry_submission", opp_id):
                st.session_state.flash = t("done")
                st.rerun()

    if submission == "outcome_unknown":
        st.warning(t("unknown_help"))
        if st.button(t("not_sent"), key=f"not_sent_{opp_id}"):
            if admin_action("mark_not_submitted", opp_id):
                st.session_state.flash = t("done")
                st.rerun()

    if submission not in ("submitted", "in_progress"):
        manual_submission_form(opp_id, sent_label=submission == "outcome_unknown")

    if tender["application_status"] != "not_applied":
        outcome_form(opp_id, tender["application_status"])

    with st.expander(t("history")):
        if attempts:
            st.dataframe([{k: a.get(k) for k in ("attempt_no", "adapter", "dry_run", "status", "triggered_by",
                                                  "actor", "started_at", "finished_at", "error")}
                          for a in attempts], hide_index=True, width="stretch")
        st.dataframe([{k: e.get(k) for k in ("event", "previous_status", "decision", "reviewer", "decided_at")}
                      for e in history["audit"]], hide_index=True, width="stretch")


def manual_submission_form(opp_id: str, sent_label: bool = False) -> None:
    """The admin confirms the academy actually submitted (or, after an unknown result, that it was sent)."""
    with st.form(f"manual_submission_{opp_id}", border=True):
        st.markdown(f"**{t('was_sent') if sent_label else t('record_submission')}**")
        left, right = st.columns(2)
        submitted_on = left.date_input(t("submitted_on"), key=f"submitted_on_{opp_id}")
        reference = right.text_input(t("reference"), key=f"reference_{opp_id}")
        link = st.text_input(t("submission_link"), key=f"link_{opp_id}")
        notes = st.text_input(t("notes"), key=f"notes_{opp_id}")
        confirmed = st.checkbox(t("confirm_submitted"), key=f"confirm_{opp_id}")
        if st.form_submit_button(t("record"), type="primary", key=f"record_{opp_id}"):
            if not confirmed:
                st.error(t("need_confirm"))
            elif admin_action("record_manual_submission", opp_id, submitted_on.isoformat(),
                              reference=reference.strip() or None, url=link.strip() or None,
                              notes=notes.strip() or None):
                st.session_state.flash = t("done")
                st.rerun()


def outcome_form(opp_id: str, current: str) -> None:
    options = ["awarded", "not_awarded", "withdrawn"] if current == "applied" else ["applied"]
    with st.form(f"outcome_{opp_id}", border=True):
        st.markdown(f"**{t('outcome')}**")
        outcome = st.selectbox(t("outcome"), options, key=f"outcome_choice_{opp_id}",
                               format_func=lambda s: t("app_status")[s], label_visibility="collapsed")
        notes = st.text_input(t("notes"), key=f"outcome_notes_{opp_id}")
        label = t("record_outcome") if current == "applied" else t("correct_applied")
        if st.form_submit_button(label, key=f"record_outcome_{opp_id}"):
            if admin_action("record_application_outcome", opp_id, outcome, notes=notes.strip() or None):
                st.session_state.flash = t("done")
                st.rerun()


def profile_tab() -> None:
    """The academy's real details and documents (never invented)."""
    st.caption(t("profile_intro"))
    profile = admin_call("application_profile")
    labels = t("field_label")

    with st.form("profile_form", border=True):
        values = {}
        columns = st.columns(2)
        for i, field in enumerate(profile["field_labels"]):
            values[field] = columns[i % 2].text_input(
                labels.get(field, field), value=profile["fields"].get(field, ""), key=f"profile_{field}")
        if st.form_submit_button(t("save_profile"), type="primary", key="save_profile"):
            if admin_action("update_application_profile", values):
                st.session_state.flash = t("done")
                st.rerun()

    st.markdown(f"**{t('documents_title')}**")
    doc_labels = t("doc_label")
    if not profile["documents"]:
        st.info(t("no_documents"))
    for doc in profile["documents"]:
        name, action = st.columns([5, 1], vertical_alignment="center")
        name.markdown(f"{plain(doc_labels.get(doc['doc_type'], doc['doc_type']))}: "
                      f"**{plain(doc['original_name'])}** ({doc['size_bytes'] // 1024 or 1} KB)")
        if action.button(t("remove"), key=f"remove_doc_{doc['document_id']}"):
            if admin_action("remove_application_document", doc["document_id"]):
                st.rerun()

    with st.form("upload_form", clear_on_submit=True, border=True):
        doc_type = st.selectbox(t("doc_type"), list(profile["document_types"]),
                                format_func=lambda d: doc_labels.get(d, d))
        upload = st.file_uploader(t("upload_file"), type=["pdf", "docx", "png", "jpg", "jpeg"])
        if st.form_submit_button(t("upload"), icon=":material/upload:") and upload is not None:
            if admin_action("upload_application_document", doc_type, upload.name, upload.getvalue()):
                st.session_state.flash = t("done")
                st.rerun()


def curricula_tab() -> None:
    """Agent 3's curricula: read a version, download its PDF, approve or request changes."""
    st.caption(t("curricula_intro"))
    labels = t("cur_status")
    counts = admin_call("curriculum_counts")
    metrics = st.columns(len(labels))
    for column, (status, label) in zip(metrics, labels.items()):
        column.metric(label, counts.get(status, 0))

    options = [None, *labels]
    status = st.selectbox(t("cur_filter"), options, key="cur_filter",
                          format_func=lambda s: t("cur_all") if s is None else labels[s])
    rows = admin_call("list_curricula", status)
    if not rows:
        st.info(t("cur_none"))
        return
    for row in rows:
        title = (f"{plain(row['title'])} · v{row['current_version']} · {row.get('total_hours') or '?'} h · "
                 f"{(row.get('language') or '').upper()} · {labels.get(row['status'], row['status'])}")
        with st.expander(title):
            curriculum_panel(row)


def curriculum_panel(row: dict) -> None:
    curriculum_id = row["curriculum_id"]
    labels = t("cur_status")
    versions = [v["version"] for v in admin_call("get_curriculum", curriculum_id)["versions"]]
    version = st.selectbox(
        t("cur_version"), list(reversed(versions)), key=f"cur_version_{curriculum_id}",
        format_func=lambda v: f"v{v}" + (f" ({t('cur_current')})" if v == row["current_version"] else ""))
    detail = admin_call("get_curriculum", curriculum_id, version)
    record = detail["curriculum"]
    content = record["content"]
    version_status = record.get("version_status") or record["status"]

    st.markdown(badge(f"cur-{version_status}", labels.get(version_status, version_status))
                + f" `{curriculum_id}`", unsafe_allow_html=True)
    if content.get("overview"):
        st.markdown(plain(content["overview"]))
    meta_labels = t("cur_meta")
    meta = {"level": content.get("level"), "target_audience": content.get("target_audience"),
            "total_hours": content.get("total_hours"), "language": content.get("language"),
            "origin": record.get("origin"), "change_summary": record.get("change_summary")}
    st.markdown("  \n".join(f"**{meta_labels[k]}:** {plain(v)}" for k, v in meta.items() if v))
    if content.get("data_notice"):
        st.info(plain(content["data_notice"]))

    st.markdown(f"**{t('cur_outcomes')}**")
    st.markdown("\n".join(f"- {plain(o.get('id'))}: {plain(o.get('statement'))}"
                          for o in content.get("learning_outcomes") or []))
    st.markdown(f"**{t('cur_modules')}**")
    st.dataframe([{"ID": m.get("id"), "Module": m.get("title"), "h": m.get("hours"),
                   "Topics": ", ".join(m.get("topics") or []), "LO": ", ".join(m.get("outcome_ids") or [])}
                  for m in content.get("modules") or []], hide_index=True, width="stretch")
    st.markdown(f"**{t('cur_assessments')}**")
    st.dataframe([{"ID": a.get("id"), "Type": a.get("type"), "Description": a.get("description"),
                   "LO": ", ".join(a.get("outcome_ids") or [])} for a in content.get("assessments") or []],
                 hide_index=True, width="stretch")
    capstone = content.get("capstone")
    st.markdown(f"**{t('cur_capstone')}:** " + (f"{plain(capstone.get('title'))} — {plain(capstone.get('description'))}"
                                               if isinstance(capstone, dict) else t("cur_no_capstone")))

    errors = (record.get("validation") or {}).get("errors") or []
    if errors:
        st.warning(t("cur_validation_bad") + "\n" + "\n".join(f"- {plain(e.get('message'))}" for e in errors))
    else:
        st.success(t("cur_validation_ok"))

    st.markdown(f"**{t('cur_alignment')}**")
    alignment = content.get("alignment") or {}
    if alignment.get("items"):
        framework = alignment.get("framework") or {}
        st.caption(plain(f"{framework.get('name')} — {framework.get('publisher')}, {framework.get('version')}"))
        st.dataframe([{"Requirement": f"{i['requirement_id']}: {i.get('requirement') or ''}", "Result": i["status"],
                       "Evidence": ", ".join(i.get("element_ids") or []),
                       "Explanation": i.get("recommendation") or i.get("explanation")}
                      for i in alignment["items"]], hide_index=True, width="stretch")
    else:
        st.caption(t("cur_no_alignment"))

    references = content.get("references") or []
    st.markdown(f"**{t('cur_references')}**")
    st.markdown("\n".join(f"- {plain(r.get('title'))} — {plain(r.get('url'))}" for r in references)
                or t("cur_no_references"))
    if content.get("assumptions"):
        st.markdown(f"**{t('cur_assumptions')}**")
        st.markdown("\n".join(f"- {plain(a)}" for a in content["assumptions"]))

    has_pdf = any(d["version"] == version for d in detail["documents"])
    if has_pdf:
        pdf = admin_call("curriculum_pdf", curriculum_id, version)
        st.download_button(t("cur_pdf_download"), pdf["data"], file_name=pdf["filename"], mime="application/pdf",
                           icon=":material/download:", key=f"cur_pdf_{curriculum_id}_{version}")
    elif st.button(t("cur_pdf_create"), icon=":material/picture_as_pdf:", key=f"cur_pdf_make_{curriculum_id}_{version}"):
        if admin_action("curriculum_pdf", curriculum_id, version):
            st.rerun()

    reviewable = version == row["current_version"] and record["status"] == "pending_review"
    st.markdown(f"**{t('cur_review')}**")
    if not reviewable:
        st.caption(t("cur_not_reviewable"))
    else:
        with st.form(f"cur_review_{curriculum_id}", border=True):
            notes = st.text_area(t("cur_notes"), key=f"cur_notes_{curriculum_id}")
            approve, changes = st.columns(2)
            if approve.form_submit_button(t("cur_approve"), type="primary", icon=":material/check:",
                                          key=f"cur_approve_{curriculum_id}"):
                if admin_action("approve_curriculum", curriculum_id, version, notes or None):
                    st.session_state.flash = t("cur_approved_msg")
                    st.rerun()
            if changes.form_submit_button(t("cur_request_changes"), icon=":material/edit_note:",
                                          key=f"cur_changes_{curriculum_id}"):
                if admin_action("request_curriculum_changes", curriculum_id, version, notes):
                    st.session_state.flash = t("cur_changes_msg")
                    st.rerun()

    if detail["audit"]:
        with st.expander(t("cur_audit")):
            st.dataframe(list(reversed(detail["audit"])), hide_index=True, width="stretch")


def audit_tab() -> None:
    trail = admin_call("audit_trail")
    if not trail:
        st.info(t("no_audit"))
        return
    st.dataframe(list(reversed(trail)), hide_index=True, width="stretch")


def admin_view() -> None:
    admin = get_admin_service()
    if not admin.is_valid(st.session_state.admin_token):
        st.session_state.admin_token = None
        go_home(t("session_expired"))

    top_bar()
    header, action = st.columns([3, 2], vertical_alignment="center")
    with header:
        st.markdown(f'<div class="rex-title"><h2>{t("dashboard")}</h2></div>',
                    unsafe_allow_html=True)
    with action:
        if st.button(t("sign_out"), icon=":material/logout:", key="sign_out"):
            admin.logout(st.session_state.admin_token)
            st.session_state.admin_token = None
            go_home()

    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None
    alerts_banner()
    counts = admin_call("tender_counts")
    metrics = st.columns(6)
    metrics[0].metric(t("pending"), counts["pending"])
    metrics[1].metric(t("flagged_pending"), counts["flagged_pending"])
    metrics[2].metric(t("approved"), counts["approved"])
    metrics[3].metric(t("metric_manual"), counts["needs_manual_action"])
    metrics[4].metric(t("metric_applied"), counts["applied"])
    metrics[5].metric(t("rejected"), counts["rejected"])

    review, search, agents, dashboard, curricula, profile, audit = st.tabs([
        t("tab_review"), t("tab_search"), t("tab_agents"), t("tab_dashboard"), t("tab_curricula"),
        t("tab_profile"), t("tab_audit"),
    ])
    with review:
        review_tab(counts)
    with search:
        search_tab()
    with agents:
        agents_tab()
    with dashboard:
        dashboard_tab()
    with curricula:
        curricula_tab()
    with profile:
        profile_tab()
    with audit:
        audit_tab()


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main() -> None:
    init_session()
    st.set_page_config(
        page_title="REX AI",
        page_icon=":material/school:",
        layout="wide" if st.session_state.view == "admin" else "centered",
    )
    apply_style()

    views = {"welcome": welcome_view, "user": user_view, "admin": admin_view}
    views.get(st.session_state.view, welcome_view)()


main()
