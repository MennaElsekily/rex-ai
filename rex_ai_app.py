"""
REX AI -- one Streamlit app, two doors.

    User  -> REX Assistant directly (agents/user_service.py)
    Admin -> passcode checked on the server (agents/admin_service.py)
          -> Admin orchestrator -> RFP Research, Curriculum Research, Document Generation, Marketing
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
from typing import Callable

import streamlit as st

from agents import admin_service as admin_module
from agents import register_agents
from agents import user_service as user_module
from agents.document_generation.templates import CURRICULUM_PDF, EXTRA_FIELDS, TEMPLATES, field_label, type_label
from agents.marketing import schemas as marketing_schemas
from agents.curriculum_research import editing as curriculum_editing
from agents.rfp_research.application_profile import LONG_PROFILE_FIELDS
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
        "new_conversation": "New conversation",
        "floating_chat_label": "Ask REX AI",
        "agents_involved": "Agents involved",
        "partial_failure": "Part of this request could not be completed; see below.",
        "resumed_after_approval": "Curriculum approved — continuing your request automatically.",
        "bilingual_companion_ready": "Its Arabic/English companion was also prepared: `{id}`. Review it in the Curricula tab.",
        "bilingual_companion_failed": "Its bilingual companion could not be prepared automatically: {error} You can "
                                      "still use this version; a companion is not required.",
        "cur_card_title": "Review and download",
        "cur_card_language": {"en": "English", "ar": "Arabic"},
        "cur_card_draft": "Draft: fix these validation problems (Curricula tab → Edit this version) before it can "
                          "be approved:",
        "cur_card_more": "Full content, side-by-side view and editing are in the Curricula tab.",
        "awaiting_approval": "Waiting for the curriculum to be approved.",
        "country": "Country",
        "category": "Category",
        "keyword": "Keyword (title, organisation, summary)",
        "no_audit": "No review decisions yet.",
        "audit_intro": "Every decision, newest first: who did what, when, and the status it changed.",
        "audit_area": "Area",
        "audit_areas": {"tender": "Tenders", "curriculum": "Curricula", "document": "Documents",
                        "campaign": "Marketing"},
        "audit_columns": {"when": "When (UTC)", "area": "Area", "item": "Item", "version": "Version",
                          "action": "Action", "status": "Status", "by": "By", "notes": "Notes"},
        "audit_tender_detail": "Tender decisions (full detail)",
        "details": "Details",
        # Approval and applications
        "approve_title": "Approve tender",
        "route_label": "How to apply",
        "automatic_yes": "REX AI can prepare this application (dry run: it is never sent; "
                         "you review and send it yourself).",
        "automatic_no": "Manual only: REX AI records the official steps for you.",
        "authorise_label": "I authorise REX AI to submit on the academy's behalf where supported",
        "authorise_help": "Without this, nothing is ever submitted or sent for you. Approving prepares a "
                          "personalised email draft only; sending it is always a separate step you confirm.",
        "confirm_approve": "Approve",
        "cancel": "Cancel",
        "approved_flash": "Approved. Application: {status}.",
        "open_application_email": "Review application email",
        "email_ready_flash": " Its application email draft is ready to review in Tender dashboard.",
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
        "will_send_to": "Will be sent to",
        "tender_contact": "Tender's own contact",
        "tender_contact_unused": "never used: every email goes to the configured recipient",
        "no_verified_contact": "none verified",
        "authorise_to_send": "Sending needs your authorisation to submit on the academy's behalf: use "
                             "\"Authorise submission\" below, then send.",
        "from": "From",
        "subject": "Subject",
        "attachments": "Attachments",
        "recommended_missing": "Recommended but not uploaded",
        "send_test_email_help": "This really sends the message above, through Resend -- the only action in REX AI "
                                "that does. Test mode (the default) always sends it to **{recipient}**, never to "
                                "the tender's own contact, so you can see a real email arrive without risking a "
                                "real submission.",
        "send_production_warning": "Production mode is active: this will really send the email to the configured "
                                   "recipient, **{recipient}** (never to the tender's own contact).",
        "confirm_send": "I understand this sends a real email",
        "send_now": "Send now",
        "email_sent_summary": "Sent to {to} on {at}.",
        "email_redirected_note": "Test mode: the tender's real contact ({real}) was never used.",
        "email_send_failed": "The send failed: {error}",
        "body": "Body",
        "demo_profile_email_warning": "This draft was generated from a DEMONSTRATION profile (the saved "
                                      "contact email uses the reserved example.com domain), not a real "
                                      "organisation's details.",
        "draft_edited_note": "Edited by {by} on {at}.",
        "save_draft_email": "Save draft",
        "discard_draft_email": "Discard edit (revert to the generated draft)",
        "recipient_not_configured": "Cannot send: REX_EMAIL_TEST_RECIPIENT is not configured with a valid email "
                                    "address. Every application email goes only to that address; set it in your "
                                    ".env file first.",
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
        # Marketing (Agent 5)
        "tab_marketing": "Marketing",
        "mk_intro": "Create marketing campaigns for an approved course: a simple strategy and several variations of social "
                    "captions, ad copy, LinkedIn posts, e-mail, WhatsApp messages, headlines and calls to action. Every "
                    "campaign waits for your approval. Nothing is published, e-mailed or sent: approved content can only "
                    "be downloaded.",
        "mk_status": {"draft": "Draft", "pending_review": "Pending review", "approved": "Approved"},
        "mk_new": "New campaign",
        "mk_none_approved": "No course is approved yet. Approve a curriculum in the Curricula tab first.",
        "mk_list_error": "The approved courses could not be listed: {error}",
        "mk_showing": "Showing {shown} of {total} approved courses.",
        "mk_course": "Approved course",
        "mk_audience": "Target audience (confirm or change)",
        "mk_audience_from_course": "Taken from the approved course. Change it only if this campaign targets someone else.",
        "mk_audience_missing": "The approved course does not state its target audience. Describe who this campaign is for.",
        "mk_tone": "Tone",
        "mk_channels": "Content to generate",
        "mk_variations": "Variations of each",
        "mk_goal": "Campaign goal (optional)",
        "mk_notes": "Extra facts you supply (optional, for example a start date)",
        "mk_language_note": "Language: {language}. A campaign is written in its course's language and is not translated.",
        "mk_generate": "Create campaign",
        "mk_generating": "Writing the campaign. This can take several minutes on a small local model (one part at a time). "
                         "Nothing is saved until every part has passed the checks.",
        "mk_progress": "Writing {label}: step {done} of {total}",
        "mk_progress_saving": "All parts written. Checking and saving…",
        "mk_need_channel": "Choose at least one kind of content.",
        "mk_filter": "Filter by status",
        "mk_all": "All",
        "mk_none": "No campaigns yet.",
        "mk_page": "Page",
        "mk_version": "Version",
        "mk_current": "current",
        "mk_approved_tag": "version {n} is approved",
        "mk_audience_label": "Target audience",
        "mk_tone_label": "Tone",
        "mk_course_changed": "The course has a newer approved version than the one this campaign was written from.",
        "mk_problems": "Problems that must be fixed before this version can be approved",
        "mk_strategy": "Campaign strategy",
        "mk_objective": "Objective",
        "mk_main_message": "Main message",
        "mk_rec_channels": "Recommended channels",
        "mk_sequence": "Suggested content sequence (step {n})",
        "mk_cta": "Call to action",
        "mk_variation": "Variation {n}",
        "mk_edit_approved": "Version {n} is approved. Saving an edit creates a new version that needs approval; the approved version stays as it is.",
        "mk_save_edit": "Save edits as a new version",
        "mk_unchanged": "No changes to save.",
        "mk_edited_msg": "Saved as a new version.",
        "mk_edit_draft": "Saved, but it does not pass the checks, so it is a draft and cannot be approved until the problems are fixed.",
        "mk_hashtags": "Hashtags (separated by spaces)",
        "mk_review_notes": "Review notes (optional)",
        "mk_approve": "Approve this version",
        "mk_approved_msg": "Approved. It can now be downloaded. Nothing was published or sent.",
        "mk_not_reviewable": "Only the current version, while it is pending review, can be approved.",
        "mk_download_note": "Downloads are the approved version {n} only.",
        "mk_download_md": "Download approved version (Markdown)",
        "mk_download_json": "Download approved version (JSON)",
        "mk_nothing_to_download": "Nothing to download yet: draft and pending-review content is not ready for use. Approve a version first.",
        "mk_regenerate": "Write a new version with the same choices",
        "mk_audit": "Review history",
        # Documents (Agent 4)
        "tab_documents": "Documents",
        "doc_intro": "Generate training documents from an approved curriculum, review and edit the text the AI wrote, "
                     "then approve the final version and download the PDF. Text copied from the curriculum cannot be "
                     "edited here: revise and re-approve the curriculum instead.",
        "doc_status": {"draft": "Draft", "pending_review": "Pending review", "approved": "Approved"},
        "doc_new": "New document",
        "doc_none_approved": "No curriculum is approved yet. Approve one in the Curricula tab first.",
        "doc_curriculum": "Approved curriculum",
        "doc_type": "Document type",
        "doc_format": "Training format (optional)",
        "doc_org": "Organisation or client (optional)",
        "doc_schedule": "Schedule notes (optional)",
        "doc_generate": "Generate document",
        "doc_none": "No documents yet.",
        "doc_filter": "Show",
        "doc_all": "All",
        "doc_version": "Version",
        "doc_current": "current",
        "doc_curriculum_changed": "The curriculum has been revised since this document was built. It still uses the "
                                  "approved version it names.",
        "doc_missing": "Not specified in the curriculum (marked in the document)",
        "doc_problems": "Validation problems",
        "doc_edit_hint": "Editable text (written by the AI or an administrator)",
        "doc_save_edit": "Save edits as a new version",
        "doc_edited_msg": "Saved as a new version. It needs review again.",
        "doc_notes": "Review notes (optional)",
        "doc_approve": "Approve this version",
        "doc_approved_msg": "Document approved.",
        "doc_regenerate": "Regenerate (new version)",
        "doc_not_reviewable": "Only the current version, while it is pending review, can be approved.",
        "doc_pdf_download": "Download PDF",
        "doc_pdf_create": "Create PDF",
        "doc_audit": "Review history",
        "doc_unchanged": "No changes to save.",
        "doc_approved_tag": "approved v{n}",
        "doc_edit_approved": "This version is approved. Saving your edits creates a new version for review; "
                             "version {n} stays the approved one until the new version is approved.",
        "doc_approved_pdf": "Download approved PDF",
        "doc_review_pdf": "Download PDF (review copy)",
        "doc_page": "Page",
        "doc_showing": "Showing {shown} of {total} approved curricula (most recent first).",
        "doc_list_error": "The approved curricula could not be listed: {error}",
        "doc_generating": "Generating the document. This can take several minutes (the model writes one module at a "
                          "time). Nothing is saved until every part has passed the checks.",
        "doc_progress": "Writing {label}: step {done} of {total}",
        "doc_progress_saving": "All parts written. Checking, saving and creating the PDF…",
        "doc_language_note": "Language: {language}. A document is written in its curriculum's language and is not translated.",
        "doc_cpdf_note": "The Curriculum PDF is the Curriculum Agent's own PDF of the approved version. It is not a "
                         "generated document: nothing is created or reviewed here.",
        "doc_cpdf_download": "Download curriculum PDF (approved version {n})",
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
        "cur_pdf_download_approved": "Download approved PDF (v{v})",
        "cur_pdf_after_approval": "The PDF is generated when an administrator approves this curriculum.",
        "cur_pdf_newer_pending": "v{v} is newer and not approved yet; this PDF is the last approved version.",
        "cur_zip_after_approval": "The ZIP of both languages is available once both are approved.",
        "cur_bilingual_companion": "Bilingual companion: {id}",
        "cur_show_side_by_side": "Show both languages side by side",
        "cur_download_both": "Download both PDFs (ZIP)",
        "cur_edit": "Edit this version",
        "cur_edit_only_current": "Select the current version above to edit it.",
        "cur_edit_title": "Course title",
        "cur_edit_overview": "Course description",
        "cur_edit_audience": "Target audience",
        "cur_edit_prerequisites": "Prerequisites",
        "cur_edit_objectives": "Learning objectives",
        "cur_edit_outcomes": "Expected learning outcomes",
        "cur_edit_outcomes_help": "One per line, as \"LO1: statement\". A line with no id gets the next free one.",
        "cur_edit_modules": "Modules",
        "cur_edit_modules_help": "One module per block, separated by a line of dashes (--------------------). "
                                 "Fields: id (leave blank for a new module), title, hours, topics (comma-separated), "
                                 "description, activities (\"type: description\", separated by |), outcome_ids "
                                 "(comma-separated). Delete a whole block to remove that module.",
        "cur_edit_assessments": "Assessment methods",
        "cur_edit_assessments_help": "One assessment per block, separated by a line of dashes. Fields: id (blank "
                                     "for new), type, description, outcome_ids (comma-separated).",
        "cur_edit_include_capstone": "Include a capstone project",
        "cur_edit_capstone_title": "Capstone title",
        "cur_edit_capstone_description": "Capstone description",
        "cur_edit_capstone_outcomes": "Capstone learning outcomes (comma-separated ids)",
        "cur_edit_capstone_deliverables": "Capstone deliverables",
        "cur_edit_tools": "Tools and resources",
        "cur_edit_assumptions": "Additional notes and assumptions",
        "cur_edit_one_per_line": "One per line.",
        "cur_save_edit": "Save changes",
        "cur_edit_saved_msg": "Changes saved as a new version, pending review.",
        "cur_edit_saved_draft_msg": "Changes saved as a new version, but it is a draft: fix the validation problems "
                                    "shown to make it reviewable.",
        "cur_notes": "Notes (required to request changes)",
        "cur_approve": "Approve",
        "cur_request_changes": "Request changes",
        "cur_not_reviewable": "Only the current version, while pending review, can be approved or sent back.",
        "cur_why_not_reviewable": {
            "draft": "Draft: it cannot be approved until the validation problems above are fixed. Use "
                     "\"Edit this version\" below and save a corrected version.",
            "approved": "Approved. Editing it creates a new version that needs its own approval.",
            "changes_requested": "Changes were requested. Edit it below to save a new version for review.",
            "old": "This is an older version. Select the current version to review it.",
        },
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
                        "contact_phone": "Contact phone", "website": "Website",
                        "overview": "Organisation overview", "services": "Main services and areas of expertise",
                        "experience": "Relevant experience and previous projects",
                        "qualifications": "Qualifications and certifications"},
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
        "new_conversation": "محادثة جديدة",
        "floating_chat_label": "اسأل ريكس",
        "agents_involved": "الوكلاء المشاركون",
        "partial_failure": "جزء من هذا الطلب لم يكتمل؛ انظر أدناه.",
        "resumed_after_approval": "تم اعتماد المنهج — سيستمر طلبك تلقائيًا.",
        "bilingual_companion_ready": "تم أيضًا تجهيز نسخته العربية/الإنجليزية المقابلة: `{id}`. راجعها في تبويب المناهج.",
        "bilingual_companion_failed": "تعذّر تجهيز نسخته اللغوية المقابلة تلقائيًا: {error} يمكنك الاستمرار "
                                      "باستخدام هذه النسخة؛ النسخة المقابلة ليست مطلوبة.",
        "cur_card_title": "المراجعة والتنزيل",
        "cur_card_language": {"en": "الإنجليزية", "ar": "العربية"},
        "cur_card_draft": "مسودة: أصلح مشكلات التحقق التالية (تبويب المناهج ← تعديل هذه النسخة) قبل أن يمكن اعتمادها:",
        "cur_card_more": "المحتوى الكامل والعرض جنبًا إلى جنب والتعديل متاحة في تبويب المناهج.",
        "awaiting_approval": "في انتظار اعتماد المنهج.",
        "country": "الدولة",
        "category": "المجال",
        "keyword": "كلمة (العنوان، الجهة، الملخص)",
        "no_audit": "لا توجد قرارات مراجعة بعد.",
        "audit_intro": "كل القرارات، الأحدث أولًا: من فعل ماذا، ومتى، والحالة التي تغيّرت.",
        "audit_area": "المجال",
        "audit_areas": {"tender": "المناقصات", "curriculum": "المناهج", "document": "الوثائق",
                        "campaign": "التسويق"},
        "audit_columns": {"when": "الوقت (UTC)", "area": "المجال", "item": "العنصر", "version": "النسخة",
                          "action": "الإجراء", "status": "الحالة", "by": "بواسطة", "notes": "ملاحظات"},
        "audit_tender_detail": "قرارات المناقصات (تفاصيل كاملة)",
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
        "open_application_email": "مراجعة بريد التقديم",
        "email_ready_flash": " مسودة بريد التقديم جاهزة للمراجعة في لوحة المناقصات.",
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
        "will_send_to": "سيُرسَل إلى",
        "tender_contact": "جهة اتصال المناقصة",
        "tender_contact_unused": "لا تُستخدم أبدًا: يُرسَل كل بريد إلى المستلم المُهيأ",
        "no_verified_contact": "لا يوجد عنوان موثّق",
        "authorise_to_send": "الإرسال يتطلب تفويضك بالتقديم نيابةً عن الأكاديمية: استخدم «تفويض التقديم» أدناه "
                             "ثم أرسل.",
        "from": "من",
        "subject": "الموضوع",
        "attachments": "المرفقات",
        "recommended_missing": "مستحسن لكن لم يُرفع",
        "send_test_email_help": "هذا يرسل الرسالة أعلاه فعليًا عبر Resend -- الإجراء الوحيد في REX AI الذي يفعل "
                                "ذلك. وضع التجربة (الافتراضي) يرسلها دائمًا إلى **{recipient}**، لا إلى جهة اتصال "
                                "المنافسة نفسها، لتتمكن من رؤية بريد حقيقي يصل دون المخاطرة بتقديم فعلي.",
        "send_production_warning": "وضع الإنتاج مفعَّل: سيُرسَل هذا البريد فعليًا إلى المستلم المُهيأ "
                                   "**{recipient}** (وليس أبدًا إلى جهة اتصال المناقصة).",
        "confirm_send": "أدرك أن هذا يرسل بريدًا إلكترونيًا حقيقيًا",
        "send_now": "إرسال الآن",
        "email_sent_summary": "أُرسل إلى {to} في {at}.",
        "email_redirected_note": "وضع التجربة: لم تُستخدم جهة اتصال المنافسة الحقيقية ({real}).",
        "email_send_failed": "فشل الإرسال: {error}",
        "body": "النص",
        "demo_profile_email_warning": "هذا المسودة تم إنشاؤها من ملف تعريف تجريبي (البريد الإلكتروني "
                                      "المحفوظ يستخدم نطاق example.com المحجوز)، وليس تفاصيل منظمة حقيقية.",
        "draft_edited_note": "عدّله {by} في {at}.",
        "save_draft_email": "حفظ المسودة",
        "discard_draft_email": "تجاهل التعديل (الرجوع إلى المسودة المُنشأة)",
        "recipient_not_configured": "لا يمكن الإرسال: REX_EMAIL_TEST_RECIPIENT غير مُهيأ ببريد إلكتروني صالح. "
                                    "يُرسَل كل بريد تقديم إلى هذا العنوان فقط؛ اضبطه في ملف .env أولًا.",
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
        # Marketing (Agent 5)
        "tab_marketing": "التسويق",
        "mk_intro": "أنشئ حملات تسويقية لمقرر معتمد: استراتيجية بسيطة وعدة صيغ من تعليقات وسائل التواصل ونصوص الإعلانات ومنشورات "
                    "لينكدإن والبريد ورسائل واتساب والعناوين والدعوات إلى اتخاذ إجراء. تنتظر كل حملة اعتمادك. لا يُنشر ولا "
                    "يُرسل شيء: المحتوى المعتمد يمكن تنزيله فقط.",
        "mk_status": {"draft": "مسودة", "pending_review": "بانتظار المراجعة", "approved": "معتمدة"},
        "mk_new": "حملة جديدة",
        "mk_none_approved": "لا يوجد مقرر معتمد بعد. اعتمد منهجًا في تبويب المناهج أولًا.",
        "mk_list_error": "تعذر عرض المقررات المعتمدة: {error}",
        "mk_showing": "عرض {shown} من {total} مقررًا معتمدًا.",
        "mk_course": "المقرر المعتمد",
        "mk_audience": "الجمهور المستهدف (أكّده أو غيّره)",
        "mk_audience_from_course": "مأخوذ من المقرر المعتمد. غيّره فقط إذا كانت الحملة موجهة لجمهور آخر.",
        "mk_audience_missing": "لا يذكر المقرر المعتمد جمهوره المستهدف. صف لمن هذه الحملة.",
        "mk_tone": "النبرة",
        "mk_channels": "المحتوى المطلوب",
        "mk_variations": "عدد الصيغ لكل نوع",
        "mk_goal": "هدف الحملة (اختياري)",
        "mk_notes": "معلومات إضافية تقدمها (اختياري، مثل موعد البدء)",
        "mk_language_note": "اللغة: {language}. تُكتب الحملة بلغة مقررها ولا تُترجم.",
        "mk_generate": "إنشاء الحملة",
        "mk_generating": "جارٍ كتابة الحملة. قد يستغرق ذلك عدة دقائق على نموذج محلي صغير (جزء بعد جزء). لا يُحفظ شيء إلا بعد "
                         "أن يجتاز كل جزء الفحوص.",
        "mk_progress": "جارٍ كتابة {label}: الخطوة {done} من {total}",
        "mk_progress_saving": "اكتملت كتابة الأجزاء. جارٍ الفحص والحفظ…",
        "mk_need_channel": "اختر نوعًا واحدًا على الأقل من المحتوى.",
        "mk_filter": "تصفية حسب الحالة",
        "mk_all": "الكل",
        "mk_none": "لا توجد حملات بعد.",
        "mk_page": "الصفحة",
        "mk_version": "النسخة",
        "mk_current": "الحالية",
        "mk_approved_tag": "النسخة {n} معتمدة",
        "mk_audience_label": "الجمهور المستهدف",
        "mk_tone_label": "النبرة",
        "mk_course_changed": "للمقرر نسخة معتمدة أحدث من النسخة التي كُتبت منها هذه الحملة.",
        "mk_problems": "مشكلات يجب إصلاحها قبل اعتماد هذه النسخة",
        "mk_strategy": "استراتيجية الحملة",
        "mk_objective": "الهدف",
        "mk_main_message": "الرسالة الرئيسية",
        "mk_rec_channels": "القنوات الموصى بها",
        "mk_sequence": "تسلسل المحتوى المقترح (الخطوة {n})",
        "mk_cta": "الدعوة إلى اتخاذ إجراء",
        "mk_variation": "الصيغة {n}",
        "mk_edit_approved": "النسخة {n} معتمدة. يؤدي حفظ تعديل إلى نسخة جديدة تحتاج اعتمادًا؛ وتبقى النسخة المعتمدة كما هي.",
        "mk_save_edit": "حفظ التعديلات كنسخة جديدة",
        "mk_unchanged": "لا توجد تغييرات للحفظ.",
        "mk_edited_msg": "تم الحفظ كنسخة جديدة.",
        "mk_edit_draft": "تم الحفظ، لكنها لا تجتاز الفحوص فهي مسودة ولا يمكن اعتمادها قبل إصلاح المشكلات.",
        "mk_hashtags": "الوسوم (تفصل بينها مسافات)",
        "mk_review_notes": "ملاحظات المراجعة (اختياري)",
        "mk_approve": "اعتماد هذه النسخة",
        "mk_approved_msg": "تم الاعتماد. يمكن الآن تنزيلها. لم يُنشر ولم يُرسل شيء.",
        "mk_not_reviewable": "لا يمكن اعتماد إلا النسخة الحالية وهي بانتظار المراجعة.",
        "mk_download_note": "التنزيلات هي النسخة المعتمدة {n} فقط.",
        "mk_download_md": "تنزيل النسخة المعتمدة (Markdown)",
        "mk_download_json": "تنزيل النسخة المعتمدة (JSON)",
        "mk_nothing_to_download": "لا شيء للتنزيل بعد: المسودات والمحتوى بانتظار المراجعة غير جاهز للاستخدام. اعتمد نسخة أولًا.",
        "mk_regenerate": "كتابة نسخة جديدة بالاختيارات نفسها",
        "mk_audit": "سجل المراجعة",
        # Documents (Agent 4)
        "tab_documents": "الوثائق",
        "doc_intro": "أنشئ وثائق تدريبية من منهج معتمد، وراجع النص الذي كتبه الذكاء الاصطناعي وعدّله، ثم اعتمد النسخة "
                     "النهائية ونزّل ملف PDF. لا يمكن تعديل النص المنقول من المنهج هنا: عدّل المنهج وأعد اعتماده.",
        "doc_status": {"draft": "مسودة", "pending_review": "بانتظار المراجعة", "approved": "معتمدة"},
        "doc_new": "وثيقة جديدة",
        "doc_none_approved": "لا يوجد منهج معتمد بعد. اعتمد منهجًا في تبويب المناهج أولًا.",
        "doc_curriculum": "المنهج المعتمد",
        "doc_type": "نوع الوثيقة",
        "doc_format": "نمط التدريب (اختياري)",
        "doc_org": "الجهة أو العميل (اختياري)",
        "doc_schedule": "ملاحظات الجدول (اختياري)",
        "doc_generate": "إنشاء الوثيقة",
        "doc_none": "لا توجد وثائق بعد.",
        "doc_filter": "عرض",
        "doc_all": "الكل",
        "doc_version": "النسخة",
        "doc_current": "الحالية",
        "doc_curriculum_changed": "عُدّل المنهج بعد إنشاء هذه الوثيقة. ما زالت تعتمد على النسخة المعتمدة المذكورة فيها.",
        "doc_missing": "غير محدد في المنهج (موضّح في الوثيقة)",
        "doc_problems": "مشكلات التحقق",
        "doc_edit_hint": "نص قابل للتعديل (كتبه الذكاء الاصطناعي أو مسؤول)",
        "doc_save_edit": "حفظ التعديلات كنسخة جديدة",
        "doc_edited_msg": "حُفظت كنسخة جديدة. تحتاج إلى مراجعة من جديد.",
        "doc_notes": "ملاحظات المراجعة (اختياري)",
        "doc_approve": "اعتماد هذه النسخة",
        "doc_approved_msg": "تم اعتماد الوثيقة.",
        "doc_regenerate": "إعادة الإنشاء (نسخة جديدة)",
        "doc_not_reviewable": "لا يمكن اعتماد إلا النسخة الحالية وهي بانتظار المراجعة.",
        "doc_pdf_download": "تنزيل PDF",
        "doc_pdf_create": "إنشاء PDF",
        "doc_audit": "سجل المراجعة",
        "doc_unchanged": "لا توجد تعديلات للحفظ.",
        "doc_approved_tag": "المعتمدة: النسخة {n}",
        "doc_edit_approved": "هذه النسخة معتمدة. حفظ تعديلاتك ينشئ نسخة جديدة للمراجعة، وتبقى النسخة {n} هي "
                             "المعتمدة إلى أن تُعتمد النسخة الجديدة.",
        "doc_approved_pdf": "تنزيل ملف PDF المعتمد",
        "doc_review_pdf": "تنزيل PDF (نسخة المراجعة)",
        "doc_page": "الصفحة",
        "doc_showing": "عرض {shown} من {total} منهجًا معتمدًا (الأحدث أولًا).",
        "doc_list_error": "تعذر عرض المناهج المعتمدة: {error}",
        "doc_generating": "جارٍ إنشاء الوثيقة. قد يستغرق ذلك عدة دقائق (يكتب النموذج وحدة تلو الأخرى). لا يُحفظ شيء "
                          "إلا بعد أن يجتاز كل جزء الفحوص.",
        "doc_progress": "جارٍ كتابة {label}: الخطوة {done} من {total}",
        "doc_progress_saving": "اكتملت كتابة الأجزاء. جارٍ الفحص والحفظ وإنشاء ملف PDF…",
        "doc_language_note": "اللغة: {language}. تُكتب الوثيقة بلغة منهجها ولا تُترجم.",
        "doc_cpdf_note": "ملف PDF للمنهج هو ملف وكيل المناهج للنسخة المعتمدة. وهو ليس وثيقة مُنشأة: لا يُنشأ شيء ولا "
                         "تجري مراجعة هنا.",
        "doc_cpdf_download": "تنزيل ملف PDF للمنهج (النسخة المعتمدة {n})",
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
        "cur_pdf_download_approved": "تنزيل ملف PDF المعتمد (v{v})",
        "cur_pdf_after_approval": "يُنشأ ملف PDF عندما يعتمد المسؤول هذا المنهج.",
        "cur_pdf_newer_pending": "النسخة v{v} أحدث ولم تُعتمد بعد؛ هذا الملف هو آخر نسخة معتمدة.",
        "cur_zip_after_approval": "يتوفر ملف ZIP للغتين بعد اعتماد كلتيهما.",
        "cur_bilingual_companion": "النسخة اللغوية المقابلة: {id}",
        "cur_show_side_by_side": "عرض اللغتين جنبًا إلى جنب",
        "cur_download_both": "تنزيل ملفي PDF معًا (ZIP)",
        "cur_edit": "تعديل هذه النسخة",
        "cur_edit_only_current": "اختر النسخة الحالية أعلاه لتعديلها.",
        "cur_edit_title": "عنوان الدورة",
        "cur_edit_overview": "وصف الدورة",
        "cur_edit_audience": "الجمهور المستهدف",
        "cur_edit_prerequisites": "المتطلبات المسبقة",
        "cur_edit_objectives": "أهداف التعلم",
        "cur_edit_outcomes": "مخرجات التعلم المتوقعة",
        "cur_edit_outcomes_help": "سطر لكل مخرج، بصيغة «LO1: النص». السطر بلا معرّف يُعطى المعرّف التالي المتاح.",
        "cur_edit_modules": "الوحدات",
        "cur_edit_modules_help": "وحدة واحدة لكل كتلة، تفصل بينها سطر شرطات (--------------------). الحقول: "
                                "id (اتركه فارغًا لوحدة جديدة)، title، hours، topics (مفصولة بفواصل)، description، "
                                "activities («النوع: الوصف»، مفصولة بـ |)، outcome_ids (مفصولة بفواصل). لحذف "
                                "وحدة، احذف كتلتها بالكامل.",
        "cur_edit_assessments": "طرق التقييم",
        "cur_edit_assessments_help": "تقييم واحد لكل كتلة، تفصل بينها سطر شرطات. الحقول: id (فارغ لتقييم جديد)، "
                                     "type، description، outcome_ids (مفصولة بفواصل).",
        "cur_edit_include_capstone": "تضمين مشروع ختامي",
        "cur_edit_capstone_title": "عنوان المشروع الختامي",
        "cur_edit_capstone_description": "وصف المشروع الختامي",
        "cur_edit_capstone_outcomes": "مخرجات التعلم للمشروع الختامي (معرّفات مفصولة بفواصل)",
        "cur_edit_capstone_deliverables": "مخرجات تسليم المشروع الختامي",
        "cur_edit_tools": "الأدوات والموارد",
        "cur_edit_assumptions": "ملاحظات وافتراضات إضافية",
        "cur_edit_one_per_line": "سطر واحد لكل عنصر.",
        "cur_save_edit": "حفظ التعديلات",
        "cur_edit_saved_msg": "حُفظت التعديلات كنسخة جديدة، في انتظار المراجعة.",
        "cur_edit_saved_draft_msg": "حُفظت التعديلات كنسخة جديدة، لكنها مسودة: أصلح مشكلات التحقق المعروضة "
                                    "لتصبح قابلة للمراجعة.",
        "cur_notes": "ملاحظات (مطلوبة عند طلب التعديلات)",
        "cur_approve": "اعتماد",
        "cur_request_changes": "طلب تعديلات",
        "cur_not_reviewable": "يمكن اعتماد النسخة الحالية أو إعادتها فقط وهي بانتظار المراجعة.",
        "cur_why_not_reviewable": {
            "draft": "مسودة: لا يمكن اعتمادها حتى تُصلح مشكلات التحقق أعلاه. استخدم «تعديل هذه النسخة» "
                     "أدناه واحفظ نسخة مصححة.",
            "approved": "معتمدة. تعديلها يُنشئ نسخة جديدة تحتاج إلى اعتماد خاص بها.",
            "changes_requested": "طُلبت تعديلات. عدّلها أدناه لحفظ نسخة جديدة للمراجعة.",
            "old": "هذه نسخة أقدم. اختر النسخة الحالية لمراجعتها.",
        },
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
                        "contact_phone": "هاتف التواصل", "website": "الموقع الإلكتروني",
                        "overview": "نظرة عامة عن المنظمة", "services": "الخدمات الرئيسية ومجالات الخبرة",
                        "experience": "الخبرة ذات الصلة والمشاريع السابقة",
                        "qualifications": "المؤهلات والشهادات"},
        "doc_label": {"company_profile": "ملف تعريف الشركة",
                      "registration_certificate": "شهادة السجل التجاري",
                      "tax_card": "البطاقة الضريبية", "experience_references": "خطابات الخبرة والمراجع",
                      "cvs": "السير الذاتية للفريق", "other": "أخرى"},
    },
}

AGENT_LABELS = {
    "rfp_research": {"en": "RFP Research", "ar": "وكيل المناقصات"},
    "curriculum_research": {"en": "Curriculum Research", "ar": "وكيل المناهج"},
    "document_generation": {"en": "Document Generation", "ar": "وكيل الوثائق"},
    "marketing": {"en": "Marketing", "ar": "وكيل التسويق"},
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
        "admin_section": "review",   # the active admin section; only the sidebar (or "open in ...") changes it
        "open_curriculum": None,     # a curriculum the admin asked to open in the Curriculum section
        "open_tender": None,         # an approved tender the admin asked to open in Tender dashboard
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


CSS = """
<style>
/* Deliberately minimal: the theme (.streamlit/config.toml) sets the white background and the one
   accent colour; this only adds what the theme cannot -- a title style and small status badges. */
.rex-hero {padding: 1.4rem 0 .6rem; border-bottom: 1px solid #e5e7eb; margin-bottom: 1.2rem;}
.rex-hero h1 {font-size: 2.1rem; margin: 0; letter-spacing: -0.01em;}
.rex-hero p {color: #6b7280; margin: .25rem 0 0; font-size: 1.02rem;}
.rex-title {display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .6rem;}
.rex-title h2 {margin: 0;}

.stButton button, .stDownloadButton button, .stFormSubmitButton button {border-radius: 8px;}
[data-testid="stSidebar"] .stButton button {justify-content: flex-start;}

.rex-badge {display: inline-block; padding: .05rem .55rem; border-radius: 999px;
  font-size: .78rem; font-weight: 600; margin-inline-end: .35rem; border: 1px solid transparent;}
.rex-pending, .rex-cur-pending_review, .rex-sub-requires_manual_action, .rex-sub-in_progress
  {background: #fff7e6; color: #92400e; border-color: #fde3b0;}
.rex-approved, .rex-cur-approved, .rex-sub-submitted, .rex-app-applied, .rex-app-awarded
  {background: #ecfdf3; color: #05603a; border-color: #b7ebc9;}
.rex-rejected, .rex-flagged, .rex-cur-changes_requested, .rex-sub-submission_failed,
.rex-sub-outcome_unknown, .rex-app-not_awarded {background: #fef2f2; color: #991b1b; border-color: #fbd0d0;}
.rex-agent, .rex-monitored, .rex-priority {background: #eef2ff; color: #3b4cb8; border-color: #d6dcfb;}
.rex-cur-draft, .rex-sub-not_started, .rex-app-not_applied, .rex-app-withdrawn
  {background: #f3f4f6; color: #374151; border-color: #e5e7eb;}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
.rex-auto p {unicode-bidi: plaintext; text-align: start;}
</style>
"""

RTL_CSS = """
<style>
[data-testid="stMainBlockContainer"], .main .block-container,
[data-testid="stDialog"] [role="dialog"] {direction: rtl;}
[data-testid="stSidebar"] {direction: rtl;}
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


def section_header(icon: str, title_key: str) -> None:
    """One consistent header (icon + title + a quiet divider) at the top of every admin section."""
    st.subheader(t(title_key), icon=icon, divider="gray")


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
            client_id = st.session_state.setdefault("admin_client_id", uuid.uuid4().hex)
            token = admin.login(code, client_id)
            if token:
                st.session_state.admin_token = token
                st.session_state.passcode_open = False
                st.session_state.view = "admin"
                st.rerun()
            elif admin.is_locked(client_id):
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
            # Exactly what the knowledge base returned: file, page, course,
            # chunk, ranking and the retrieved text (nothing is added here).
            reference = f"**[{source['reference']}]** " if source.get("reference") is not None else ""
            page = f", {t('page')} {source['page']}" if source.get("page") else ""
            course = (f" · {plain(source.get('course_id') or '')} {plain(source.get('course_name') or '')}"
                      if source.get("course_id") else "")
            st.markdown(f"{reference}{plain(source.get('source'))}{page}{course}")
            trace = [f"chunk {source['chunk_id']}" if source.get("chunk_id") is not None else "",
                     f"relevance {source['relevance_score']}" if source.get("relevance_score") is not None else "",
                     f"distance {source['distance']}" if source.get("distance") is not None else ""]
            if any(trace) or source.get("excerpt"):
                st.caption(" · ".join(p for p in trace if p) + (f" — {plain(source['excerpt'])}"
                                                                 if source.get("excerpt") else ""))


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
        if status == "approved" and buttons[column].button(t("open_application_email"), icon=":material/mail:",
                                                           key=f"open_email_{opp_id}"):
            open_tender_application(opp_id)


def open_tender_application(opp_id: str) -> None:
    """An explicit navigation the admin asked for: Tender dashboard, with this tender's application opened."""
    st.session_state.admin_section = "dashboard"
    st.session_state.open_tender = opp_id
    st.rerun()


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
            prepared = ((result.get("application") or {}).get("confirmation") or {}).get("prepared_email")
            st.session_state.flash = t("approved_flash").format(
                status=t("sub_status")[result["submission_status"]]) + (t("email_ready_flash") if prepared else "")
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
    section_header(":material/fact_check:", "tab_review")
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
    section_header(":material/travel_explore:", "tab_search")
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


def render_agent_result(result: dict, key_suffix: str = "") -> None:
    agent = result.get("agent_name", "")
    label = AGENT_LABELS.get(agent, {}).get(lang(), agent)
    st.markdown(f'<span class="rex-badge rex-agent">{label}</span>', unsafe_allow_html=True)

    if result.get("status") == "waiting_for_input":
        st.info((result.get("result") or {}).get("message") or t("awaiting_approval"))
        return
    if result.get("status") == "paused":
        body = result.get("result") or {}
        st.info(body.get("message") or t("awaiting_approval"))
        if body.get("curriculum_id"):
            st.caption(f"{t('tab_curricula')}: `{body['curriculum_id']}`")
            curriculum_review_card(body["curriculum_id"], key_suffix)
        return
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
    for i, opportunity in enumerate(body.get("opportunities") or []):
        with st.expander(plain(opportunity.get("title") or opportunity.get("url")),
                         key=f"agent_result_opp_{id(result)}_{i}{key_suffix}"):
            st.markdown(tender_badges(opportunity, with_status=False), unsafe_allow_html=True)
            tender_details(opportunity)
            tender_links(opportunity, st.columns(2))
    if body.get("curriculum_id"):
        st.caption(f"{t('tab_curricula')}: `{body['curriculum_id']}`")
    if body.get("bilingual_companion_id"):
        st.success(t("bilingual_companion_ready").format(id=body["bilingual_companion_id"]))
    elif body.get("bilingual_companion_error"):
        st.warning(t("bilingual_companion_failed").format(error=plain(body["bilingual_companion_error"])))
    if body.get("kind") == "curriculum" and body.get("curriculum_id"):
        curriculum_review_card(body["curriculum_id"], key_suffix)
    # The full curriculum is read in the Curricula tab, not as raw JSON here.
    extra = {k: v for k, v in body.items()
            if k not in ("message", "opportunities", "curriculum", "bilingual_companion_id", "bilingual_companion_error",
                         "bilingual_companion_status")}
    if extra:
        with st.expander(t("details"), key=f"agent_result_details_{id(result)}{key_suffix}"):
            st.json(extra, expanded=False)


def curriculum_review_card(curriculum_id: str, key_suffix: str) -> None:
    """
    Right under a curriculum result in the chat: the curriculum and its
    bilingual companion, each with its LIVE status, its PDF and -- only
    while its current version is pending review -- an Approve button for
    exactly that version. A draft says which validation problems keep it
    from being approved instead of silently showing no button.
    """
    companion_id = admin_call("curriculum_translation", curriculum_id)
    entries = [(cid, detail) for cid in [curriculum_id, *([companion_id] if companion_id else [])]
               if (detail := admin_call("get_curriculum", cid))]
    if not entries:
        return
    entries.sort(key=lambda entry: entry[1]["curriculum"]["content"].get("language") != "en")
    labels, languages = t("cur_status"), t("cur_card_language")

    st.markdown(f"**{t('cur_card_title')}**")
    for column, (cid, detail) in zip(st.columns(len(entries)), entries):
        record = detail["curriculum"]
        version, status, language = record["version"], record["status"], record["content"].get("language")
        with column.container(border=True):
            st.markdown(f"**{languages.get(language, language)}** · {plain(record['content'].get('title'))}")
            st.markdown(badge(f"cur-{status}", labels.get(status, status)) + f" `{cid}` · v{version}",
                        unsafe_allow_html=True)
            errors = (record.get("validation") or {}).get("errors") or []
            if status == "draft" and errors:
                st.warning(t("cur_card_draft") + "\n" + "\n".join(f"- {plain(e.get('message'))}" for e in errors[:5]))
                if st.button(t("cur_edit"), icon=":material/edit:", key=f"chat_cur_edit_{cid}_v{version}{key_suffix}"):
                    open_curriculum_section(cid)
            # The version is part of the key: a click can never approve a newer version than the one shown.
            if status == "pending_review" and st.button(t("cur_approve"), type="primary", icon=":material/check:",
                                                        key=f"chat_cur_approve_{cid}_v{version}{key_suffix}"):
                if admin_action("approve_curriculum", cid, version):
                    st.session_state.flash = t("cur_approved_msg")
                    st.rerun()
            approved_pdf_button(cid, record, f"chat_cur_pdf_{cid}{key_suffix}")
    if companion_id:
        bilingual_zip_button(curriculum_id, [detail["curriculum"] for _, detail in entries],
                             f"chat_cur_zip_{curriculum_id}{key_suffix}")
    st.caption(t("cur_card_more"))


def _download_from_service(method: str, *args) -> Callable[[], bytes]:
    """
    Download data built only when the admin clicks (Streamlit runs it on its
    own thread, without this session's state, hence the bound service and
    token). Pages never re-read every PDF on every rerun.
    """
    service, token = get_admin_service(), st.session_state.admin_token
    return lambda: getattr(service, method)(token, *args)["data"]


def approved_pdf_button(curriculum_id: str, record: dict, key: str) -> None:
    """The latest APPROVED version's PDF, or why there is none yet. Never a PDF of an unapproved version."""
    approved = record.get("approved_version")
    if not approved:
        st.caption(t("cur_pdf_after_approval"))
        return
    language = (record["content"].get("language") if approved == record["version"]
                else admin_call("get_curriculum", curriculum_id, approved)["curriculum"]["content"].get("language"))
    st.download_button(t("cur_pdf_download_approved").format(v=approved),
                       _download_from_service("curriculum_pdf", curriculum_id, approved),
                       file_name=f"{curriculum_id}_v{approved}_{language}.pdf", mime="application/pdf",
                       icon=":material/download:", on_click="ignore", key=f"{key}_v{approved}")
    if approved != record["version"]:
        st.caption(t("cur_pdf_newer_pending").format(v=record["version"]))


def bilingual_zip_button(curriculum_id: str, records: list[dict], key: str) -> None:
    """Both languages' approved PDFs in one ZIP -- offered once every language has an approved version."""
    if not all(r.get("approved_version") for r in records):
        st.caption(t("cur_zip_after_approval"))
        return
    st.download_button(t("cur_download_both"), _download_from_service("curricula_pdf_zip", curriculum_id),
                       file_name=f"{curriculum_id}_bilingual.zip", mime="application/zip",
                       icon=":material/folder_zip:", on_click="ignore", key=key)


def open_curriculum_section(curriculum_id: str) -> None:
    """An explicit navigation the admin asked for: the Curriculum section, with this curriculum opened."""
    st.session_state.admin_section = "curricula"
    st.session_state.open_curriculum = curriculum_id
    st.rerun()


def agents_tab(key_suffix: str = "") -> None:
    """
    One conversational interface for every admin agent: the admin types a
    request in plain language (English or Arabic), the existing Central
    Orchestrator decides which specialist agent(s) handle it -- including
    chained, multi-agent requests -- and the reply is shown as one
    assistant turn. Nothing here bypasses the orchestrator or creates a
    second routing system; this function only changes how its replies look.

    key_suffix: the floating launcher (floating_assistant()) renders this
    same function a second time, over the same conversation, so its
    widgets need their own keys -- Streamlit requires every widget key to
    be unique even when two widgets show the same data.
    """
    if not key_suffix:   # the floating popover is compact; the full header belongs to the tab only
        section_header(":material/forum:", "tab_agents")
    top = st.columns([5, 1])
    top[0].caption(t("agents_intro"))
    if top[1].button(t("new_conversation"), icon=":material/add_comment:", width="stretch",
                     key=f"new_conversation{key_suffix}", disabled=not st.session_state.admin_chat):
        st.session_state.admin_chat = []
        st.session_state.admin_conversation_id = uuid.uuid4().hex
        st.rerun()

    resumed = admin_call("pending_chat_update_for", st.session_state.admin_conversation_id)
    if resumed:
        st.session_state.admin_chat.append({"role": "system", "content": t("resumed_after_approval")})
        st.session_state.admin_chat.append({"role": "assistant", "response": resumed})

    prompt = None
    if not st.session_state.admin_chat:
        cols = st.columns(len(t("agents_examples")))
        for i, (col, example) in enumerate(zip(cols, t("agents_examples"))):
            if col.button(example, key=f"agent_example_{i}{key_suffix}", width="stretch"):
                prompt = example

    with st.form(f"agent_form{key_suffix}", clear_on_submit=True, border=False):
        typed = st.text_input(t("tab_agents"), placeholder=t("agents_placeholder"),
                              label_visibility="collapsed", key=f"agent_input{key_suffix}")
        if st.form_submit_button(t("send"), icon=":material/send:", type="primary",
                                 key=f"agent_send{key_suffix}"):
            prompt = typed or prompt

    if prompt and prompt.strip():
        with st.spinner(t("thinking")):
            response = admin_call("ask", prompt.strip(), context={
                "conversation_id": st.session_state.admin_conversation_id,
                "preferred_language": lang(),
            })
        st.session_state.admin_chat.append({"role": "user", "content": prompt.strip()})
        st.session_state.admin_chat.append({"role": "assistant", "response": response})
        st.rerun()   # redraw as a finished turn, not mid-spinner

    for index, message in enumerate(reversed(_pairs(st.session_state.admin_chat))):
        user, reply = message
        is_system = user.get("role") == "system"
        with st.chat_message(user.get("role", "user"), avatar=":material/notifications:" if is_system else ":material/person:"):
            st.markdown(plain(user["content"]))
        with st.chat_message("assistant", avatar=":material/hub:"):
            response = reply["response"] or {}
            results = response.get("result") or []
            if not results and response.get("error_message"):
                st.error(response["error_message"])
                continue
            if len(results) > 1:
                involved = ", ".join(AGENT_LABELS.get(r.get("agent_name"), {}).get(
                    lang(), r.get("agent_name")) for r in results)
                st.caption(f"{t('agents_involved')}: {involved}")
                if response.get("summary"):   # the orchestrator's own account of the whole request
                    st.markdown(plain(response["summary"]).replace("\n", "  \n"))
            failed = [r for r in results if r.get("status") not in ("success", "paused", "waiting_for_input")]
            if failed and len(failed) < len(results):
                st.warning(t("partial_failure"))
            for result in results:
                render_agent_result(result, key_suffix=f"{key_suffix}_{index}")


def _pairs(messages: list[dict]) -> list[tuple[dict, dict]]:
    return [(messages[i], messages[i + 1]) for i in range(0, len(messages) - 1, 2)]


def status_badges(tender: dict) -> str:
    submission, application = tender["submission_status"], tender["application_status"]
    return (badge(f"sub-{submission}", f"{t('submission')}: {t('sub_status')[submission]}")
            + badge(f"app-{application}", f"{t('application')}: {t('app_status')[application]}"))


def dashboard_tab() -> None:
    """Approved tenders, searchable (brief requirement 16), with their applications."""
    section_header(":material/dashboard:", "tab_dashboard")
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
    # The columns an admin scans for; every other detail is in each tender's own panel below.
    columns = ["title", "organization", "country", "deadline", "submission_status", "application_status",
               "submission_reference", "submission_url", "source_url"]
    st.dataframe(
        [{c: (row.get(c) if c != "source_url" else _web_link(row.get("source_url") or row.get("url")))
          for c in columns} for row in rows],
        hide_index=True, width="stretch",
        column_config={"submission_url": st.column_config.LinkColumn(t("open_submission")),
                       "source_url": st.column_config.LinkColumn(t("open_source"))},
    )
    for row in rows:
        with st.expander(plain(row.get("title")), expanded=row["opportunity_id"] == st.session_state.get("open_tender")):
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
            draft = admin_call("current_email_draft", opp_id)
            send_record = next(
                (s for s in history["email_sends"] if s["attempt_no"] == latest["attempt_no"]), None)

            mode_info = admin_call("email_sending_mode")
            # The address it will ACTUALLY go to, in every mode, enforced by the backend (email_sending.py).
            actual_to = mode_info["recipient"]
            with st.container(border=True):
                st.markdown(f"**{t('prepared_email')}**")
                if draft.get("is_demo_profile"):
                    st.warning(t("demo_profile_email_warning"))
                st.markdown(f"**{t('will_send_to')}:** {plain(actual_to or '—')} · "
                            f"**{t('from')}:** {plain(draft['from'] or '—')}")
                st.caption(f"{t('tender_contact')}: {plain(draft['to'] or t('no_verified_contact'))} "
                           f"({t('tender_contact_unused')})")
                st.markdown(f"**{t('attachments')}:** " + "; ".join(
                    plain(a["file_name"]) for a in draft["attachments"]))
                recommended = latest["confirmation"].get("recommended_documents_missing")
                if recommended:
                    st.caption(f"{t('recommended_missing')}: " + ", ".join(plain(r) for r in recommended))

                if send_record:
                    st.markdown(f"**{t('subject')}:** {plain(draft['subject'])}")
                    st.text(draft["body"])
                else:
                    if draft.get("edited"):
                        st.caption(t("draft_edited_note").format(
                            by=plain(draft["draft_updated_by"]), at=plain(draft["draft_updated_at"][:19])))
                    with st.form(f"edit_email_{opp_id}", border=False):
                        subject = st.text_input(t("subject"), value=draft["subject"],
                                                key=f"edit_subject_{opp_id}")
                        body = st.text_area(t("body"), value=draft["body"], height=260,
                                            key=f"edit_body_{opp_id}")
                        save_col, discard_col = st.columns(2)
                        if save_col.form_submit_button(t("save_draft_email"), key=f"save_draft_{opp_id}"):
                            if admin_action("save_email_draft", opp_id, subject, body):
                                st.session_state.flash = t("done")
                                st.rerun()
                        if draft.get("edited") and discard_col.form_submit_button(
                            t("discard_draft_email"), key=f"discard_draft_{opp_id}",
                        ):
                            if admin_action("discard_email_draft", opp_id):
                                st.session_state.flash = t("done")
                                st.rerun()

            if send_record:
                if send_record["status"] == "sent":
                    st.success(t("email_sent_summary").format(
                        to=plain(send_record["recipient"]), at=plain(send_record["created_at"][:19])))
                    if send_record["recipient"] != send_record.get("intended_recipient"):
                        st.caption(t("email_redirected_note").format(
                            real=plain(send_record["intended_recipient"])))
                else:
                    st.error(t("email_send_failed").format(error=plain(send_record.get("error") or "")))
            else:
                if not mode_info["recipient_configured"]:
                    st.error(t("recipient_not_configured"))
                elif not tender["submission_authorized"]:
                    st.info(t("authorise_to_send"))
                else:
                    with st.form(f"send_email_{opp_id}", border=True):
                        if mode_info["is_production"]:
                            st.warning(t("send_production_warning").format(recipient=plain(actual_to)))
                        else:
                            st.info(t("send_test_email_help").format(recipient=plain(actual_to)))
                        confirmed = st.checkbox(t("confirm_send"), key=f"confirm_send_{opp_id}")
                        if st.form_submit_button(t("send_now"), key=f"send_now_{opp_id}", type="primary"):
                            if not confirmed:
                                st.error(t("need_confirm"))
                            elif admin_action("send_application_email", opp_id):
                                st.session_state.flash = t("done")
                                st.rerun()

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
    section_header(":material/business:", "tab_profile")
    st.caption(t("profile_intro"))
    profile = admin_call("application_profile")
    labels = t("field_label")

    with st.form("profile_form", border=True):
        values = {}
        short_fields = [f for f in profile["field_labels"] if f not in LONG_PROFILE_FIELDS]
        long_fields = [f for f in profile["field_labels"] if f in LONG_PROFILE_FIELDS]
        columns = st.columns(2)
        for i, field in enumerate(short_fields):
            values[field] = columns[i % 2].text_input(
                labels.get(field, field), value=profile["fields"].get(field, ""), key=f"profile_{field}")
        for field in long_fields:
            values[field] = st.text_area(
                labels.get(field, field), value=profile["fields"].get(field, ""), key=f"profile_{field}",
                height=120)
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
    section_header(":material/school:", "tab_curricula")
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
        with st.expander(title, expanded=row["curriculum_id"] == st.session_state.get("open_curriculum")):
            curriculum_panel(row)


def curriculum_panel(row: dict, show_companion_toggle: bool = True, key_suffix: str = "") -> None:
    """
    key_suffix: only used for a bilingual companion shown side by side
    with the curriculum the admin actually opened -- that companion also
    has its OWN top-level expander elsewhere in this tab's listing, so
    its widgets here need different keys to avoid colliding with it.
    """
    curriculum_id = row["curriculum_id"]
    labels = t("cur_status")

    if show_companion_toggle:
        companion_id = admin_call("curriculum_translation", curriculum_id)
        if companion_id:
            st.caption(t("cur_bilingual_companion").format(id=companion_id))
            if st.checkbox(t("cur_show_side_by_side"), key=f"cur_side_by_side_{curriculum_id}"):
                companion_row = next((r for r in admin_call("list_curricula")
                                     if r["curriculum_id"] == companion_id), None)
                left, right = st.columns(2)
                with left:
                    curriculum_panel(row, show_companion_toggle=False)
                if companion_row:
                    with right:
                        curriculum_panel(companion_row, show_companion_toggle=False, key_suffix="_companion")
                return
    versions = [v["version"] for v in admin_call("get_curriculum", curriculum_id)["versions"]]
    version = st.selectbox(
        # The key includes the current version: after a revision the selector shows the new version
        # instead of staying on the one chosen before.
        t("cur_version"), list(reversed(versions)),
        key=f"cur_version_{curriculum_id}_{row['current_version']}{key_suffix}",
        format_func=lambda v: f"v{v}" + (f" ({t('cur_current')})" if v == row["current_version"] else ""))
    detail = admin_call("get_curriculum", curriculum_id, version)
    record = detail["curriculum"]
    content = record["content"]
    version_status = record.get("version_status") or record["status"]

    # The decision first: status, what (if anything) blocks approval, the review actions, the downloads.
    st.markdown(badge(f"cur-{version_status}", labels.get(version_status, version_status))
                + f" `{curriculum_id}` · v{version}", unsafe_allow_html=True)
    errors = (record.get("validation") or {}).get("errors") or []
    if errors:
        st.warning(t("cur_validation_bad") + "\n" + "\n".join(f"- {plain(e.get('message'))}" for e in errors))
    else:
        st.success(t("cur_validation_ok"))

    reviewable = version == row["current_version"] and record["status"] == "pending_review"
    if reviewable:
        with st.form(f"cur_review_{curriculum_id}{key_suffix}", border=True):
            notes = st.text_area(t("cur_notes"), key=f"cur_notes_{curriculum_id}{key_suffix}")
            approve, changes = st.columns(2)
            if approve.form_submit_button(t("cur_approve"), type="primary", icon=":material/check:",
                                          key=f"cur_approve_{curriculum_id}{key_suffix}"):
                if admin_action("approve_curriculum", curriculum_id, version, notes or None):
                    st.session_state.flash = t("cur_approved_msg")
                    st.rerun()
            if changes.form_submit_button(t("cur_request_changes"), icon=":material/edit_note:",
                                          key=f"cur_changes_{curriculum_id}{key_suffix}"):
                if admin_action("request_curriculum_changes", curriculum_id, version, notes):
                    st.session_state.flash = t("cur_changes_msg")
                    st.rerun()
    else:
        st.caption(t("cur_why_not_reviewable").get(version_status if version == row["current_version"] else "old",
                                                   t("cur_not_reviewable")))

    head = record if version == row["current_version"] else admin_call("get_curriculum", curriculum_id)["curriculum"]
    approved_pdf_button(curriculum_id, head, f"cur_pdf_{curriculum_id}{key_suffix}")
    companion = admin_call("curriculum_translation", curriculum_id)
    if companion:
        companion_head = admin_call("get_curriculum", companion)
        if companion_head:
            bilingual_zip_button(curriculum_id, [head, companion_head["curriculum"]],
                                 f"cur_zip_{curriculum_id}{key_suffix}")

    st.divider()
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

    alignment = content.get("alignment") or {}
    if alignment.get("items"):
        st.markdown(f"**{t('cur_alignment')}**")
        framework = alignment.get("framework") or {}
        st.caption(plain(f"{framework.get('name')} — {framework.get('publisher')}, {framework.get('version')}"))
        st.dataframe([{"Requirement": f"{i['requirement_id']}: {i.get('requirement') or ''}", "Result": i["status"],
                       "Evidence": ", ".join(i.get("element_ids") or []),
                       "Explanation": i.get("recommendation") or i.get("explanation")}
                      for i in alignment["items"]], hide_index=True, width="stretch")
    else:
        st.caption(t("cur_no_alignment"))

    references = content.get("references") or []
    if references:
        st.markdown(f"**{t('cur_references')}**")
        st.markdown("\n".join(f"- {plain(r.get('title'))} — {plain(r.get('url'))}" for r in references))
    else:
        st.caption(t("cur_no_references"))
    if content.get("assumptions"):
        st.markdown(f"**{t('cur_assumptions')}**")
        st.markdown("\n".join(f"- {plain(a)}" for a in content["assumptions"]))

    curriculum_edit_form(curriculum_id, row, version, content, key_suffix)

    if detail["audit"]:
        with st.expander(t("cur_audit"), key=f"cur_audit_{curriculum_id}{key_suffix}"):
            st.dataframe(list(reversed(detail["audit"])), hide_index=True, width="stretch")


def curriculum_edit_form(curriculum_id: str, row: dict, version: int, content: dict, key_suffix: str) -> None:
    """
    A structured editor (no raw JSON) for the current version's content,
    saved as a new version (agents/admin_service.py's edit_curriculum,
    agents/curriculum_research/editing.py's text <-> content
    conversion): the original and every earlier version stay exactly as
    they were, and the new version always needs its own approval, even
    when the version it was edited from was already approved.
    """
    editable = version == row["current_version"]
    with st.expander(t("cur_edit"), key=f"cur_edit_expander_{curriculum_id}{key_suffix}"):
        if not editable:
            st.caption(t("cur_edit_only_current"))
            return
        with st.form(f"cur_edit_{curriculum_id}{key_suffix}", border=False):
            title = st.text_input(t("cur_edit_title"), value=content.get("title", ""),
                                  key=f"cur_edit_title_{curriculum_id}{key_suffix}")
            overview = st.text_area(t("cur_edit_overview"), value=content.get("overview", ""),
                                    key=f"cur_edit_overview_{curriculum_id}{key_suffix}")
            audience = st.text_input(t("cur_edit_audience"), value=content.get("target_audience", ""),
                                     key=f"cur_edit_audience_{curriculum_id}{key_suffix}")
            prerequisites = st.text_area(
                t("cur_edit_prerequisites"), value=curriculum_editing.format_lines(content.get("prerequisites") or []),
                help=t("cur_edit_one_per_line"), key=f"cur_edit_prereq_{curriculum_id}{key_suffix}")
            objectives = st.text_area(
                t("cur_edit_objectives"), value=curriculum_editing.format_lines(content.get("objectives") or []),
                help=t("cur_edit_one_per_line"), key=f"cur_edit_objectives_{curriculum_id}{key_suffix}")
            outcomes = st.text_area(
                t("cur_edit_outcomes"), value=curriculum_editing.format_outcomes(content.get("learning_outcomes") or []),
                help=t("cur_edit_outcomes_help"), key=f"cur_edit_outcomes_{curriculum_id}{key_suffix}")
            modules = st.text_area(
                t("cur_edit_modules"), value=curriculum_editing.format_modules(content.get("modules") or []),
                height=260, help=t("cur_edit_modules_help"), key=f"cur_edit_modules_{curriculum_id}{key_suffix}")
            assessments = st.text_area(
                t("cur_edit_assessments"), value=curriculum_editing.format_assessments(content.get("assessments") or []),
                help=t("cur_edit_assessments_help"), key=f"cur_edit_assessments_{curriculum_id}{key_suffix}")
            capstone_current = curriculum_editing.capstone_fields(content.get("capstone"))
            include_capstone = st.checkbox(t("cur_edit_include_capstone"), value=bool(content.get("capstone")),
                                           key=f"cur_edit_has_capstone_{curriculum_id}{key_suffix}")
            capstone_title = st.text_input(t("cur_edit_capstone_title"), value=capstone_current["title"],
                                           key=f"cur_edit_capstone_title_{curriculum_id}{key_suffix}")
            capstone_description = st.text_area(t("cur_edit_capstone_description"), value=capstone_current["description"],
                                                key=f"cur_edit_capstone_desc_{curriculum_id}{key_suffix}")
            capstone_outcomes = st.text_input(t("cur_edit_capstone_outcomes"), value=capstone_current["outcome_ids"],
                                              key=f"cur_edit_capstone_outcomes_{curriculum_id}{key_suffix}")
            capstone_deliverables = st.text_area(
                t("cur_edit_capstone_deliverables"), value=capstone_current["deliverables"],
                help=t("cur_edit_one_per_line"), key=f"cur_edit_capstone_deliverables_{curriculum_id}{key_suffix}")
            tools_resources = st.text_area(
                t("cur_edit_tools"), value=curriculum_editing.format_lines(content.get("tools_resources") or []),
                help=t("cur_edit_one_per_line"), key=f"cur_edit_tools_{curriculum_id}{key_suffix}")
            assumptions = st.text_area(
                t("cur_edit_assumptions"), value=curriculum_editing.format_lines(content.get("assumptions") or []),
                help=t("cur_edit_one_per_line"), key=f"cur_edit_assumptions_{curriculum_id}{key_suffix}")
            if st.form_submit_button(t("cur_save_edit"), icon=":material/save:",
                                     key=f"cur_save_edit_{curriculum_id}{key_suffix}"):
                edits = {
                    "title": title.strip(), "overview": overview.strip(), "target_audience": audience.strip(),
                    "prerequisites": curriculum_editing.parse_lines(prerequisites),
                    "objectives": curriculum_editing.parse_lines(objectives),
                    "learning_outcomes": curriculum_editing.parse_outcomes(outcomes, []),
                    "modules": curriculum_editing.parse_modules(modules),
                    "assessments": curriculum_editing.parse_assessments(assessments),
                    "capstone": curriculum_editing.parse_capstone(
                        include_capstone, capstone_title, capstone_description, capstone_outcomes,
                        capstone_deliverables),
                    "tools_resources": curriculum_editing.parse_lines(tools_resources),
                    "assumptions": curriculum_editing.parse_lines(assumptions),
                }
                try:
                    saved = admin_call("edit_curriculum", curriculum_id, edits)
                except (ValueError, KeyError) as exc:
                    st.error(str(exc).strip("'\""))
                else:
                    # The new version is re-validated: say so when it is still a draft, never "pending review".
                    st.session_state.flash = t("cur_edit_saved_msg" if saved["status"] == "pending_review"
                                               else "cur_edit_saved_draft_msg")
                    st.rerun()


DOCUMENTS_PER_PAGE = 25


def documents_tab() -> None:
    """Agent 4's documents: generate from an approved curriculum, edit, approve, download."""
    section_header(":material/description:", "tab_documents")
    st.caption(t("doc_intro"))
    labels = t("doc_status")
    counts = admin_call("document_counts")
    for column, (status, label) in zip(st.columns(len(labels)), labels.items()):
        column.metric(label, counts.get(status, 0))

    approved, total_approved, list_error = [], 0, None
    try:
        page_of_curricula = admin_call("list_approved_curricula")
        approved, total_approved = page_of_curricula["curricula"], page_of_curricula["total"]
    except ValueError as exc:   # a failure is shown as one, never as "none approved"
        list_error = str(exc)
    with st.expander(t("doc_new"), expanded=not sum(counts.values())):
        if list_error:
            st.error(t("doc_list_error").format(error=plain(list_error)))
        elif not approved:
            st.info(t("doc_none_approved"))
        else:
            if total_approved > len(approved):
                st.caption(t("doc_showing").format(shown=len(approved), total=total_approved))
            # The two choices are outside the form so that choosing "Curriculum PDF" shows its download at once.
            curriculum = st.selectbox(
                t("doc_curriculum"), approved, key="doc_curriculum",
                format_func=lambda c: f"{plain(c['title'])} · {c['curriculum_id']} · v{c['approved_version']} · "
                                      f"{(c.get('language') or '').upper()}")
            doc_type = st.selectbox(t("doc_type"), [*TEMPLATES, CURRICULUM_PDF], key="doc_type",
                                    format_func=lambda d: type_label(d, lang()))
            st.caption(t("doc_language_note").format(
                language={"ar": "العربية" if lang() == "ar" else "Arabic", "en": "الإنجليزية" if lang() == "ar" else "English"}
                .get(curriculum.get("language"), curriculum.get("language") or "")))
            if doc_type == CURRICULUM_PDF:
                curriculum_pdf_download(curriculum)
            else:
                with st.form("doc_form", border=False):
                    extra = {"training_format": st.text_input(t("doc_format"), key="doc_extra_format",
                                                              max_chars=EXTRA_FIELDS["training_format"]),
                             "organization": st.text_input(t("doc_org"), key="doc_extra_org",
                                                           max_chars=EXTRA_FIELDS["organization"]),
                             "schedule_notes": st.text_input(t("doc_schedule"), key="doc_extra_schedule",
                                                             max_chars=EXTRA_FIELDS["schedule_notes"])}
                    submitted = st.form_submit_button(t("doc_generate"), type="primary", icon=":material/description:")
                if submitted:
                    _generate_document(curriculum["curriculum_id"], doc_type, {k: v for k, v in extra.items() if v})

    options = [None, *labels]
    status = st.selectbox(t("doc_filter"), options, key="doc_filter",
                          format_func=lambda s: t("doc_all") if s is None else labels[s])
    total = sum(counts.values()) if status is None else counts.get(status, 0)
    pages = max(1, -(-total // DOCUMENTS_PER_PAGE))
    page = 1
    if pages > 1:
        page = st.number_input(t("doc_page"), min_value=1, max_value=pages, value=1, step=1, key=f"doc_page_{status}")
    rows = admin_call("list_documents", status, limit=DOCUMENTS_PER_PAGE, offset=(int(page) - 1) * DOCUMENTS_PER_PAGE)
    if not rows:
        st.info(t("doc_none"))
        return
    for row in rows:
        approved_tag = (" · " + t("doc_approved_tag").format(n=row["approved_version"])
                        if row.get("approved_version") and row["status"] != "approved" else "")
        title = (f"{plain(row['title'])} · v{row['current_version']} · {row['curriculum_id']} · "
                 f"{labels.get(row['status'], row['status'])}{approved_tag}")
        with st.expander(title):
            document_panel(row)


def curriculum_pdf_download(curriculum: dict) -> None:
    """The "Curriculum PDF" choice: Agent 3's own PDF of the approved version. Read-only; nothing is generated or stored."""
    st.info(t("doc_cpdf_note"))
    try:
        pdf = admin_call("approved_curriculum_pdf", curriculum["curriculum_id"])
    except ValueError as exc:
        st.error(plain(str(exc)))
        return
    st.download_button(t("doc_cpdf_download").format(n=pdf["version"]), pdf["data"], file_name=pdf["filename"],
                       mime="application/pdf", icon=":material/download:", key=f"cpdf_{curriculum['curriculum_id']}_{pdf['version']}")


def _generate_document(curriculum_id: str, doc_type: str, extra: dict, document_id: str | None = None) -> None:
    """Ask the Document agent (document_id: a new version of that document) and show its answer."""
    # Writing can take several minutes (one model call per module): say so, and show each step as it happens.
    progress_box = st.empty()

    def progress(done: int, total: int, label: str) -> None:
        progress_box.info(t("doc_progress").format(label=plain(label), done=done + 1, total=total) if label
                          else t("doc_progress_saving"))

    with st.spinner(t("doc_generating")):
        response = admin_call("generate_document", curriculum_id, doc_type, None, extra, document_id,
                              context={"conversation_id": st.session_state.admin_conversation_id,
                                       "preferred_language": lang(), "progress": progress})
    progress_box.empty()
    results = response.get("result") or []
    if response.get("status") == "success" and results:
        # Redraw so the counts, the list and the version selector show the new version; the agent's
        # message is kept in the flash banner.
        st.session_state.flash = "  \n".join(plain((r.get("result") or {}).get("message")) for r in results
                                             if isinstance(r.get("result"), dict))
        st.rerun()
    for result in results:
        render_agent_result(result)
    if not results and response.get("error_message"):
        st.error(response["error_message"])


def _render_block(block: dict) -> None:
    kind = block.get("type")
    if kind == "paragraph":
        st.markdown(plain(block.get("text")))
    elif kind == "bullets":
        st.markdown("\n".join(f"- {plain(i)}" for i in block.get("items") or []))
    elif kind == "heading":
        st.markdown(f"**{plain(block.get('text'))}**")
    elif kind == "table":
        st.dataframe([dict(zip(block.get("header") or [], row)) for row in block.get("rows") or []],
                     hide_index=True, width="stretch")


def document_panel(row: dict) -> None:
    document_id = row["document_id"]
    labels = t("doc_status")
    versions = [v["version"] for v in admin_call("get_document", document_id)["versions"]]
    version = st.selectbox(
        t("doc_version"), list(reversed(versions)), key=f"doc_version_{document_id}_{row['current_version']}",
        format_func=lambda v: f"v{v}" + (f" ({t('doc_current')})" if v == row["current_version"] else ""))
    detail = admin_call("get_document", document_id, version)
    record = detail["document"]
    content = record["content"]
    version_status = record["version_status"]
    is_current = version == row["current_version"]

    st.markdown(badge(f"cur-{version_status}", labels.get(version_status, version_status))
                + f" `{document_id}` · `{content['curriculum_id']}` v{content['curriculum_version']}",
                unsafe_allow_html=True)
    if detail["curriculum_changed"]:
        st.info(t("doc_curriculum_changed"))
    if content.get("missing"):
        st.warning(t("doc_missing") + ": " + ("، " if lang() == "ar" else ", ").join(
            sorted({field_label(m["field"], lang()) for m in content["missing"]})))
    errors = (record.get("validation") or {}).get("errors") or []
    if errors:
        st.warning(t("doc_problems") + "\n" + "\n".join(f"- {plain(e.get('message'))}" for e in errors))

    edits_allowed = is_current   # an edit of an approved version is saved as a new version
    if edits_allowed and version_status == "approved":
        st.info(t("doc_edit_approved").format(n=version))
    container = st.form(f"doc_edit_{document_id}", border=True) if edits_allowed else st.container()
    pending_edits: list[tuple[dict, str]] = []
    with container:
        for section in content["sections"]:
            st.markdown(f"**{plain(section['heading'])}**")
            for index, block in enumerate(section["blocks"]):
                free = block.get("source") in ("generated", "admin") and block.get("type") in ("paragraph", "bullets")
                if free and edits_allowed:
                    current = block["text"] if block["type"] == "paragraph" else "\n".join(block.get("items") or [])
                    value = st.text_area(t("doc_edit_hint"), current,
                                         key=f"doc_{document_id}_{version}_{section['key']}_{index}",
                                         label_visibility="collapsed")
                    if value != current:
                        pending_edits.append(({"section": section["key"], "block": index}, value))
                else:
                    _render_block(block)
        if edits_allowed and st.form_submit_button(t("doc_save_edit"), icon=":material/save:",
                                                   key=f"doc_save_{document_id}"):
            if not pending_edits:
                st.info(t("doc_unchanged"))
            elif admin_action("edit_document", document_id,
                              [{**target, "value": value} for target, value in pending_edits]):
                st.session_state.flash = t("doc_edited_msg")
                st.rerun()

    if record["filename"]:
        pdf = admin_call("document_pdf", document_id, version)
        st.download_button(t("doc_approved_pdf") if pdf["approved"] else t("doc_review_pdf"), pdf["data"],
                           file_name=pdf["filename"], mime="application/pdf", icon=":material/download:",
                           key=f"doc_pdf_{document_id}_{version}")
    elif st.button(t("doc_pdf_create"), icon=":material/picture_as_pdf:", key=f"doc_pdf_make_{document_id}_{version}"):
        if admin_action("document_pdf", document_id, version):
            st.rerun()

    if not (is_current and version_status == "pending_review"):
        st.caption(t("doc_not_reviewable"))
    else:
        with st.form(f"doc_review_{document_id}", border=True):
            notes = st.text_area(t("doc_notes"), key=f"doc_notes_{document_id}")
            if st.form_submit_button(t("doc_approve"), type="primary", icon=":material/check:",
                                     key=f"doc_approve_{document_id}"):
                if admin_action("approve_document", document_id, version, notes or None):
                    st.session_state.flash = t("doc_approved_msg")
                    st.rerun()
    if is_current and st.button(t("doc_regenerate"), icon=":material/refresh:", key=f"doc_regen_{document_id}"):
        _generate_document(content["curriculum_id"], content["document_type"], content.get("extra") or {}, document_id)

    if detail["audit"]:
        with st.expander(t("doc_audit")):
            st.dataframe(list(reversed(detail["audit"])), hide_index=True, width="stretch")


CAMPAIGNS_PER_PAGE = 25


def marketing_tab() -> None:
    """Agent 5's campaigns: write from an approved course, review, edit, approve, download. Nothing is ever sent."""
    section_header(":material/campaign:", "tab_marketing")
    st.caption(t("mk_intro"))
    labels = t("mk_status")
    counts = admin_call("campaign_counts")
    for column, (status, label) in zip(st.columns(len(labels)), labels.items()):
        column.metric(label, counts.get(status, 0))

    approved, total_approved, list_error = [], 0, None
    try:
        page_of_courses = admin_call("list_approved_courses")
        approved, total_approved = page_of_courses["courses"], page_of_courses["total"]
    except ValueError as exc:   # a failure is shown as one, never as "none approved"
        list_error = str(exc)
    with st.expander(t("mk_new"), expanded=not sum(counts.values())):
        if list_error:
            st.error(t("mk_list_error").format(error=plain(list_error)))
        elif not approved:
            st.info(t("mk_none_approved"))
        else:
            if total_approved > len(approved):
                st.caption(t("mk_showing").format(shown=len(approved), total=total_approved))
            course = st.selectbox(
                t("mk_course"), approved, key="mk_course",
                format_func=lambda c: f"{plain(c['title'])} · {c['curriculum_id']} · v{c['approved_version']} · "
                                      f"{(c.get('language') or '').upper()}")
            st.caption(t("mk_language_note").format(
                language={"ar": "العربية" if lang() == "ar" else "Arabic", "en": "الإنجليزية" if lang() == "ar" else "English"}
                .get(course.get("language"), course.get("language") or "")))
            from_course = (course.get("target_audience") or "").strip()
            with st.form("mk_form", border=False):
                audience = st.text_input(t("mk_audience"), value=from_course, key=f"mk_audience_{course['curriculum_id']}",
                                         max_chars=300, help=t("mk_audience_from_course") if from_course else t("mk_audience_missing"))
                if not from_course:
                    st.caption(t("mk_audience_missing"))
                first, second, third = st.columns(3)
                tone = first.selectbox(t("mk_tone"), marketing_schemas.TONES, key="mk_tone",
                                       format_func=lambda x: marketing_schemas.TONE_LABELS[x][lang()])
                variations = second.selectbox(t("mk_variations"), [1, 2, 3], index=1, key="mk_variations")
                channels = third.multiselect(t("mk_channels"), marketing_schemas.CHANNELS, default=list(marketing_schemas.CHANNELS),
                                             key="mk_channels", format_func=lambda x: marketing_schemas.CHANNEL_LABELS[x][lang()])
                goal = st.text_input(t("mk_goal"), key="mk_goal", max_chars=300)
                notes = st.text_input(t("mk_notes"), key="mk_notes", max_chars=300)
                submitted = st.form_submit_button(t("mk_generate"), type="primary", icon=":material/campaign:")
            if submitted:
                if not channels:
                    st.warning(t("mk_need_channel"))
                else:
                    _generate_campaign({"curriculum_id": course["curriculum_id"], "audience": audience.strip() or None, "tone": tone,
                                        "variations": variations, "channels": list(channels), "goal": goal or None,
                                        "notes": notes or None})

    options = [None, *labels]
    status = st.selectbox(t("mk_filter"), options, key="mk_filter", format_func=lambda s: t("mk_all") if s is None else labels[s])
    total = sum(counts.values()) if status is None else counts.get(status, 0)
    pages = max(1, -(-total // CAMPAIGNS_PER_PAGE))
    page = 1
    if pages > 1:
        page = st.number_input(t("mk_page"), min_value=1, max_value=pages, value=1, step=1, key=f"mk_page_{status}")
    rows = admin_call("list_campaigns", status, limit=CAMPAIGNS_PER_PAGE, offset=(int(page) - 1) * CAMPAIGNS_PER_PAGE)
    if not rows:
        st.info(t("mk_none"))
        return
    for row in rows:
        approved_tag = (" · " + t("mk_approved_tag").format(n=row["approved_version"])
                        if row.get("approved_version") and row["status"] != "approved" else "")
        title = (f"{plain(row['title'])} · v{row['current_version']} · {row['curriculum_id']} · "
                 f"{labels.get(row['status'], row['status'])}{approved_tag}")
        with st.expander(title):
            campaign_panel(row)


def _generate_campaign(request: dict, campaign_id: str | None = None) -> None:
    """Ask the Marketing agent (campaign_id: a new version of that campaign) and show its answer."""
    progress_box = st.empty()

    def progress(done: int, total: int, label: str) -> None:
        progress_box.info(t("mk_progress").format(label=plain(label), done=done + 1, total=total) if label
                          else t("mk_progress_saving"))

    with st.spinner(t("mk_generating")):
        response = admin_call("generate_campaign", {**request, "campaign_id": campaign_id},
                              context={"conversation_id": st.session_state.admin_conversation_id,
                                       "preferred_language": lang(), "progress": progress})
    progress_box.empty()
    results = response.get("result") or []
    if response.get("status") == "success" and results and all((r.get("result") or {}).get("kind") == "campaign" for r in results):
        st.session_state.flash = "  \n".join(plain((r.get("result") or {}).get("message")) for r in results)
        st.rerun()
    for result in results:   # a question (audience? course?) or an error that says what failed and that nothing was saved
        render_agent_result(result)
    if not results and response.get("error_message"):
        st.error(response["error_message"])


def campaign_panel(row: dict) -> None:
    campaign_id = row["campaign_id"]
    labels = t("mk_status")
    versions = [v["version"] for v in admin_call("get_campaign", campaign_id)["versions"]]
    version = st.selectbox(
        t("mk_version"), list(reversed(versions)), key=f"mk_version_{campaign_id}_{row['current_version']}",
        format_func=lambda v: f"v{v}" + (f" ({t('mk_current')})" if v == row["current_version"] else ""))
    detail = admin_call("get_campaign", campaign_id, version)
    record = detail["campaign"]
    content = record["content"]
    version_status = record["version_status"]
    is_current = version == row["current_version"]
    language = content["language"]

    st.markdown(badge(f"cur-{version_status}", labels.get(version_status, version_status))
                + f" `{campaign_id}` · `{content['curriculum_id']}` v{content['curriculum_version']}", unsafe_allow_html=True)
    st.markdown(f"**{t('mk_audience_label')}:** {plain(content['audience']['text'])}  \n"
                f"**{t('mk_tone_label')}:** {marketing_schemas.TONE_LABELS[content['tone']][lang()]}")
    if detail["course_changed"]:
        st.info(t("mk_course_changed"))
    errors = (record.get("validation") or {}).get("errors") or []
    if errors:
        st.warning(t("mk_problems") + "\n" + "\n".join(f"- {plain(e.get('message'))}" for e in errors))

    edits_allowed = is_current   # an edit of an approved version is saved as a new version
    if edits_allowed and version_status == "approved":
        st.info(t("mk_edit_approved").format(n=version))
    container = st.form(f"mk_edit_{campaign_id}", border=True) if edits_allowed else st.container()
    pending_edits: list[tuple[list, str]] = []

    def field(label: str, path: list, current: str, multiline: bool = True) -> None:
        key = f"mk_{campaign_id}_{version}_" + "_".join(str(p) for p in path)
        if edits_allowed:
            value = (st.text_area if multiline else st.text_input)(label, current, key=key)
            if value != current:
                pending_edits.append((path, value))
        else:
            st.markdown(f"**{label}**  \n{plain(current)}")

    with container:
        strategy = content["strategy"]
        st.markdown(f"#### {t('mk_strategy')}")
        field(t("mk_objective"), ["strategy", "objective"], strategy["objective"])
        field(t("mk_main_message"), ["strategy", "main_message"], strategy["main_message"])
        st.markdown(f"**{t('mk_rec_channels')}:** " + ", ".join(marketing_schemas.CHANNEL_LABELS[c][lang()] for c in strategy["recommended_channels"]))
        for index, step in enumerate(strategy["content_sequence"]):
            field(t("mk_sequence").format(n=index + 1), ["strategy", "content_sequence", index], step)
        field(t("mk_cta"), ["strategy", "call_to_action"], strategy["call_to_action"], multiline=False)
        for channel, body in content["channels"].items():
            st.markdown(f"#### {marketing_schemas.CHANNEL_LABELS[channel][lang()]}")
            columns = st.columns(len(body["variations"]))     # the variations side by side, to compare
            for number, (column, variation) in enumerate(zip(columns, body["variations"])):
                with column:
                    st.markdown(f"**{t('mk_variation').format(n=number + 1)}**")
                    for name in marketing_schemas.LIMITS[channel]:
                        field(marketing_composer_label(name), ["channels", channel, "variations", number, name], variation[name])
                    if channel == "social_captions":
                        field(t("mk_hashtags"), ["channels", channel, "variations", number, "hashtags"],
                              " ".join(variation.get("hashtags") or []), multiline=False)
        if edits_allowed and st.form_submit_button(t("mk_save_edit"), icon=":material/save:", key=f"mk_save_{campaign_id}"):
            if not pending_edits:
                st.info(t("mk_unchanged"))
            else:
                try:
                    saved = admin_call("edit_campaign", campaign_id, [{"path": path, "value": value} for path, value in pending_edits])
                except ValueError as exc:
                    st.error(str(exc).strip("'\""))
                else:
                    st.session_state.flash = t("mk_edited_msg") if saved["review_status"] == "pending_review" else t("mk_edit_draft")
                    st.rerun()

    if record["approved_version"]:
        st.markdown(f"##### {t('mk_download_note').format(n=record['approved_version'])}")
        for fmt, label_key in (("markdown", "mk_download_md"), ("json", "mk_download_json")):
            export = admin_call("campaign_export", campaign_id, fmt)
            st.download_button(t(label_key), export["data"], file_name=export["filename"], mime=export["mime"],
                               icon=":material/download:", key=f"mk_dl_{fmt}_{campaign_id}_{export['version']}")
    else:
        st.caption(t("mk_nothing_to_download"))

    if not (is_current and version_status == "pending_review"):
        st.caption(t("mk_not_reviewable"))
    else:
        with st.form(f"mk_review_{campaign_id}", border=True):
            notes = st.text_area(t("mk_review_notes"), key=f"mk_notes_{campaign_id}")
            if st.form_submit_button(t("mk_approve"), type="primary", icon=":material/check:", key=f"mk_approve_{campaign_id}"):
                if admin_action("approve_campaign", campaign_id, version, notes or None):
                    st.session_state.flash = t("mk_approved_msg")
                    st.rerun()
    if is_current and st.button(t("mk_regenerate"), icon=":material/refresh:", key=f"mk_regen_{campaign_id}"):
        _generate_campaign({"curriculum_id": content["curriculum_id"], "audience": content["audience"]["text"], "tone": content["tone"],
                            "variations": content["variations"], "channels": list(content["channels"]), "goal": content.get("goal"),
                            "notes": content.get("notes")}, campaign_id)

    if detail["audit"]:
        with st.expander(t("mk_audit")):
            st.dataframe(list(reversed(detail["audit"])), hide_index=True, width="stretch")


def marketing_composer_label(name: str) -> str:
    """The label of one text field of a variation (subject, body, headline...), in the interface language."""
    from agents.marketing.composer import STRINGS

    return STRINGS[lang()].get(name, name.replace("_", " "))


def audit_tab() -> None:
    """Every audited decision across tenders, curricula, documents and campaigns, newest first."""
    section_header(":material/history:", "tab_audit")
    st.caption(t("audit_intro"))
    areas = t("audit_areas")
    area = st.segmented_control(t("audit_area"), [None, *areas], key="audit_area", label_visibility="collapsed",
                                format_func=lambda a: t("all") if a is None else areas[a])
    rows = [r for r in admin_call("activity_log") if area is None or r["area"] == area]
    if not rows:
        st.info(t("no_audit"))
    else:
        columns = t("audit_columns")
        st.dataframe([{columns[k]: (areas.get(v, v) if k == "area" else (v[:19].replace("T", " ") if k == "when" else v))
                       for k, v in row.items()} for row in rows], hide_index=True, width="stretch")
    trail = admin_call("audit_trail")
    if trail:
        with st.expander(t("audit_tender_detail")):
            st.dataframe(list(reversed(trail)), hide_index=True, width="stretch")


def floating_assistant() -> None:
    """
    The REX AI chatbot launcher: a popover, available from every section,
    that opens the EXACT SAME Ask Agents conversation (same orchestrator
    call, same session_state.admin_chat) in a floating panel. A popover
    is an overlay, not a navigation action, so opening or closing it
    never changes st.session_state.view or st.session_state.admin_section.
    """
    with st.popover(t("floating_chat_label"), icon=":material/smart_toy:", width="stretch"):
        agents_tab(key_suffix="_floating")


# The nine admin sections, in sidebar order: (key, icon, title key).
ADMIN_SECTIONS = [
    ("review", ":material/fact_check:", "tab_review"),
    ("search", ":material/travel_explore:", "tab_search"),
    ("agents", ":material/forum:", "tab_agents"),
    ("dashboard", ":material/dashboard:", "tab_dashboard"),
    ("curricula", ":material/school:", "tab_curricula"),
    ("documents", ":material/description:", "tab_documents"),
    ("marketing", ":material/campaign:", "tab_marketing"),
    ("profile", ":material/business:", "tab_profile"),
    ("audit", ":material/history:", "tab_audit"),
]


def _go_section(section: str) -> None:
    st.session_state.admin_section = section


def admin_sidebar() -> None:
    """
    The admin's navigation. The active section is server-side session state
    (admin_section), changed ONLY by these buttons or an explicit "open in"
    action -- never by an approval, a save, a download, the language switch
    or the floating assistant. (The nine sections used to be st.tabs, whose
    selected tab lived only in the browser: any message shown above them
    after an action moved them on the page, and the browser re-opened the
    first tab, Tender review.)
    """
    with st.sidebar:
        st.markdown("### REX AI")
        st.caption(t("dashboard"))
        for key, icon, title_key in ADMIN_SECTIONS:
            st.button(t(title_key), icon=icon, key=f"nav_{key}", width="stretch",
                      type="primary" if st.session_state.admin_section == key else "tertiary",
                      on_click=_go_section, args=(key,))
        st.divider()
        if st.button(t("sign_out"), icon=":material/logout:", key="sign_out", width="stretch"):
            get_admin_service().logout(st.session_state.admin_token)
            st.session_state.admin_token = None
            go_home()


def tender_metrics(counts: dict) -> None:
    metrics = st.columns(6)
    metrics[0].metric(t("pending"), counts["pending"])
    metrics[1].metric(t("flagged_pending"), counts["flagged_pending"])
    metrics[2].metric(t("approved"), counts["approved"])
    metrics[3].metric(t("metric_manual"), counts["needs_manual_action"])
    metrics[4].metric(t("metric_applied"), counts["applied"])
    metrics[5].metric(t("rejected"), counts["rejected"])


def _review_section() -> None:
    counts = admin_call("tender_counts")
    tender_metrics(counts)
    review_tab(counts)


def _dashboard_section() -> None:
    tender_metrics(admin_call("tender_counts"))
    dashboard_tab()


SECTION_VIEWS = {
    "review": _review_section, "search": lambda: search_tab(), "agents": lambda: agents_tab(),
    "dashboard": _dashboard_section, "curricula": lambda: curricula_tab(), "documents": lambda: documents_tab(),
    "marketing": lambda: marketing_tab(), "profile": lambda: profile_tab(), "audit": lambda: audit_tab(),
}


def admin_view() -> None:
    admin = get_admin_service()
    if not admin.is_valid(st.session_state.admin_token):
        st.session_state.admin_token = None
        go_home(t("session_expired"))

    section = st.session_state.admin_section if st.session_state.admin_section in SECTION_VIEWS else "review"
    admin_sidebar()
    top_bar()
    if section != "agents":   # in Ask agents the same chat is already the page itself
        _, launcher = st.columns([4, 1], vertical_alignment="center")
        with launcher:
            floating_assistant()

    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None
    alerts_banner()
    SECTION_VIEWS[section]()


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
