# REX AI — Manual Test Plan

Test cases for everything implemented so far: the RAG knowledge base, REX Assistant (Agent 1), RFP Research (Agent 2), the Curriculum Agent (Agent 3), Arabic/English chat, the User/Admin app, the tender and curriculum review workflows, and error handling.

- Fill in the **Result** column with ✅ pass, ❌ fail, or a note.
- The expected facts come from the **synthetic demonstration catalogue** (`knowledge_base/datasets/andalusia_catalogue/catalogue_source.json`). They are not real Andalusia Academy data.
- The AI model words answers differently each time. Check the **facts** (names, numbers, language, sources), not the exact wording.
- Technical background for every component: [PROJECT_TECHNICAL_GUIDE.md](PROJECT_TECHNICAL_GUIDE.md).

---

## 0. Setup (do once)

| # | Step | Expected | Result |
|---|---|---|---|
| 0.1 | `.\.venv\Scripts\Activate.ps1` | Prompt shows `(.venv)` | |
| 0.2 | `ollama list` | `qwen2.5:3b` is listed (if not: `ollama pull qwen2.5:3b`) | |
| 0.3 | Open `.env` in VS Code and check that `TAVILY_API_KEY=` and `REX_ADMIN_PASSCODE=` both have values. **Don't paste them anywhere.** | Both set | |
| 0.4 | `python -m streamlit run rex_ai_app.py` | Browser opens at `http://localhost:8501` | |

---

## A. Automated checks and the knowledge base (terminal)

Run these in a **second** terminal. Stop the app before A.4.

| # | Command | Expected | Result |
|---|---|---|---|
| A.1 | `python -m pytest -q` | `2796 passed` (no failures; the count grows as tests are added) | |
| A.2 | `python -m knowledge_base.evaluation --eval-set knowledge_base/datasets/andalusia_catalogue/eval_set.json` | `"recall_at_k": 1.0` and `"mrr"` around 0.98; `ar`, `en` and `mixed` all have recall 1.0 | |
| A.3 | Same command; look at `"items"` | Every item has a `"rank"` (none `null`) | |
| A.4 | **Rebuild test in a temp folder** (the real index and metadata are not touched), as one line: `python -c "from knowledge_base.pipeline import ingest_directory; from knowledge_base.vector_store import VectorStore; import tempfile, os; d = os.path.join(tempfile.gettempdir(), 'rex_test_index'); s = ingest_directory(vector_store=VectorStore(d), metadata_path=os.path.join(d, 'meta.json'), manifest_path=os.path.join(d, 'manifest.json')); print({k: s[k] for k in ('files_discovered', 'files_processed', 'chunks_stored', 'chunks_deleted')}, s['files_failed'])"` | `files_discovered 10`, `files_processed 10`, `chunks_stored 98`, `chunks_deleted 0`, failures `[]` | |
| A.5 | Run A.4 again | `chunks_stored 98` and `chunks_deleted 98` (old copies replaced, no duplicates) | |
| A.6 | `python -m knowledge_base.datasets.andalusia_catalogue.build_documents --out $env:TEMP\rex_docs` and open one PDF | 5 catalogue PDFs + `instructors.csv`, `schedules_and_fees.xlsx`, `faqs.json`, `admission_requirements.md`, `course_list.md`. Every PDF page says "Synthetic demonstration data" | |

---

## B. REX: direct course and academy questions (English)

Welcome screen → **I'm a learner**. Click **New conversation** between groups. Open the **Sources** dropdown under each answer.

| # | Message | Expected facts | Result |
|---|---|---|---|
| B.1 | `How much does the Flutter Mobile App Development course cost?` | **7000 EGP**; source `schedules_and_fees.xlsx` or `catalogue_mob_mobile_development.pdf` | |
| B.2 | `Is it online?` (follow-up, same chat) | **Online** (keeps the Flutter topic) | |
| B.3 | `When does it start?` | **2026-11-17**, Sunday and Wednesday 7–10 pm | |
| B.4 | `Who teaches Building LLM Applications with RAG?` | **Omar Haddad** | |
| B.5 | `What are the entry requirements for Machine Learning Practitioner?` | Comfortable with **Python and basic statistics**; AI-101 or equivalent | |
| B.6 | `Is the Excel for Business Analysis course online or on campus?` | **Offline (on campus, Cairo)** | |
| B.7 | `When does the SQL and Power BI course start?` | **2026-11-22** | |
| B.8 | `What are the prerequisites for Backend APIs with Python and Django?` | **SW-201** Full Stack Web Development with React and Node.js | |
| B.9 | `Which courses do you offer?` | All **14** courses, one line each (name, delivery, fee); sources `course_list.md` | |
| B.10 | `What is the refund policy?` | Full refund up to **7 days** before start; **50%** until the end of the first week | |
| B.11 | `Do I get a certificate?` | Yes, with **80%** attendance and passing the final project; certificate of completion | |
| B.12 | `Are your courses accredited?` | **No**, not officially accredited (must never claim accreditation) | |
| B.13 | `Is there a student discount?` | **STUDENT20**: 20% off, beginner courses, needs a student ID | |
| B.14 | `What documents do I need to register?` | Copy of **national ID** or passport, photo | |
| B.15 | `How experienced is Karim Nabil?` | **13 years** | |
| B.16 | `What happens if I miss an online session?` | Sessions are **recorded** (available 30 days) | |
| B.17 | `How much does it cost?` (new chat, no course mentioned) | Lists matching courses and/or asks which one you mean; **no invented price** | |

## C. REX: grounding and unsupported questions

| # | Message | Expected | Result |
|---|---|---|---|
| C.1 | `Do you have a course in quantum underwater basket weaving?` | "I couldn't find relevant information in the available documents." No invented course | |
| C.2 | `What is the salary of a pilot?` | Says the documents don't cover it; no general-knowledge answer | |
| C.3 | `How much is a pizza` | "Couldn't find" message; no discovery questions start | |
| C.4 | `Hello` / `Thanks!` | Greeting only; no search, no discovery questions | |
| C.5 | Any answer in B: compare the **[n]** citations with the Sources list | Each cited number exists in Sources; no file paths shown in the answer | |

## D. REX: Arabic, Egyptian Arabic, mixed and Arabizi

| # | Message | Expected | Result |
|---|---|---|---|
| D.1 | `كم سعر كورس فلاتر؟` | Reply **in Arabic**; **7000** جنيه; Flutter Mobile App Development | |
| D.2 | `وهل هو أونلاين؟` (follow-up) | Arabic reply: online (أونلاين) | |
| D.3 | `ما هي سياسة استرداد الرسوم؟` | Arabic: full refund 7 days before start, 50% in the first week | |
| D.4 | `هل يوجد خصم للطلاب؟` | Arabic: STUDENT20, 20% | |
| D.5 | `هل الدورات معتمدة؟` | Arabic: not accredited | |
| D.6 | `ما هي المستندات المطلوبة للتسجيل؟` | Arabic: national ID / passport copy, photo | |
| D.7 | `إيه اللي بيغطيه كورس تحليل البيانات؟` (Egyptian) | Arabic answer about a data course (SQL and Power BI or Python for Data Analysis); may ask which one | |
| D.8 | `إمتى يبدأ كورس SQL and Power BI؟` (mixed) | Arabic: **2026-11-22** | |
| D.9 | `مين المدرب في كورس Building LLM Applications with RAG؟` (mixed) | Arabic: **Omar Haddad** | |
| D.10 | `كام سعر Advanced Flutter with Firebase؟` (mixed) | Arabic: **9500** | |
| D.11 | `Ana 3ayez akhod course fel programming` (Arabizi) | Reply in **Arabic**; starts course discovery | |
| D.12 | Ask D.1, then reply `ok` | Still replies **in Arabic** (language stays) | |
| D.13 | Ask in English, then ask in Arabic in the same chat | Each reply follows the language you wrote in | |
| D.14 | `answer in English please` after an Arabic question | Switches to English | |

Known limitation: for Arabic course lists the 3B model may translate course names and skip some courses (English lists are complete). Arabic wording can also be odd (e.g. a strange phrase for "online"), and citations are sometimes missing in Arabic. Record it, but check the **facts**.

## E. REX: personalised recommendation (discovery)

New conversation. Send one message at a time.

| # | Message | Expected | Result |
|---|---|---|---|
| E.1 | `Help me choose a course` | Intro ("skip" / "recommend now"), then: **background** question | |
| E.2 | `I'm an accountant` | Asks **career path** | |
| E.3 | `marketing` | Asks **skill level** (must not skip it) | |
| E.4 | `begginner` (typo) | "Sorry, I didn't quite get that", then asks skill level again | |
| E.5 | `beginner` | Asks **online or offline** | |
| E.6 | `online` | Asks **budget** | |
| E.7 | `5000` | Asks **available time** | |
| E.8 | `2 hours per day` | Recommendation: **Digital Marketing Fundamentals** (MKT-101) (Beginner, Online, **4500 EGP**); each course shows its catalogue ID and lists Why it fits you / Who it is for / Level / Duration / Delivery / Fee / Entry requirements / Source; then asks to save your details | |
| E.9 | Check E.8's facts against the catalogue (`knowledge_base/documents/course_list.md`) | Every course ID and name exists exactly as in the catalogue; no course listed twice; no prerequisite the catalogue doesn't list; a fact the catalogue doesn't state says "not stated in the catalogue"; a course above beginner level is left out when a beginner course fits, or marked as above your level with its entry requirements | |
| E.9b | New conversation: `Help me choose a course`, then `Computer Engineer`, `Data Analytics`, `Beginner`, `Online or offline`, `4000`, `3 days per week` | Recommends **Excel for Business Analysis** (DA-101): Offline (on campus, Cairo), 3200 EGP within your budget of 4000. Never "Basic Computer Literacy" or a "Data-led Labs" course. "Online or offline" counts as no preference | |
| E.10 | `no` | "Nothing was saved" | |

**Variations** (new conversation each time):

| # | Steps | Expected | Result |
|---|---|---|---|
| E.11 | `Recommend a course`, then `skip` | Background recorded as skipped; next question asked | |
| E.12 | `Recommend a course`, then `what do you mean?` | Asks again; doesn't store it as your background | |
| E.13 | `Recommend a course`, then `yes`, then `no` | After 2 unclear replies it moves on to the next question | |
| E.14 | `Recommend a course`, then `I'm a beginner, I prefer online, budget 5000 EGP` | Stores all three; asks only the missing ones | |
| E.15 | `Recommend a course`, then `I want to become a data analyst` | Stores the goal; still **asks skill level and online/offline** (doesn't invent them) | |
| E.16 | `Recommend a course`, then `recommend now` | Recommends right away and lists the missing details | |
| E.17 | Mid-discovery: `What are the prerequisites of the Flutter Mobile App Development course?` | Answers the question, then repeats the pending discovery question | |
| E.18 | Arabic: `رشحلي كورس مناسب ليا` and answer in Arabic | All questions and the recommendation in Arabic | |
| E.19 | Arabic button: switch the UI to العربية, New conversation, click the example `ساعدني أختار دورة` | Starts discovery (not a greeting) | |

## E2. REX: the three modes over RAG/MCP (English and Arabic)

Which catalogue you see depends on the index the app uses (`REX_VECTOR_DB_DIR`, default `knowledge_base/vector_db/`). The expected results below are for the catalogue in `knowledge_base/documents/` (14 courses, see `course_list.md`). Open **Sources** under each answer to see the file, page, course, chunk, scores and retrieved text.

| # | Message (new conversation each time unless noted) | Expected | Result |
|---|---|---|---|
| E2.1 | `What courses does the academy offer?` | "The academy catalogue lists 14 courses:", numbered 1-14 by department. Every course appears once with its exact name and ID; facts not in the catalogue say "not stated in the catalogue" | |
| E2.2 | `What Data Analytics courses does the academy offer?` | Exactly DA-101, DA-201, DA-301 ("3 of the catalogue's 14 courses…") | |
| E2.3 | `What beginner courses do you offer?` | Only the Beginner courses (AI-101, DA-101, MKT-101, MOB-101, SW-101) | |
| E2.4 | `ما هي جميع الدورات التي تقدمها الأكاديمية؟` | Arabic heading "يضم كتالوج الأكاديمية 14 دورة:" and all 14 courses with their English catalogue names | |
| E2.5 | `Tell me more about SQL and Power BI for Data Analysts.` then `What are its prerequisites?`, `How long does it take?`, `What level is it?`, `What are the fees?` | Every answer is about DA-201: prerequisite DA-101, 48 hours / 8 weeks, Intermediate, 7800 EGP. No question needs the course name again | |
| E2.6 | `ما هي متطلبات الالتحاق بدورة Excel for Business Analysis؟` then `كم عدد ساعاتها؟` | Arabic answers about DA-101: basic computer skills; 24 hours | |
| E2.7 | Same conversation as E2.1, then `Recommend beginner courses for me` and the profile (`Computer Engineer`, `Data Analytics`, `Online or offline`, `4000`, `3 days per week`), then `no`, then `Tell me more about the first course` | The recommendation (e.g. DA-101) is followed by details of the **first recommended** course | |
| E2.8 | `I am a beginner interested in data analytics. Recommend suitable courses.` and the same profile | Only catalogue courses with catalogue facts; DA-101 3200 EGP "within your budget of 4000"; nothing above beginner level when a beginner course fits | |
| E2.9 | `أنا مبتدئ ومهتم بتحليل البيانات. رشح لي دورات مناسبة.` and the profile in Arabic (`مهندس كمبيوتر`, `تحليل البيانات`, `أونلاين أو حضوري`, `4000`, `3 أيام في الأسبوع`) | REX asks for your background (it does not take "مبتدئ" as one); the same grounding rules as E2.8, labels in Arabic | |
| E2.10 | `Tell me more about the Kubernetes Administration course.` | No invented course: REX says the catalogue does not cover it | |

**Real execution trace (terminal).** With Ollama running, `$env:REX_REAL_MODEL_TESTS = "1"; python -m pytest tests/test_rex_real_mcp_e2e.py -s` ingests the catalogue into a temporary index and runs REX through the real RAG MCP server subprocess, in English and Arabic.

## F. REX: consent and lead saving

| # | Steps | Expected | Result |
|---|---|---|---|
| F.1 | After a recommendation, reply `yes` | "Saved" message. `agents/rex_assistant/leads.jsonl` gets one new line with **profile fields only** (no chat text) | |
| F.2 | After a recommendation, reply `maybe later` | Asks again for yes/no; **nothing saved** | |
| F.3 | After a recommendation, ask `What topics does Full Stack Web Development cover?` | Answers the question (no "yes or no" nagging); nothing saved | |
| F.4 | After a recommendation, reply `no` | "Nothing saved"; `leads.jsonl` unchanged | |

## G. User interface

| # | Steps | Expected | Result |
|---|---|---|---|
| G.1 | Open the app | Welcome banner, two cards (learner / admin), English/العربية switch | |
| G.2 | Click **العربية** | All UI text in Arabic; layout right-to-left | |
| G.3 | Arabic UI, write an English question | REX answers in **English** (follows what you write) | |
| G.4 | Click **English** | Back to left-to-right English | |
| G.5 | **New conversation** | Chat cleared; previous profile forgotten | |
| G.6 | **Home** button | Returns to the welcome screen | |
| G.7 | Sources dropdown | Shows `[n] file name, p. page`; never a full disk path | |

---

## H. Admin access and security

| # | Steps | Expected | Result |
|---|---|---|---|
| H.1 | Empty `REX_ADMIN_PASSCODE` in `.env`, restart, click **I'm an admin** | Warning: admin access disabled; no passcode box. **Restore the passcode and restart afterwards.** | |
| H.2 | **I'm an admin**, then a wrong passcode | "Passcode is not correct"; stays on welcome | |
| H.3 | 5 wrong passcodes in a row | "Too many attempts"; even the correct passcode fails for ~60 s | |
| H.4 | Wait 60 s, then the correct passcode | Dashboard opens: 4 counters, 4 tabs | |
| H.5 | **Sign out**, then **I'm an admin** | Asks for the passcode again | |
| H.6 | Log in, idle for 30+ minutes, then click something | "Your admin session has ended" → welcome | |

## I. Agent 2: RFP research (Admin → **New tender search** and **Ask agents** tabs)

Needs internet and `TAVILY_API_KEY`. Each search takes about 1–2 minutes.

| # | Steps | Expected | Result |
|---|---|---|---|
| I.1 | **New tender search**, both countries, no other choices → **Search tenders** | Label **RFP Research**; "Found N relevant tender(s): N new, waiting for admin review (M Egypt tender(s) flagged…)"; a **Sources** line with World Bank, Etimad **unavailable**, and 4 web searches per country; the note that Saudi tenders come from web research | |
| I.2 | Expand a tender in the result | Badges (Egypt · immediate review or Saudi Arabia · monitored), organisation, country, category, deadline, qualification, AI summary, source / submission links; "Not stated by the source" lists the empty fields | |
| I.3 | Search only **Saudi Arabia** | Saudi tenders found through web research (not "No tender data could be retrieved") | |
| I.4 | Choose category **Corporate Learning** only | Only tenders labelled Corporate Learning (often none) | |
| I.5 | Set **Deadline to** = next week | Only tenders whose deadline is before then (or unknown) | |
| I.6 | Priority `Ministry of Education` | Matching tenders get the **Priority match** badge | |
| I.7 | Repeat I.1 | Mostly "already known": no duplicates added to the queue | |
| I.8 | **Ask agents**: `ابحث عن مناقصات تدريب في مصر` | Reply **in Arabic**; tenders saved as pending | |
| I.9 | **Ask agents**: `Find training tenders in Germany` | **Refused**: only Egypt and Saudi Arabia | |
| I.10 | **Ask agents**: `Which course do you recommend for me?` | **Refused**: course questions belong to REX in the User interface | |
| I.11 | **Ask agents**: `Create a curriculum about data analysis` | Label **Curriculum Research** with a curriculum draft (teammate's agent) | |

## J. Tender review, dashboard and alerts

| # | Steps | Expected | Result |
|---|---|---|---|
| J.1 | After I.1, look at the top of the dashboard | Alert "N new tender(s) found since you last checked (M in Egypt…)"; counters Pending / Egypt: review now / Approved / Needs manual action / Applied / Rejected | |
| J.2 | **Mark as seen** | The alert disappears until a search finds new tenders | |
| J.3 | **Tender review**, filter **Pending** | Egypt-flagged tenders listed first; country, category and keyword filters work | |
| J.4 | **Source page** / **Submission page** | Open the real web pages | |
| J.5 | **Approve** one tender (confirm in the dialog), **Reject** another | Counters update; each moves to its filter | |
| J.6 | **Tender dashboard** | Only the approved tender; filters work; links clickable | |
| J.7 | **Back to pending** on the rejected one | It returns to Pending | |
| J.8 | **Audit trail** tab | Every decision with time and `admin-session-…`; never the passcode or token | |
| J.9 | User side: ask REX about an approved tender | REX does **not** know it (tenders never enter the course knowledge base) | |

### J2. Applications after approval

**These steps write real records** to `agents/rfp_research/rfp_opportunities.sqlite3` (decisions, attempts, a manual submission). To test without keeping them, stop the app first and copy the file aside, then put it back afterwards. Preparing a draft never sends anything; sending is a separate, confirmed action (J.23 to J.27).

| # | Steps | Expected | Result |
|---|---|---|---|
| J.10 | **Application profile** tab: enter a legal name and an invalid email → **Save details** | Red error "not a valid email address"; nothing saved | |
| J.11 | Fix the email, add a contact person → **Save details**; upload a PDF as *Company profile* | "Saved."; the document is listed with its size | |
| J.12 | Upload a `.txt` renamed to `.pdf` | Refused: "not a valid .pdf file" | |
| J.13 | **Tender review** → **Approve** on an Etimad or listing-site tender | Dialog: "How to apply" with the steps (Etimad: Nafath, authorised signer); the checkbox is **unticked** | |
| J.14 | **Approve** without ticking | "Approved. Application: Needs manual action."; in the dashboard: *Submission not authorised*, the numbered steps, the Record manual submission form | |
| J.15 | On a World Bank "Request for Expression of Interest" tender with a contact email (if one is found), approve **with** the box ticked | Dashboard shows "Prepared email (dry run: NOT sent)" with To (the API address), From (your profile email), subject, body and attachments; status *Needs manual action*, application *Not applied* | |
| J.16 | Same, with the profile incomplete | "Missing: …" lists exactly what is missing; no email prepared | |
| J.17 | **Record manual submission** without the confirmation tick | Red "Tick the confirmation first."; still *Not applied* | |
| J.18 | Tick it, enter a reference → **Record** | Submission *Submitted*, application *Applied*; the reference in the table; the **Applied** counter goes up | |
| J.19 | Try **Reject** on that tender (Tender review → Approved) | Refused with a red message; it stays approved | |
| J.20 | **Outcome** → *Awarded* → **Record outcome**; then **Correct back to Applied** | Application *Awarded*, then *Applied* again | |
| J.21 | **History** in the tender panel; **Audit trail** tab | Attempts (adapter, dry run, triggered by `admin-session-…`, actor `system:application_workflow`) and events `review` / `application` | |
| J.22 | Switch to Arabic | Tabs and statuses in Arabic (e.g. ملف التقديم, لم يتم التقدم) | |

**Sending an application email.** Before these steps set `REX_EMAIL_TEST_RECIPIENT` in `.env` to an inbox **you** control, and restart the app. Never use a tender's address. To test without sending at all, leave `RESEND_API_KEY` empty: every send is then refused (J.27).

| # | Steps | Expected | Result |
|---|---|---|---|
| J.23 | Approve a tender with the authorisation box ticked; **Tender review** → *Approved* → **Review application email** | The Tender dashboard opens the prepared draft: subject and body editable, "Will be sent to: <your REX_EMAIL_TEST_RECIPIENT>", and "Tender's own contact: … (never used: every email goes to the configured recipient)" | |
| J.24 | Edit the body → **Save draft** | The edit is kept after the page redraws | |
| J.25 | **Send** without ticking the confirmation | Refused: tick the confirmation first; nothing sent | |
| J.26 | Tick the confirmation → **Send** (with a Resend key) | "Email sent" to your configured address only; the send history shows that same address as the recipient and the tender's contact only as reference | |
| J.27 | Empty `RESEND_API_KEY` (or `REX_EMAIL_TEST_RECIPIENT`, or set it to `not-an-email`), restart, repeat J.26 | Red error naming the missing or invalid setting; nothing sent. Same in `REX_EMAIL_MODE=production`: the recipient never changes to the tender's contact | |

---

## J3. Agent 3: Curriculum Agent (Admin → **Ask agents** and **Curricula** tabs)

Use one chat conversation for J3.1–J3.8 (don't reload the page in between). Each generation takes about 20–60 seconds with `qwen2.5:3b`. Content wording varies; check the structure, numbers, language and statuses.

| # | Step | Expected | Result |
|---|---|---|---|
| J3.1 | Ask agents: `Create an intermediate curriculum about SQL for business analysts, 24 hours` | Reply "Curriculum drafted" with an id `CUR-…`, version 1, status **pending review** (or **draft** with the validation problems listed), Level Intermediate, total 24 hours, a module list whose hours add up to 24, a PDF file name, and "only an administrator can approve" | |
| J3.2 | Same chat: `make it 30 hours` | "Curriculum revised", **version 2**, total 30 hours, module hours rescaled | |
| J3.3 | Same chat: `add a module on query optimisation` | Version 3, one more module, total still 30 hours | |
| J3.4 | Same chat: `approve it` | An error saying only an administrator can approve, in the Curricula tab. The status is unchanged | |
| J3.5 | Same chat: `Align it with the NICE framework` | "unable to determine": no approved framework document is loaded. No accreditation claim | |
| J3.6 | New chat (reload the page): `Create a curriculum about Kubernetes` | One question asking for **total hours and level**, and nothing else; the topic is shown as known | |
| J3.7 | Same chat: `40 hours, beginner` | The curriculum is drafted, with no further question | |
| J3.8 | Same chat: `translate it to Arabic` | A new version written in Arabic (the reply stays in English) | |
| J3.9 | Switch the interface to Arabic, new chat: `أنشئ منهج عن الأمن السيبراني للموظفين لمدة 20 ساعة للمبتدئين` | An Arabic reply "تم إعداد مسودة المنهج" with the id, version and status | |
| J3.10 | New chat: `Suggest new courses for the Data Analytics department` | 1–5 suggestions with a basis (catalogue gap ids or "model proposal"), confidence, possible overlaps, and a note that the catalogue data is **synthetic** and the gaps are not verified needs | |
| J3.11 | Same chat: `generate suggestion 1` | A curriculum drafted from that suggestion | |
| J3.12 | Ask agents: `Recommend new courses for me` | The admin side explains that course recommendations are for REX in the User interface; Agent 3 does not answer | |
| J3.13 | **Curricula** tab | Status counts; each curriculum as an expander showing its title, version, hours, language and status | |
| J3.14 | Open the SQL curriculum; pick version 1, then the current version | Version 1 shows 24 hours and the current one 30 hours; the sections, validation result and references are shown for each | |
| J3.15 | **Download PDF** for the current version and open it | A professional PDF: cover with the id, version and status "Pending review (not approved)", module and assessment tables, references, disclaimer, page numbers | |
| J3.16 | Open the Arabic curriculum's PDF | Arabic letters joined, lines reading right to left, tables starting from the right, and Latin terms (for example Python) readable | |
| J3.17 | On the current SQL version: **Request changes** with an empty note | An error "Say what should change."; the status is unchanged | |
| J3.18 | Write a note, then **Request changes** | Status **changes requested**; the review history shows your admin session | |
| J3.19 | In chat (the SQL conversation, or a new one with `show CUR-…`): `make it 32 hours`, then in the Curricula tab **Approve** the new version | The new version is pending review, then **approved**; the history shows the approval by the admin session | |
| J3.20 | A draft curriculum (with validation problems) | No Approve / Request changes buttons; the problems are listed | |

---

## J4. Agent 4: Document Generation (Admin → **Documents** tab and **Ask agents**)

**Status of this section:** these are manual steps to run in a real browser with a running Ollama model. The automated tests (Section L) cover the same behaviour with fake models in Streamlit `AppTest`; **no browser run of this section has been recorded yet**, so the Result column is empty. Prerequisite: at least one **approved** curriculum (J3, for example the SQL one; approve it in the Curricula tab). Each document takes from about a minute to many minutes on `qwen2.5:3b`; the tab shows each step.

| # | Step | Expected | Result |
|---|---|---|---|
| J4.1 | Open the **Documents** tab with no approved curriculum (fresh database) | "No curriculum is approved yet. Approve one in the Curricula tab first." and no Generate button | |
| J4.2 | With an approved curriculum: open **Documents** | The approved curriculum is offered (a pending or draft one is not); the language note says the document is written in the curriculum's language | |
| J4.3 | Choose **Course description** → **Generate document** | Progress steps, then the document: status **Pending review**, the text sections, a validation area with no problems | |
| J4.4 | Download the review PDF | A PDF labelled as a review copy, not final | |
| J4.5 | Edit one sentence → **Save edits as a new version** | Version 2, pending review; version 1 is still listed | |
| J4.6 | **Approve this version** | "Document approved."; the approved PDF is now available (**Download approved PDF**) | |
| J4.7 | Edit the approved document and save | A new version needing review; the approved version stays the approved one and its PDF is unchanged | |
| J4.8 | Repeat J4.3 for **Course proposal**, **Participant guide**, **Instructor guide**, **Question bank**, **Assessment rubric**, **Presentation outline**, **Capstone brief**, **Outcomes assessment plan** (at least three of them) | Each is created in the curriculum's language, pending review, with a PDF; a type that fails validation is not saved and the error names the type | |
| J4.9 | In **Participant guide**, check the hours wording | Hours come from the curriculum; nothing like a different total | |
| J4.10 | Choose **Curriculum PDF** | The approved curriculum's own PDF is offered for download; nothing is generated by the model | |
| J4.11 | Ask agents: `Create a course description for curriculum CUR-…` (your id) | Routed to Document Generation only (reply "Document created" or equivalent); no new curriculum is made | |
| J4.12 | Ask agents: `Create a curriculum about marketing analytics.` | Routed to the Curriculum Agent only; no document and no campaign | |
| J4.13 | Ask agents: `Create a curriculum about data analysis and prepare a course proposal.` | Curriculum first, then a document request; the proposal needs an **approved** curriculum, so it asks for one or refuses, and does not use the unapproved draft | |
| J4.14 | Switch to Arabic and generate a course description for an Arabic curriculum | The document is Arabic with joined letters and right-to-left text in the PDF (open it). Arabic question banks may fail on the small model; that is a known limitation | |
| J4.15 | Revise the curriculum (J3.2), then look at the older document | A note says the curriculum was revised since the document was built | |
| J4.16 | Look for any e-mail, publish or send button | There is none: documents are only downloaded | |

---

## J5. Agent 5: Marketing Agent (Admin → **Marketing** tab and **Ask agents**)

**Status of this section:** manual steps for a real browser and Ollama; the automated tests (Section L) cover them with fake models. **No browser run of this section has been recorded yet.** Prerequisite: one approved curriculum (J3). A full campaign makes about eight model calls and can take many minutes on `qwen2.5:3b`.

| # | Step | Expected | Result |
|---|---|---|---|
| J5.1 | Open **Marketing** with no approved course | "No course is approved yet…" and no **Create campaign** button | |
| J5.2 | With an approved course | Only approved courses are offered; the audience is filled from the course ("Taken from the approved course…"); tone *professional*, 2 variations, all seven content types selected; the language note says the campaign is not translated | |
| J5.3 | Click **Create campaign** (English course) | Progress steps; then a campaign **Pending review** with the strategy (objective, main message, channels, sequence, call to action) and the variations side by side | |
| J5.4 | Read the text for invented facts | No prices, discounts, free offers, guarantees, certificates, deadlines, links, phone numbers or e-mail addresses that the course does not state | |
| J5.5 | Try to download | "Nothing to download yet…": pending content is not downloadable | |
| J5.6 | Edit one caption, **Save edits as a new version** | Version 2 pending review; add a price ("only 500 EGP") and save: it is kept as a **Draft**, the problems are listed and **Approve** is refused | |
| J5.7 | **Approve this version** (a valid one) | "Approved. It can now be downloaded. Nothing was published or sent."; the review history shows `admin-session-…` | |
| J5.8 | Download Markdown and JSON | Both contain the approved version only | |
| J5.9 | Edit the approved campaign | A new pending version; the approved one stays downloadable and unchanged | |
| J5.10 | Create a campaign with only *LinkedIn posts* and 1 variation | Only that content is generated | |
| J5.11 | Ask agents: `Create a social media campaign for approved course CUR-…` | Routed to Marketing only | |
| J5.12 | Ask agents: `Write LinkedIn posts promoting approved course CUR-…` | Marketing only | |
| J5.13 | Ask agents: `Create a curriculum about digital marketing.` | Curriculum Agent only. No campaign, no document | |
| J5.14 | Ask agents: `Generate a question bank for CUR-… and create a marketing campaign for the same course.` | Document Generation, then Marketing, both on the approved course | |
| J5.15 | Ask agents: `Create a marketing campaign` (no course) | A list of approved courses; reply with a number to continue | |
| J5.16 | Use a course without a stated audience (or ask `Create a campaign for CUR-…` for one): at the audience question, answer `hello` | The agent asks again; `hello` is not used as the audience | |
| J5.17 | At the audience question, answer `Create a curriculum about SQL` | The question is dropped and the curriculum request is handled on its own | |
| J5.18 | At the audience question, answer `cancel` | "Cancelled…"; nothing is saved; a later `Final-year students` is treated as a new message, not as the audience | |
| J5.19 | At the audience question, wait more than 15 minutes, then answer `Final-year students` | Treated as a new message (the old question expired), not as the audience | |
| J5.20 | Answer the audience question with `Career changers and junior analysts` | The campaign is generated for that audience | |
| J5.21 | Look for any publish, schedule, e-mail or WhatsApp send control | There is none; approval only enables the download | |
| J5.22 | Switch to Arabic and create a campaign for an Arabic course | Arabic text; strict checks may reject it on the small model (nothing saved, error names the content type) | |

---

## J6. Admin sign-in lockout (extends Section H)

| # | Step | Expected | Result |
|---|---|---|---|
| J6.1 | In one browser window enter a wrong passcode 5 times | "Too many attempts. Please wait a minute and try again."; even the right passcode is refused for a minute | |
| J6.2 | While that window is locked, open a second window (a different browser session) and enter the right passcode | The second session signs in; it is not locked | |
| J6.3 | After a minute, use the first window again | The right passcode works | |

---

## M. Multi-agent workflows (Admin → **Ask agents**)

These run the real model and can take 5 to 15 minutes each. They write curricula, documents and campaigns to the app's databases.

| # | Request | Expected | Result |
|---|---|---|---|
| M.1 | `Create a 30-hour Excel for Business Analysis curriculum for beginners, then a course proposal document from it` | Summary: "1. Curriculum research: Curriculum drafted … [CUR-…, pending review; companion CUR-…]", "2. Document generation: Waiting for an administrator's approval…", "Step 2 used the curriculum from step 1.", "Done: 1/2 · Waiting: 1"; a review card with both languages | |
| M.2 | Approve both languages in the card | The rest resumes by itself: a message "Resumed after approval" and the course proposal generated from the **approved version** of that curriculum | |
| M.3 | `Create a 24-hour beginner curriculum about SQL for data analysts and then a marketing campaign for it`, then approve both languages | The campaign step runs for that course after the approval. Known limitation: with `qwen2.5:3b` a campaign with every channel is often refused by the claim checks (e.g. "no prior experience" when the course lists prerequisites); that refusal is correct, nothing is saved | |
| M.4 | `Find training tenders in Egypt and create a curriculum for a data analytics course` | Two steps: Tender research (Egypt) and Curriculum research; the curriculum's title does not mention tenders or Egypt; "These steps ran independently." | |
| M.5 | `Create a curriculum about SQL and then a marketing campaign for it` (no hours, no level) | The Curriculum agent asks for the missing details; step 2 is shown as waiting for your answer, **not** as failed | |
| M.6 | `Recommend a course for me` | Not handled by the admin agents (REX is not in the admin orchestrator) | |

---

## N. Browser acceptance (real browser, isolated copy)

Run against a **copy** of the project with its own data folders, `RESEND_API_KEY` empty and `REX_EMAIL_TEST_RECIPIENT` set to a test address, so nothing is sent and the real databases are not touched. Can be done by hand or with a browser driver (e.g. Playwright with Chrome).

| # | Steps | Expected | Result |
|---|---|---|---|
| N.1 | Learner: "What courses does the academy offer?" | 14 courses, each once | |
| N.2 | Learner: "What Data Analytics courses does the academy offer?" | DA-101, DA-201, DA-301 | |
| N.3 | Learner: "Tell me more about SQL and Power BI for Data Analysts.", then "What are the fees?" | DA-201 answered; the follow-up gives 7,800 EGP | |
| N.4 | Learner: a recommendation (Computer Engineer, Data Analytics, beginner, online or offline, 4000, 3 days a week) | Excel for Business Analysis (DA-101), 3,200 EGP, within budget; only catalogue courses | |
| N.5 | Switch to العربية; ask the catalogue and a course question in Arabic | Right-to-left layout; Arabic answers; 14 courses | |
| N.6 | Admin: sign in, open all nine sections; switch to Arabic and back | Each section opens; the section is kept when the language changes; the sidebar is right-to-left in Arabic | |
| N.7 | Admin: J.23 to J.25 and J.27 | As described there; the confirmed send is blocked ("RESEND_API_KEY is not configured") | |
| N.8 | Ask agents: create a curriculum; approve both languages in the chat card; download each PDF and the ZIP | Approve buttons disappear; still on Ask agents; the PDFs open; the ZIP holds two PDFs. If a language is a **Draft** (e.g. "Written in the wrong language: modules[1].title", a module title the translation left in English), it has no Approve button: open it in Curricula → Edit this version, fix the text, save (v2, pending review), approve, then download | |
| N.9 | Curricula: edit the English curriculum's overview → **Save changes**; approve the new version | "Changes saved as a new version, pending review."; v2 listed and approved; v1's approval kept in the history | |
| N.10 | Documents: generate a course proposal from that curriculum (v2); create its PDF; approve | The document names approved version 2; the PDF downloads; "Document approved." | |
| N.11 | Marketing: create a campaign for that course; approve; download Markdown | The campaign names the course; "Approved … Nothing was published or sent."; the Markdown downloads | |

---

## K. Failure handling

| # | Steps | Expected | Result |
|---|---|---|---|
| K.1 | Stop Ollama (`taskkill /IM ollama.exe /F`), ask REX a question | Friendly "couldn't write an answer" or error message; the app doesn't crash. **Restart Ollama.** | |
| K.2 | Empty `TAVILY_API_KEY`, restart, repeat I.1 | World Bank results still shown; the Sources line shows the web searches as `awaiting_credentials`. **Restore the key.** | |
| K.3 | Disconnect from the internet, repeat I.1 | Says data couldn't be retrieved; **not** "no tenders found" | |
| K.4 | Stop the app while the MCP server runs, start it again, ask REX | Works again (the MCP server restarts) | |

---

## L. What the automated tests already cover

These are checked by `python -m pytest -q` (A.1), with fake models and temporary databases:
- every file format;
- the chunking strategies;
- metadata filters;
- re-ingestion without duplicates;
- stale-vector removal;
- the translation edge cases;
- the cross-encoder threshold and the embedding-model guard;
- the MCP client restart and crash recovery;
- the "not stated in the catalogue" note (the new catalogue states every fee and delivery mode, so it can't be triggered manually);
- forged and replayed admin tokens;
- RFP web research with a fake web search: extraction, listing pages, filters, duplicates across sites, the business rule, pending status, search form options;
- the tender review, dashboard and alert actions;
- the tender MCP server (real subprocess) and that the RFP agent uses it;
- the application workflow: routes, the atomic claim against 6 racing connections, dry runs never recorded as submitted, unknown results blocking retries, restart recovery, review locks, and admin authorisation of every application action;
- Agents 4 and 5 with stand-in models: routing by deliverable, document and campaign validation, the approval workflow, approved-only downloads, the Documents and Marketing tabs (`AppTest`, not a real browser), audience follow-ups (valid, unrelated, empty, expired, cancelled, new request), and the per-session admin lockout;
- SQLite: concurrent readers and writers, rollback, busy timeout, real multi-process approvals, existing-database compatibility;
- the MCP client closing its server processes at exit;
- the Curriculum Agent with a stand-in model:
  - every route: generation, questions, repair, revisions, suggestions, alignment, refusals;
  - every validation rule, the tools and the curriculum MCP server (real subprocess);
  - English and Arabic PDFs read back, including letter joining, line order and embedded fonts;
  - routing (admin planning goes to Agent 3, learner requests to REX);
  - admin approval and its rules;
  - the Curricula tab.

A manual pass here confirms the **live** system: real models, real services, the real browser.

## Results summary

| Section | Passed | Failed | Notes |
|---|---|---|---|
| A. Automated + KB | | | |
| B. REX English | | | |
| C. Grounding | | | |
| D. Arabic | | | |
| E. Recommendation | | | |
| F. Consent | | | |
| G. UI | | | |
| H. Admin security | | | |
| I. RFP research | | | |
| J. Tender review | | | |
| J2. Applications | | | |
| J3. Curriculum Agent | | | |
| J4. Documents | | | |
| J5. Marketing | | | |
| M. Multi-agent | | | |
| N. Browser acceptance | | | |
| K. Failures | | | |

Tested by: ______  Date: ______
