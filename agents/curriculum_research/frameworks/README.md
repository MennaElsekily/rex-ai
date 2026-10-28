# Approved framework documents

Put one JSON file per framework here, named `<framework_id>.json`, taken
from an **official** source whose licence allows the use (see the
allowlist in `docs/CURRICULUM_AGENT_ARCHITECTURE.md` §16 — for example
SFIA needs a licence for external mapping). None is shipped: until a file
is added, alignment reports **unable to determine**.

```json
{
  "framework_id": "nice-cybersecurity",
  "name": "NICE Workforce Framework for Cybersecurity",
  "publisher": "NIST",
  "version": "SP 800-181 Rev. 1",
  "source_url": "https://csrc.nist.gov/pubs/sp/800/181/r1/final",
  "retrieved_on": "2026-10-27",
  "language": "en",
  "licence_note": "US government publication",
  "requirements": [
    {"id": "K0001", "text": "Knowledge of ...", "category": "knowledge"}
  ]
}
```

Rules enforced by `frameworks.py`: the id is 2–40 lower-case letters,
digits, `-` or `_`; it must match the file name; all provenance fields
are required; 1–200 requirements, each with an `id` and a `text`. Copy
requirement texts exactly as published; never write them yourself.
Alignment against a framework is **not** accreditation.
