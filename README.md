<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=20&height=220&section=header&text=testwallah.in&fontSize=80&fontColor=ffffff&animation=fadeIn&fontAlignY=38" />

### *Upload your mock test. Get graded in 90 seconds.*

<br>

<p>
  <img src="https://img.shields.io/badge/STATUS-LIVE-22c55e?style=for-the-badge&labelColor=000" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=000" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white&labelColor=000" />
  <img src="https://img.shields.io/badge/GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white&labelColor=000" />
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white&labelColor=000" />
  <img src="https://img.shields.io/badge/license-AGPL--3.0-93c5fd?style=for-the-badge&labelColor=000" />
</p>

<br>

</div>

## What this does

Upload a question paper and an answer sheet. Get back a graded score, a per-question breakdown, skill tags, five analytics charts, and an AI-written feedback memo. End to end, about 90 seconds for a 10-question paper.

The format does not matter. PDFs, Word docs, photos of handwritten pages, plain text, any combination. The system OCRs what it needs to OCR, parses what it can parse, and feeds everything to GPT-4o vision to do both the reading and the marking in one pass.

The interesting problem here is not "can an LLM grade a paper". It's *can an LLM grade a paper reliably enough that the feedback is actually useful?* That comes down to three things:

1. Constraining the model's output to a strict JSON schema so the feedback is structured, comparable, and chartable
2. Sending images directly to a multimodal model instead of OCR'ing first and stacking error sources
3. Wrapping every API call in retries with exponential backoff so transient failures stop looking like app crashes

That's the whole engineering problem. The rest is plumbing.

<br>

<table>
<tr>
<td width="50%" valign="top">

### What you give it

- The question paper *(any format)*
- Your answer sheet *(any format)*

That's it.

Supported file types:
- PDF
- Word (.docx, .doc)
- Images (PNG, JPG, JPEG, BMP, TIFF, WEBP)
- Plain text (TXT, CSV, MD)
- Multiple files per section

</td>
<td width="50%" valign="top">

### What you get back

- **Score + percentage**
- **Per-question breakdown** with reason for marks lost
- **Skill tags** on every question
- **Handwriting rank** (if handwritten)
- **5 charts** rendered with matplotlib
- **AI-written overview** with the top 3 mistakes you made and a worked example of how to fix them
- **Permanent attempt history** stored in your account

</td>
</tr>
</table>

<br>

## Why this works (when most LLM-grading projects don't)

Most "AI grades your paper" projects break the moment you hand them a real student's handwriting or a question with a diagram. Here's why this one doesn't:

<table>
<tr>
<td width="50%" valign="top">

#### 🧠 Multimodal in, structured out

Instead of running OCR first and then sending text to GPT, this sends the **raw image** directly to GPT-4o vision. The model sees the diagram, the handwriting, the equations, all of it at once.

The grading prompt is locked to a strict JSON schema. Every skill tag must be evidenced by the question or answer. No invented categories. Max 3 skills per question. The model is being graded on its grading.

</td>
<td width="50%" valign="top">

#### 🔁 Retries with backoff

Every OpenAI call goes through a `_retry()` wrapper with exponential backoff (4 attempts, 1s → 6s). Transient timeouts and rate limits stop being scary.

```python
def _retry(fn, *, attempts=4,
           base_delay=1.0, max_delay=6.0):
    # exponential backoff with jitter
    # retries on httpx errors, timeouts,
    # rate limits, transient 5xx
    ...
```

Five percent failure rate becomes effectively zero.

</td>
</tr>
</table>

<br>

## The chart that matters most

The most useful chart is **not the percentage**. It's the skill mastery bar.

Every question gets tagged with the skills it tested. *Kinematics*. *Free-body diagrams*. *Unit conversion*. *Stoichiometry*. The chart aggregates marks across all questions in each skill bucket, so you see your mastery percentage per skill across the whole paper.

Patterns emerge that the percentage hides. A student scoring 75% overall might be sitting at:

> 90% mastery on *Concept Application*
> 40% mastery on *Unit Conversion*

The mistakes aren't physics mistakes. They're arithmetic mistakes at the last step. The fix isn't *"study more physics"*, it's *"slow down on the last step."*

You cannot see that from a single percentage at the top of a paper. **You need the skill breakdown.** That's what this chart exists for.

<br>

## The stack

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser UI                          │
│        (Jinja templates, vanilla JS, drag-drop upload)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask app (app.py)                       │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────┐    │
│  │ File Intake │──▶│ OCR + Parse  │──▶│ Multimodal     │    │
│  │ PDF/DOCX/   │   │ PyPDF2,      │   │ Grading via    │    │
│  │ IMG/TXT     │   │ python-docx, │   │ GPT-4o Vision  │    │
│  └─────────────┘   │ Pillow       │   └────────┬───────┘    │
│                    └──────────────┘            │            │
│                                                ▼            │
│                              ┌──────────────────────────┐   │
│                              │ Strict JSON schema       │   │
│                              │ + retry with backoff     │   │
│                              └──────────┬───────────────┘   │
│                                         ▼                   │
│             ┌─────────────────┐  ┌─────────────────┐        │
│             │ Matplotlib      │  │ SQLite          │        │
│             │ 5 charts (Agg)  │  │ users +         │        │
│             │                 │  │ attempt history │        │
│             └─────────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

| Layer | What it is | Why this and not something else |
|-------|------------|---------------------------------|
| **Flask** | Web framework | Tiny, no magic, perfect for a 1-file app |
| **SQLite** | User accounts + every attempt ever submitted | Zero setup, persists between sessions, file-based |
| **GPT-4o (vision)** | Does both the OCR AND the grading | One model = one source of error, not two |
| **PyPDF2 + python-docx** | Parses PDFs and Word docs | The boring infra that lets "any format" actually mean any format |
| **Pillow** | Preprocesses images before OCR (downscale → grayscale → contrast) | Cuts payload size and improves OCR accuracy in one step |
| **Matplotlib (Agg)** | Renders charts headlessly | Server-side rendering, no browser needed |
| **Werkzeug** | Password hashing + session cookies | Boring, secure, ships with Flask |
| **Render** | Hosting | Free tier, supports background workers, just works |

<br>

## What's actually built

<table>
<tr>
<td width="33%" valign="top">

### 🔐 Auth

- Email + password signup
- Session cookies
- "Remember me" with 60-day persistence
- Per-user attempt isolation
- Password hashing via Werkzeug

</td>
<td width="33%" valign="top">

### 📄 File handling

- 10 file types accepted
- Up to 10 images per upload (capped to control OpenAI cost)
- Multimodal grading when images are present
- Text-only grading when they're not
- Image preprocessing pipeline

</td>
<td width="33%" valign="top">

### 📊 Reporting

- 5 matplotlib charts per attempt
- AI-written study overview
- Permanent attempt history
- Per-attempt detail page
- Skill-tagged feedback
- Marks-lost-per-question breakdown

</td>
</tr>
</table>

<br>

## Running it locally

```bash
git clone https://github.com/RicKanjilal/testwallah.in.git
cd testwallah.in
pip install -r requirements.txt
```

You'll need three environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_PROJECT_ID="proj_..."
export APP_SECRET="any-random-string-for-sessions"
```

Then:

```bash
python app.py
```

Opens on `http://localhost:8000`. SQLite database is created on first run. Sign up with any email, upload a paper, get graded.

<br>

## Stuff I learned building this

**LLM grading is shockingly good when you constrain the output format.** The difference between *"grade this paper"* and *"grade this paper and return strictly this JSON schema with these exact keys and these exact constraints"* is the difference between unusable and production-ready.

**Multimodal models change everything for OCR-heavy projects.** I was originally going to run Tesseract for OCR, then send the text to GPT-4o for grading. Two error sources stacked. Just sending the image directly to GPT-4o cuts Tesseract entirely and the model handles diagrams + handwriting in one shot.

**Retries matter more than you think.** The OpenAI API fails maybe 5% of the time on long requests. A single failure in production looks like the whole app is broken. Wrapping every LLM call in `_retry()` with exponential backoff turned a flaky app into a reliable one for the cost of about 30 lines of code.

**Most of the work in any "AI app" is the boring parts.** Auth, sessions, DB schema, file uploads, error handling, deploying to Render. The GPT-4o call is 5% of the codebase. The other 95% is making sure people can actually use it without it crashing.

<br>

## Roadmap (if I pick this back up)

- [ ] Compare two attempts side by side *("am I getting better at unit conversion?")*
- [ ] Weak-areas dashboard aggregating skill mastery across all attempts
- [ ] Export full report as a single printable PDF
- [ ] Teacher account type that pulls all attempts from a class
- [ ] Batch grading. Drop in 30 papers, get 30 reports
- [ ] Offline mode using a local LLM so it works without an API key

<br>

## License

**AGPL-3.0.** If you fork this and run it as a service, you have to share your modified source code.

I picked this license on purpose. I do not want someone wrapping this in a paywall and charging students for it without contributing back.

<br>

---

<div align="center">

<sub>Built by <a href="https://github.com/RicKanjilal"><b>Ric Kanjilal</b></a> · Grade 10 · Don Bosco School, Liluah · Kolkata</sub>

</div>
