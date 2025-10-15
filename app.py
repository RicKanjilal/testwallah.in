import uuid, time
import os, io, json, csv, base64, datetime as dt
from typing import Dict, List
import numpy as np
import pandas as pd
# ===== Headless plotting (fix Tkinter/thread errors) =====
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from openai import OpenAI
from PIL import Image, ImageEnhance
# NEW imports
import sqlite3
from contextlib import closing
from flask import session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

# After: app = Flask(__name__, template_folder="templates")

matplotlib.use("Agg")#better manners according to pycharm
# =========================
# CONFIG & THEME SETTINGS
# =========================
QP_TXT   = "questionpaper.txt"
ANS_TXT  = "answersheet.txt"
OUT_JSON = "grading_result.json"
ATTEMPTS = "attempts.csv"
OUT_DIR  = "charts_deep"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = ["#7c3aed","#06b6d4","#22c55e","#f59e0b","#ef4444",
           "#a855f7","#14b8a6","#60a5fa","#eab308","#f97316"]

plt.rcParams.update({
    "figure.facecolor": "#0f111a","axes.facecolor":"#0f111a","savefig.facecolor":"#0f111a",
    "axes.edgecolor":"#e6e6e6","axes.labelcolor":"#e6e6e6",
    "xtick.color":"#e6e6e6","ytick.color":"#e6e6e6",
    "text.color":"#e6e6e6","grid.color":"#2a2f3a",
    "axes.titleweight":"bold","axes.titlesize":14,"font.size":11
})
DB_PATH = "app.db"

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with closing(db()) as conn, conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            subject TEXT NOT NULL,
            score REAL NOT NULL,
            total INTEGER NOT NULL,
            percent REAL NOT NULL,
            handwriting REAL NOT NULL,
            run_dir TEXT NOT NULL,
            qp_path TEXT NOT NULL,
            ans_path TEXT NOT NULL,
            grading_json TEXT NOT NULL,
            overview_path TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)
init_db()
def current_user_id():
    return session.get("uid")

def login_required():
    if not current_user_id():
        return False
    return True

def _new_run_dir() -> str:
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = os.path.join(OUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
def save_attempt_to_db(user_id, subject, result, run_dir, files):
    with closing(db()) as conn, conn:
        cur = conn.execute("""
          INSERT INTO attempts(user_id,date,subject,score,total,percent,handwriting,
                               run_dir,qp_path,ans_path,grading_json,overview_path)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
          user_id, dt.date.today().isoformat(), subject,
          float(result["score"]), int(result["total"]), float(result["percentage"]),
          float(result.get("handwriting_rank", 0) or 0),
          run_dir, files["qp"], files["ans"], files["grading_json"], files["overview_txt"]
        ))
        return cur.lastrowid



# =========================
# OPENAI CLIENT
# =========================
def _get_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key and os.path.exists("config.json"):
        key = json.load(open("config.json", "r", encoding="utf-8"))["OPENAI_API_KEY"]
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (env or config.json).")
    return key

client = OpenAI(api_key=_get_key())

# =========================
# IMAGE OCR HELPERS (GPT-4o)
# =========================
def preprocess_image_bytes(in_bytes: bytes, width: int = 1200) -> bytes:
    """Grayscale, contrast boost, resize → PNG bytes (helps OCR)."""
    img = Image.open(io.BytesIO(in_bytes)).convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.resize((width, int(img.height * width / img.width)))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()
# ---------- Multimodal helpers ----------
def to_data_url(img_bytes: bytes, mime: str = "image/png") -> str:
    """Encode raw bytes to a base64 data URL for OpenAI image content."""
    b64 = base64.b64encode(img_bytes).decode()
    return f"data:{mime};base64,{b64}"


def ocr_gpt4o_from_bytes(image_png_bytes: bytes) -> str:
    """OCR via GPT-4o vision using a base64 data URL."""
    b64 = base64.b64encode(image_png_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"
    prompt = "Extract ALL readable text as plain text only. Keep natural reading order."
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        temperature=0.0,
        max_tokens=3500
    )
    return resp.choices[0].message.content.strip()

# =========================
# GRADING (STRICT JSON)
# =========================
def grade_with_skills(qp_text: str, ans_text: str) -> Dict:
    """
    Ask GPT-4o to:
      - map answers to questions
      - grade each question
      - return per-item skills, answer_excerpt, reason
      - provide expected cohort averages & handwriting rank
    Return ONLY JSON; we validate/normalize totals.
    """
    system = (
        "You are a strict school examiner and analyst. "
        "Return ONLY valid JSON with the exact keys requested. "
        "Every skill label must be explicitly evidenced by the question requirement "
        "or student approach; do not invent categories. ≤3 skills per item."
    )
    user = f"""
GRADE THIS PAPER AND RETURN ONLY JSON.

QUESTION PAPER (plain text):
{qp_text}

STUDENT ANSWERS (plain text):
{ans_text}

Map answers to questions reliably. If the student numbered answers, use that; otherwise infer by content cues.

Return ONLY JSON with keys EXACTLY:
{{
  "meta": {{"subject":"string?","topic":"string?","class":"string?","board":"string?"}},
  "items": [
    {{
      "id":"Q1",
      "question_excerpt":"<=25 words identifying the requirement",
      "answer_excerpt":"<=35 words quoting or tightly paraphrasing the student's answer to THIS question",
      "max": int,
      "awarded": float,
      "skills": ["skill1","skill2?"],     // ≤3, strictly from content
      "reason":"<=15 words, concise feedback citing evidence"
    }}
  ],
  "total": int,                          // sum of max
  "score": float,                        // sum of awarded
  "percentage": float,                   // (score/total)*100
  "handwriting_rank": int,               // 1-10
  "expected_avg_percent": float,         // estimated cohort average for THIS paper
  "expected_avg_handwriting": float,     // typical cohort handwriting rank (1-10)
  "tips_next_time": ["tip1","tip2","tip3","tip4"]  // 3–6 practical, paper-specific tips
}}

Rules:
- If marks per question aren't printed, allocate sensible max marks consistent with the paper style.
- Award fairly; at most one decimal place.
- Skills MUST be evidenced by the question or student's method; e.g., use 'Graph interpretation' ONLY if a graph is referenced.
- Tips must be derived from the actual mistakes and patterns in these answers—no generic advice.
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=0.2,
        max_tokens=2000
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.strip().lower().startswith("json"):
                i = text.find("{")
                if i != -1:
                    text = text[i:]
    data = json.loads(text)

    # Normalize totals
    total = int(round(sum(float(i.get("max",0) or 0) for i in data.get("items", []))))
    score = float(sum(float(i.get("awarded",0) or 0) for i in data.get("items", [])))
    data["total"] = data.get("total", total) or total
    data["score"] = data.get("score", score) or score
    data["percentage"] = data.get("percentage", round(100*score/max(total,1),2))

    # Clean fields
    for it in data.get("items", []):
        skills = it.get("skills", []) or []
        it["skills"] = [str(s).strip()[:60] for s in skills if str(s).strip()][:3]
        it["question_excerpt"] = (it.get("question_excerpt") or "")[:200]
        it["answer_excerpt"] = (it.get("answer_excerpt") or "")[:240]
        it["reason"] = (it.get("reason") or "")[:120]
    return data
def grade_with_skills_mm(qp_text: str,
                         ans_text: str,
                         qp_images_dataurls: List[str],
                         ans_images_dataurls: List[str]) -> Dict:
    """
    Multimodal grading:
      - Sends ORIGINAL question-paper images + answer-sheet images to GPT-4o
      - Also provides OCR text (if available) to help with small text
      - Asks the model to align answers to questions, including diagram interpretation
    Returns the SAME JSON schema as grade_with_skills so frontend/charts keep working.
    """
    system = (
        "You are a strict school examiner and analyst.\n"
        "You can see both IMAGES and extracted TEXT.\n"
        "Use visual evidence first (diagrams, graphs, figures), and text for small labels.\n"
        "Return ONLY valid JSON with the exact keys requested. ≤3 skills per item; skills must be supported by the question/answer evidence."
    )

    # Build multimodal content: first a short instruction, then all QP images, then all ANS images,
    # then the OCR text (if provided) to augment.
    content = [
        {"type": "text", "text": (
            "GRADE THIS PAPER USING IMAGES (primary) + TEXT (secondary). "
            "Map answers to questions reliably. If the student numbered answers, use that; "
            "otherwise infer by content cues and visual alignment. "
            "When questions are diagram/graph-based, interpret the figure and its axes/arrows/labels."
        )}
    ]

    # Attach question-paper images
    for url in qp_images_dataurls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    # Attach answer-sheet images
    for url in ans_images_dataurls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    # Add OCR text (if available)
    if qp_text.strip():
        content.append({"type": "text", "text": "QUESTION PAPER (OCR/text, may be imperfect):\n" + qp_text})
    if ans_text.strip():
        content.append({"type": "text", "text": "STUDENT ANSWERS (OCR/text, may be imperfect):\n" + ans_text})

    # Final JSON spec
    content.append({
        "type": "text",
        "text": """
Return ONLY JSON with keys EXACTLY:
{
  "meta": {"subject":"string?","topic":"string?","class":"string?","board":"string?"},
  "items": [
    {
      "id":"Q1",
      "question_excerpt":"<=25 words identifying the requirement (include visual cue if any, e.g., 'bar chart on growth 2018-2022')",
      "answer_excerpt":"<=35 words paraphrasing the student's answer (may reference what their diagram shows)",
      "max": int,
      "awarded": float,
      "skills": ["skill1","skill2?"],
      "reason":"<=15 words citing concrete evidence (e.g., 'mismeasured axis labels')"
    }
  ],
  "total": int,
  "score": float,
  "percentage": float,
  "handwriting_rank": int,
  "expected_avg_percent": float,
  "expected_avg_handwriting": float,
  "tips_next_time": ["tip1","tip2","tip3","tip4"]
}

Rules:
- Use diagram/graph/table evidence directly from the images when present.
- If the student's drawing is incorrect (e.g., wrong axis scale, missing legend), penalize and explain briefly in 'reason'.
- Award fairly; at most one decimal place.
- Do not invent skills; only those supported by the actual question/answer.
"""
    })

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content}],
        temperature=0.2,
        max_tokens=2000
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.strip().lower().startswith("json"):
                i = text.find("{")
                if i != -1:
                    text = text[i:]
    data = json.loads(text)

    # Normalize totals (same as your text-only grader)
    total = int(round(sum(float(i.get("max",0) or 0) for i in data.get("items", []))))
    score = float(sum(float(i.get("awarded",0) or 0) for i in data.get("items", [])))
    data["total"] = data.get("total", total) or total
    data["score"] = data.get("score", score) or score
    data["percentage"] = data.get("percentage", round(100*score/max(total,1),2))

    for it in data.get("items", []):
        skills = it.get("skills", []) or []
        it["skills"] = [str(s).strip()[:60] for s in skills if str(s).strip()][:3]
        it["question_excerpt"] = (it.get("question_excerpt") or "")[:200]
        it["answer_excerpt"] = (it.get("answer_excerpt") or "")[:240]
        it["reason"] = (it.get("reason") or "")[:120]
    return data


# =========================
# AGGREGATION FOR CHARTS
# =========================
def split_marks_by_skills(items: List[Dict]) -> Dict[str, Dict]:
    """Split each question's marks evenly across its skill tags."""
    pool = {}
    for it in items:
        mx = float(it.get("max",0) or 0)
        aw = float(it.get("awarded",0) or 0)
        skills = it.get("skills") or ["Unlabeled"]
        share = 1.0 / len(skills)
        for s in skills:
            d = pool.setdefault(s, {"max":0.0,"awarded":0.0,"count":0})
            d["max"] += mx * share
            d["awarded"] += aw * share
            d["count"] += 1
    return pool

def color_cycle(n):
    if n <= len(PALETTE): return PALETTE[:n]
    cm = plt.get_cmap("viridis")
    return [cm(i/(n-1)) for i in range(n)]

# =========================
# CHART RENDERERS (5)
# =========================
def chart_skill_mix_pie(pool: Dict, subject: str, out_dir: str):
    df = (pd.DataFrame([{"skill":k, **v} for k,v in pool.items()])
          .sort_values("max", ascending=False))
    if df.empty or df["max"].sum() <= 0: return
    vals, labels = df["max"].values, df["skill"].values
    colors = color_cycle(len(labels))
    fig = plt.figure(figsize=(8.25,6.25))  # +6 px feel
    plt.pie(vals, labels=labels, autopct="%1.0f%%",
            startangle=140, pctdistance=0.75, colors=colors)
    centre = plt.Circle((0,0),0.50,fc="#0f111a")
    fig.gca().add_artist(centre)
    plt.title(f"Skill Mix (Weight by Marks) • {subject}")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_skill_mix_pie.png"), dpi=150)
    plt.close(fig)

def chart_skill_mastery(pool: Dict, out_dir: str):

    df = pd.DataFrame([{"skill":k, **v} for k,v in pool.items()])
    if df.empty: return
    df["mastery"] = np.where(df["max"]>0, df["awarded"]/df["max"], 0.0)
    df = df.sort_values("mastery")
    colors = ["#10b981" if m>=0.75 else ("#f59e0b" if m>=0.5 else "#ef4444") for m in df["mastery"]]
    fig, ax = plt.subplots(figsize=(9.25,6.25))
    bars = ax.barh(df["skill"], df["mastery"]*100, color=colors)
    for b, m in zip(bars, df["mastery"]):
        ax.text(b.get_width()+1, b.get_y()+b.get_height()/2, f"{m*100:.0f}%", va="center")
    ax.set_xlabel("Mastery (%)"); ax.set_title("Skill Mastery (Awarded / Max)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "2_skill_mastery_bar.png"), dpi=150)
    plt.close(fig)

def chart_compare_expected(pct, handwriting, exp_pct, exp_hand, out_dir: str):
    fig, axs = plt.subplots(1,2, figsize=(10.25,5.25))
    axs[0].bar(["Obtained","Expected"], [pct, exp_pct], color=["#7c3aed","#94a3b8"])
    axs[0].set_ylim(0, 100); axs[0].set_title("Marks: Obtained vs Expected")
    for i,v in enumerate([pct,exp_pct]): axs[0].text(i, v+2, f"{v:.1f}%", ha="center")
    axs[1].bar(["Obtained","Expected"], [handwriting, exp_hand], color=["#06b6d4","#94a3b8"])
    axs[1].set_ylim(0, 10); axs[1].set_title("Handwriting Rank: Obtained vs Expected")
    for i,v in enumerate([handwriting,exp_hand]): axs[1].text(i, v+0.3, f"{v:.1f}", ha="center")
    plt.suptitle("Benchmarks Comparison", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "3_benchmarks_compare.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

def chart_marks_lost_by_question(items: List[Dict], out_dir: str):
    df = pd.DataFrame([{
        "id": it.get("id","?"),
        "lost": max(0.0, float(it.get("max",0) or 0) - float(it.get("awarded",0) or 0)),
        "reason": it.get("reason","")
    } for it in items])
    if df.empty: return
    df = df.sort_values("lost")
    fig, ax = plt.subplots(figsize=(9.25,6.25))
    cols = ["#10b981" if l == 0 else "#ef4444" for l in df["lost"]]
    bars = ax.barh(df["id"], df["lost"], color=cols)
    for b,l,reason in zip(bars, df["lost"], df["reason"]):
        if l>0: ax.text(b.get_width()+0.1, b.get_y()+b.get_height()/2, f"-{l:.1f} ({reason[:40]}...)", va="center")
    ax.set_xlabel("Marks Lost"); ax.set_title("Marks Lost by Question")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "4_marks_lost_by_question.png"), dpi=150)
    plt.close(fig)
def chart_radar(pool: Dict, out_dir: str):
    df = pd.DataFrame([{"skill":k, **v} for k,v in pool.items()])
    if df.empty: return
    df["mastery"] = np.where(df["max"]>0, df["awarded"]/df["max"], 0.0)
    df = df.sort_values("skill")
    labels = df["skill"].tolist()
    vals = df["mastery"].tolist()
    if not labels: return
    vals += vals[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    fig = plt.figure(figsize=(7.25,7.25))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0); ticks = [0.25,0.5,0.75,1.0]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{int(t*100)}%" for t in ticks], color="#94a3b8")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.plot(angles, vals, linewidth=2, color="#7c3aed")
    ax.fill(angles, vals, color="#7c3aed", alpha=0.25)
    ax.set_title("Skill Mastery Radar")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "5_skill_radar.png"), dpi=150)
    plt.close(fig)

# =========================
# OVERVIEW (LLM) → returns text and writes overview.txt
# =========================
def write_overview_with_llm(result: Dict) -> str:
    prompt = (
        "Create a comprehensive, practical overview for the student based ONLY on this JSON. "
        "Do not invent skills or generic advice. Reference actual question IDs and skills present. "
        "Include exactly:\n"
        "1) 3–5 Key takeaways tied to the data.\n"
        "2) Top 3 costly mistakes (cite question IDs and why).\n"
        "3) What to do better next time (bullet list, concrete, paper-specific).\n"
        "4) 1–2 short worked examples/templates derived from actual mistakes.\n"
        "No study schedule. Keep it exam-ready.\n\n"
        f"JSON:\n{json.dumps(result, ensure_ascii=False)}"
    )
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=900
    )
    text = r.choices[0].message.content.strip()
    path = os.path.join(OUT_DIR, "overview.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

# =========================
# PIPELINE FOR ONE RUN
# =========================
def run_once(qp_text: str, ans_text: str) -> Dict:
    # Start fresh matplotlib state each run
    plt.close('all')

    # Save input texts (debug/downloads)
    with open(QP_TXT, "w", encoding="utf-8") as f: f.write(qp_text)
    with open(ANS_TXT, "w", encoding="utf-8") as f: f.write(ans_text)

    # Grade & analyze
    result = grade_with_skills(qp_text, ans_text)

    # Save JSON result
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Per-run output dir (unique every evaluation)
    run_dir = _new_run_dir()

    # Log attempt
    subject = (result.get("meta") or {}).get("subject") or "Unknown"
    row = {
        "date": dt.date.today().isoformat(),
        "subject": subject,
        "score": result["score"],
        "total": result["total"],
        "percent": result["percentage"],
        "handwriting": result.get("handwriting_rank", 0),
        "weak_tags": ""
    }
    write_header = not os.path.exists(ATTEMPTS)
    with open(ATTEMPTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: w.writeheader()
        w.writerow(row)

    # Charts (write into this run's folder)
    pool = split_marks_by_skills(result.get("items", []))
    chart_skill_mix_pie(pool, subject, run_dir)
    chart_skill_mastery(pool, run_dir)
    chart_compare_expected(
        result["percentage"],
        result.get("handwriting_rank", 0),
        float(result.get("expected_avg_percent", 0.0) or 0.0),
        float(result.get("expected_avg_handwriting", 0.0) or 0.0),
        run_dir
    )
    chart_marks_lost_by_question(result.get("items", []), run_dir)
    chart_radar(pool, run_dir)

    # Overview text (save into this run's folder too)
    overview_text = write_overview_with_llm(result)
    with open(os.path.join(run_dir, "overview.txt"), "w", encoding="utf-8") as f:
        f.write(overview_text)

    # (Optional) add a tiny cache-buster query for browsers that still cache aggressively
    ver = f"?v={int(time.time()*1000)}"

    return {
        "result": result,
        "overview_text": overview_text,
        "files": {
            "qp": f"/static_file/{QP_TXT}{ver}",
            "ans": f"/static_file/{ANS_TXT}{ver}",
            "grading_json": f"/static_file/{OUT_JSON}{ver}",
            "overview_txt": f"/static_file/{os.path.join(run_dir,'overview.txt')}{ver}",
            "charts": [
                f"/static_file/{os.path.join(run_dir,'1_skill_mix_pie.png')}{ver}",
                f"/static_file/{os.path.join(run_dir,'2_skill_mastery_bar.png')}{ver}",
                f"/static_file/{os.path.join(run_dir,'3_benchmarks_compare.png')}{ver}",
                f"/static_file/{os.path.join(run_dir,'4_marks_lost_by_question.png')}{ver}",
                f"/static_file/{os.path.join(run_dir,'5_skill_radar.png')}{ver}",
            ]
        }
    }
def run_once_mm(qp_text: str,
                ans_text: str,
                qp_dataurls: List[str],
                ans_dataurls: List[str]) -> Dict:
    # Start fresh matplotlib state each run
    plt.close('all')

    # Persist last OCR’d text copies for download/debug
    with open(QP_TXT, "w", encoding="utf-8") as f: f.write(qp_text or "")
    with open(ANS_TXT, "w", encoding="utf-8") as f: f.write(ans_text or "")

    # Grade (multimodal)
    result = grade_with_skills_mm(qp_text or "", ans_text or "", qp_dataurls, ans_dataurls)

    # Save JSON result
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Per-run directory (you added this earlier in the caching fix)
    run_dir = _new_run_dir()

    # Log attempt
    subject = (result.get("meta") or {}).get("subject") or "Unknown"
    row = {
        "date": dt.date.today().isoformat(),
        "subject": subject,
        "score": result["score"],
        "total": result["total"],
        "percent": result["percentage"],
        "handwriting": result.get("handwriting_rank", 0),
        "weak_tags": ""
    }
    write_header = not os.path.exists(ATTEMPTS)
    with open(ATTEMPTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: w.writeheader()
        w.writerow(row)

    # Charts
    pool = split_marks_by_skills(result.get("items", []))
    chart_skill_mix_pie(pool, subject, run_dir)
    chart_skill_mastery(pool, run_dir)
    chart_compare_expected(
        result["percentage"],
        result.get("handwriting_rank", 0),
        float(result.get("expected_avg_percent", 0.0) or 0.0),
        float(result.get("expected_avg_handwriting", 0.0) or 0.0),
        run_dir
    )
    chart_marks_lost_by_question(result.get("items", []), run_dir)
    chart_radar(pool, run_dir)

    # Overview
    overview_text = write_overview_with_llm(result)
    with open(os.path.join(run_dir, "overview.txt"), "w", encoding="utf-8") as f:
        f.write(overview_text)

    ver = f"?v={int(time.time()*1000)}"
    return {
        "result": result,
        "overview_text": overview_text,
        "files": {
            "qp": f"/static_file/{QP_TXT}{ver}",
            "ans": f"/static_file/{ANS_TXT}{ver}",
            "grading_json": f"/static_file/{OUT_JSON}{ver}",
            "overview_txt": f"/static_file/{os.path.join(run_dir,'overview.txt')}{ver}",
            "charts": [
                f"/static_file/{os.path.join(run_dir,'1_skill_mix_pie.png')}{ver}",
                f"/static_file/{os.path.join(run_dir,'2_skill_mastery_bar.png')}{ver}",
                f"/static_file/{os.path.join(run_dir,'3_benchmarks_compare.png')}{ver}",
                f"/static_file/{os.path.join(run_dir,'4_marks_lost_by_question.png')}{ver}",
                f"/static_file/{os.path.join(run_dir,'5_skill_radar.png')}{ver}",
            ]
        }
    }

# =========================
# FLASK APP
# =========================
app = Flask(__name__, template_folder="templates")
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.secret_key = os.getenv("APP_SECRET", "dev-secret-change-me")  # for sessions/cookies
app.permanent_session_lifetime = dt.timedelta(days=60)            # 'remember me' window

@app.route("/auth")
def auth_page():
    return render_template("auth.html")
@app.route('/qpjen')
def qpjen():
    return render_template('qpjen.html')



@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    ph = generate_password_hash(password)
    try:
        with closing(db()) as conn, conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?,?,?)",
                (email, ph, dt.datetime.utcnow().isoformat())
            )
            uid = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()["id"]
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409
    session["uid"] = uid
    session["email"] = email
    return jsonify({"ok": True, "email": email})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    remember = bool(data.get("remember"))
    with closing(db()) as conn:
        row = conn.execute("SELECT id, password_hash FROM users WHERE email=?", (email,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401
    session["uid"] = row["id"]
    session["email"] = email
    session.permanent = remember
    return jsonify({"ok": True, "email": email, "remember": remember})

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/me")
def api_me():
    if not current_user_id():
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "email": session.get("email")})

@app.route("/")
def index():
    # Frontend HTML should be in templates/index.html
    return render_template("index.html")
def _raw_path(url: str) -> str:
    return (url or "").split("?", 1)[0]


from PyPDF2 import PdfReader
from docx import Document

@app.route("/api/grade", methods=["POST"])
def api_grade():
    # ---- Auth guard ----
    if not current_user_id():
        return jsonify({"error": "Please sign in"}), 401
    uid = current_user_id()

    """
    Universal file intake — accepts any combination of:
      - text (plain)
      - PDF
      - Word DOC/DOCX
      - images (PNG, JPG, JPEG, etc.)
      - multiple files per section
    """
    qp_text, ans_text = "", ""
    qp_dataurls, ans_dataurls = [], []

    # Collect all question paper files
    qp_files = request.files.getlist("qp_files[]") or request.files.getlist("qp_imgs[]") or []
    ans_files = request.files.getlist("ans_files[]") or request.files.getlist("ans_imgs[]") or []

    def extract_text_and_dataurls(file_list, label="QP"):
        texts, dataurls = [], []
        for f in file_list:
            try:
                filename = f.filename.lower()
                raw = f.read()

                # --- TEXT FILE ---
                if filename.endswith((".txt", ".csv", ".md", ".log")):
                    texts.append(raw.decode("utf-8", errors="ignore"))

                # --- PDF FILE ---
                elif filename.endswith(".pdf"):
                    pdf = PdfReader(io.BytesIO(raw))
                    pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    texts.append(pdf_text)

                # --- WORD DOC/DOCX ---
                elif filename.endswith((".docx", ".doc")):
                    doc = Document(io.BytesIO(raw))
                    doc_text = "\n".join(p.text for p in doc.paragraphs)
                    texts.append(doc_text)

                # --- IMAGE FILE (jpeg/png/webp etc.) ---
                elif filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
                    dataurls.append(to_data_url(raw, mime="image/png"))
                    # optional OCR (for small handwriting/labels)
                    try:
                        texts.append(ocr_gpt4o_from_bytes(preprocess_image_bytes(raw)))
                    except Exception as e:
                        print(f"OCR error ({label}):", e)

                # --- FALLBACK: treat as plain text ---
                else:
                    try:
                        texts.append(raw.decode("utf-8", errors="ignore"))
                    except Exception:
                        pass

            except Exception as e:
                print(f"File parse error ({label}):", e)
        return "\n\n".join(texts).strip(), dataurls

    # Extract everything
    qp_text, qp_dataurls = extract_text_and_dataurls(qp_files, "QP")
    ans_text, ans_dataurls = extract_text_and_dataurls(ans_files, "ANS")

    # If nothing uploaded via files, fallback to text form fields
    if not (qp_text or qp_dataurls):
        qp_text = request.form.get("qp_text", "")
    if not (ans_text or ans_dataurls):
        ans_text = request.form.get("ans_text", "")

    # Must have something to work with
    if not ((qp_text or qp_dataurls) and (ans_text or ans_dataurls)):
        return jsonify({"error": "Upload or enter question paper and answer sheet — any file type is supported."}), 400

    # ---- Run grader (MM if any images) ----
    try:
        if qp_dataurls or ans_dataurls:
            out = run_once_mm(qp_text, ans_text, qp_dataurls, ans_dataurls)
        else:
            out = run_once(qp_text, ans_text)

        # ---- Persist the attempt to SQLite ----
        # subject
        subject = (out.get("result", {}).get("meta") or {}).get("subject") or "Unknown"

        # run_dir: try to derive from stored file paths
        run_dir = ""
        try:
            # Prefer grading_json path if present
            gj = out["files"]["grading_json"]
            # supports both '/static_file/<dir>/grading.json' and absolute variants
            base = gj.split("/static_file/", 1)[-1]
            run_dir = os.path.dirname(base).replace("\\", "/")
        except Exception:
            try:
                c0 = out["files"]["charts"][0]
                base = c0.split("/static_file/", 1)[-1]
                run_dir = os.path.dirname(base).replace("\\", "/")
            except Exception:
                run_dir = ""

        attempt_id = save_attempt_to_db(
            user_id=uid,
            subject=subject,
            result=out["result"],
            run_dir=run_dir,
            files={
                "qp": _raw_path(out["files"]["qp"]),
                "ans": _raw_path(out["files"]["ans"]),
                "grading_json": _raw_path(out["files"]["grading_json"]),
                "overview_txt": _raw_path(out["files"]["overview_txt"]),
            },
        )

        out["attempt_id"] = attempt_id
        return jsonify(out)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/static_file/<path:fname>")
def static_file(fname):
    base = os.getcwd()
    resp = send_from_directory(base, fname, max_age=0)
    resp.cache_control.no_store = True
    resp.cache_control.must_revalidate = True
    resp.cache_control.max_age = 0
    return resp


@app.route("/health")
def health():
    return "ok"
@app.route("/api/history")
def api_history():
    if not current_user_id(): return jsonify({"error":"Unauthorized"}), 401
    with closing(db()) as conn:
        rows = conn.execute("""
          SELECT id,date,subject,score,total,percent,handwriting
          FROM attempts WHERE user_id=? ORDER BY id DESC LIMIT 200
        """,(current_user_id(),)).fetchall()
    return jsonify({"items":[dict(r) for r in rows]})

@app.route("/api/attempt/<int:aid>")
def api_attempt(aid):
    if not current_user_id(): return jsonify({"error":"Unauthorized"}), 401
    with closing(db()) as conn:
        a = conn.execute("SELECT * FROM attempts WHERE id=? AND user_id=?",
                         (aid, current_user_id())).fetchone()
    if not a: return jsonify({"error":"Not found"}), 404

    # try to load stored grading JSON (safe if file exists)
    try:
        p = a["grading_json"]
        # remove /static_file/ prefix and any ?v= cachebuster
        jpath = p.split("/static_file/", 1)[-1].split("?", 1)[0]
        result = json.load(open(jpath, "r", encoding="utf-8"))
    except Exception:
        result = {}

    return jsonify({
      "attempt": dict(a),
      "result": result,
      "files": {
        "qp": a["qp_path"], "ans": a["ans_path"],
        "grading_json": a["grading_json"], "overview_txt": a["overview_path"],
        "charts": [
          f"/static_file/{a['run_dir']}/1_skill_mix_pie.png",
          f"/static_file/{a['run_dir']}/2_skill_mastery_bar.png",
          f"/static_file/{a['run_dir']}/3_benchmarks_compare.png",
          f"/static_file/{a['run_dir']}/4_marks_lost_by_question.png",
          f"/static_file/{a['run_dir']}/5_skill_radar.png",
        ]
      }
    })


# =========================
# MAINYour account doesn't support custom domain names, so your PythonAnywhere web app will live at RicKanjilal.pythonanywhere.com.
# =========================
if __name__ == "__main__":
    matplotlib.use("Agg")
    app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False, threaded=False)

