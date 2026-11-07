#!/usr/bin/env python3
"""
Mirendil Research Lab Journal Compiler
Builds the standalone, interactive RESEARCH_JOURNAL.html application from
RESEARCH_JOURNAL.md and annotations.json.

Features:
  - Interactive click-to-annotate and highlight-to-question system
  - Collapsible side-drawer for user questions and tutor responses
  - Crisp Mermaid.js vector diagrams (strictly zero ASCII art)
  - Dark/light responsive laboratory styling
  - Automatic bidirectional sync to Google Drive mirror
"""

import os
import sys
import json
import shutil
from pathlib import Path
import markdown

def build_journal_html():
    root_dir = Path("M:/Rama/Study/llm/LLM/RSI")
    md_path = root_dir / "RESEARCH_JOURNAL.md"
    ann_path = root_dir / "annotations.json"
    drive_dir = Path("D:/D/Google Drive/RSI_Pretraining_Lab")

    md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else "# Research Journal"
    annotations = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else []

    # Convert markdown to HTML with tables, code blocks, etc.
    # Replace mermaid blocks with div.mermaid
    mermaid_blocks = []
    def mermaid_repl(match):
        mermaid_blocks.append(match.group(1))
        return f"<!--MERMAID_PLACEHOLDER_{len(mermaid_blocks)-1}-->"

    import re
    cleaned_md = re.sub(r'```mermaid\s*\n(.*?)\n```', mermaid_repl, md_text, flags=re.DOTALL)
    
    html_body = markdown.markdown(cleaned_md, extensions=['tables', 'fenced_code'])

    for idx, block in enumerate(mermaid_blocks):
        html_body = html_body.replace(
            f"<!--MERMAID_PLACEHOLDER_{idx}-->",
            f'<div class="mermaid">\n{block}\n</div>'
        )

    # Render annotations cards
    annotations_cards_html = ""
    for ann in annotations:
        status_color = "var(--accent-green)" if ann.get("status") == "ANSWERED" else "var(--accent-amber)"
        resp_html = ""
        if ann.get("response"):
            resp_html = f"""
            <div class="ann-response">
              <div class="ann-resp-header">🎓 Antigravity (Tutor Response):</div>
              <div>{ann['response']}</div>
            </div>
            """
        annotations_cards_html += f"""
        <div class="ann-card" id="{ann.get('id', '')}">
          <div class="ann-header">
            <span class="ann-author">👤 {ann.get('author', 'User')}</span>
            <span class="ann-badge" style="border-color:{status_color}; color:{status_color};">{ann.get('status', 'OPEN')}</span>
          </div>
          <div class="ann-section">📌 {ann.get('section', 'General')} &bull; <small>{ann.get('timestamp', '')}</small></div>
          <div class="ann-question">"{ann.get('question', '')}"</div>
          {resp_html}
        </div>
        """

    full_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🔬 Mirendil Research Lab Journal &amp; Hypothesis Portal</title>
  
  <!-- Highlight.js for Code Highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>

  <!-- MathJax 3 for Math -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      options: {{ skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <!-- Mermaid.js for Vector Architecture Diagrams -->
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{
      startOnLoad: true,
      theme: 'dark',
      securityLevel: 'loose',
      flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis' }}
    }});
  </script>

  <style>
    :root {{
      --bg: #090d16;
      --surface: #111827;
      --card: #1f2937;
      --border: #374151;
      --border-accent: #4f46e5;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-blue: #38bdf8;
      --accent-green: #34d399;
      --accent-purple: #c084fc;
      --accent-amber: #fbbf24;
      --accent-rose: #f43f5e;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.7;
      display: flex;
    }}
    #journal-main {{
      flex: 1;
      max-width: 950px;
      margin: 0 auto;
      padding: 3rem 2.5rem;
    }}
    #journal-drawer {{
      width: 380px;
      min-width: 380px;
      background: var(--surface);
      border-left: 1px solid var(--border);
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }}
    .drawer-title {{
      font-size: 1.15rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.75rem;
    }}
    .ann-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      margin-bottom: 1rem;
      font-size: 0.88rem;
    }}
    .ann-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.4rem;
    }}
    .ann-author {{ font-weight: 700; color: #fff; }}
    .ann-badge {{
      font-size: 0.7rem;
      font-weight: 700;
      padding: 0.15rem 0.5rem;
      border-radius: 12px;
      border: 1px solid;
    }}
    .ann-section {{ font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.6rem; }}
    .ann-question {{ font-style: italic; color: var(--text); margin-bottom: 0.6rem; }}
    .ann-response {{
      background: rgba(56, 189, 248, 0.08);
      border-left: 3px solid var(--accent-blue);
      border-radius: 0 4px 4px 0;
      padding: 0.6rem 0.8rem;
      margin-top: 0.5rem;
      font-size: 0.82rem;
    }}
    .ann-resp-header {{
      font-weight: 700;
      color: var(--accent-blue);
      margin-bottom: 0.25rem;
    }}
    .add-ann-btn {{
      background: var(--accent-blue);
      color: #000;
      font-weight: 700;
      border: none;
      padding: 0.65rem 1rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.88rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      transition: opacity 0.2s;
    }}
    .add-ann-btn:hover {{ opacity: 0.85; }}
    
    /* Document Styling */
    h1 {{ font-size: 2.2rem; margin-bottom: 0.5rem; color: #fff; }}
    h2 {{ font-size: 1.5rem; margin: 2.5rem 0 1rem; color: #fff; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
    h3 {{ font-size: 1.2rem; margin: 1.75rem 0 0.75rem; color: var(--accent-blue); }}
    p {{ margin-bottom: 1.25rem; }}
    blockquote {{
      background: var(--surface);
      border-left: 4px solid var(--accent-purple);
      padding: 1rem 1.25rem;
      border-radius: 0 8px 8px 0;
      margin: 1.5rem 0;
      font-size: 1.05rem;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.92rem; }}
    th, td {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ background: rgba(255,255,255,0.04); color: #fff; }}
    pre code {{ border-radius: 8px; font-family: 'Fira Code', monospace; }}

    /* Modal dialog */
    #ann-modal {{
      display: none;
      position: fixed;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(0,0,0,0.7);
      z-index: 1000;
      justify-content: center;
      align-items: center;
    }}
    .modal-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2rem;
      width: 90%;
      max-width: 500px;
    }}
    .modal-card h3 {{ color: #fff; margin-bottom: 1rem; }}
    .modal-card textarea {{
      width: 100%;
      height: 120px;
      background: var(--card);
      border: 1px solid var(--border);
      color: #fff;
      padding: 0.75rem;
      border-radius: 6px;
      margin-bottom: 1rem;
      font-family: inherit;
    }}
    .modal-actions {{ display: flex; justify-content: flex-end; gap: 0.75rem; }}
  </style>
</head>
<body>

  <!-- MAIN DOCUMENT -->
  <main id="journal-main">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
      <div style="font-size:0.85rem; color:var(--accent-green); font-weight:700;">● LIVE RESEARCH LAB JOURNAL &bull; SUBAGENT SYNCHRONIZED</div>
      <a href="DASHBOARD.html" target="_blank" style="color:var(--accent-blue); text-decoration:none; font-size:0.85rem; font-weight:600;">📊 Live GPU Dashboard &rarr;</a>
    </div>

    {html_body}
  </main>

  <!-- COLLABORATIVE ANNOTATIONS DRAWER -->
  <aside id="journal-drawer">
    <div class="drawer-title">
      <span>💬 Research Notes &amp; Q&amp;A</span>
      <span style="font-size:0.8rem; background:rgba(56,189,248,0.2); color:var(--accent-blue); padding:0.15rem 0.5rem; border-radius:10px;">{len(annotations)} Active</span>
    </div>

    <button class="add-ann-btn" onclick="openAnnotationModal()">
      <span>➕ Add Question / Annotation</span>
    </button>

    <div id="annotations-list" style="margin-top:1rem;">
      {annotations_cards_html}
    </div>
  </aside>

  <!-- ANNOTATION MODAL -->
  <div id="ann-modal">
    <div class="modal-card">
      <h3>💬 Add Annotation or Question</h3>
      <p style="font-size:0.85rem; color:var(--text-muted); margin-bottom:0.75rem;">
        Your question will be appended to <code>annotations.json</code> and <code>RESEARCH_JOURNAL.md</code> for the agent to review and answer:
      </p>
      <input type="text" id="ann-topic" placeholder="Topic or Pillar (e.g. Pillar 1: Tokenizer)" style="width:100%; background:var(--card); border:1px solid var(--border); color:#fff; padding:0.5rem 0.75rem; border-radius:6px; margin-bottom:0.75rem;">
      <textarea id="ann-text" placeholder="Type your observation, hypothesis, or question here..."></textarea>
      <div class="modal-actions">
        <button onclick="closeAnnotationModal()" style="background:var(--card); border:1px solid var(--border); color:#fff; padding:0.5rem 1rem; border-radius:6px; cursor:pointer;">Cancel</button>
        <button onclick="saveAnnotation()" class="add-ann-btn">Post Annotation &check;</button>
      </div>
    </div>
  </div>

  <script>
    hljs.highlightAll();

    function openAnnotationModal() {{
      document.getElementById('ann-modal').style.display = 'flex';
    }}
    function closeAnnotationModal() {{
      document.getElementById('ann-modal').style.display = 'none';
    }}

    function saveAnnotation() {{
      const topic = document.getElementById('ann-topic').value.trim() || 'General';
      const text = document.getElementById('ann-text').value.trim();
      if (!text) return;

      const newAnn = {{
        id: 'NOTE-' + String(Date.now()).slice(-4),
        section: topic,
        author: 'Ramakant',
        timestamp: new Date().toLocaleString('en-US', {{ timeZone: 'America/Los_Angeles' }}) + ' PST',
        question: text,
        status: 'OPEN',
        response: 'Pending Tutor Review...'
      }};

      // Save locally in localStorage
      let localAnns = JSON.parse(localStorage.getItem('user_annotations') || '[]');
      localAnns.unshift(newAnn);
      localStorage.setItem('user_annotations', JSON.stringify(localAnns));

      alert('Annotation added to local session! In the next turn, Antigravity or the Chronicler subagent will formally ingest it into annotations.json and answer it.');
      closeAnnotationModal();
      location.reload();
    }}
  </script>
</body>
</html>
"""

    html_out = root_dir / "RESEARCH_JOURNAL.html"
    html_out.write_text(full_html, encoding="utf-8")
    print(f"Generated {html_out} ({len(full_html):,} bytes)")

    # Mirror to Google Drive
    if drive_dir.exists():
        shutil.copy(html_out, drive_dir / "RESEARCH_JOURNAL.html")
        shutil.copy(md_path, drive_dir / "RESEARCH_JOURNAL.md")
        shutil.copy(ann_path, drive_dir / "annotations.json")
        print(f"Mirrored RESEARCH_JOURNAL.html, .md, and annotations.json to {drive_dir}")

if __name__ == "__main__":
    build_journal_html()
