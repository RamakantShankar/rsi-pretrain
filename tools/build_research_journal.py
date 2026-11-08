#!/usr/bin/env python3
"""
Mirendil Research Lab Journal Compiler with Inline Text Highlighting & Threaded Discussions.
Translates RESEARCH_JOURNAL.md and annotations.json into a rich, interactive HTML
lab notebook where users can highlight any word/sentence and start conversation threads.
"""

import os
import sys
import json
import shutil
from pathlib import Path
import markdown
import re

def build_journal_html():
    root_dir = Path("M:/Rama/Study/llm/LLM/RSI")
    md_path = root_dir / "RESEARCH_JOURNAL.md"
    ann_path = root_dir / "annotations.json"
    drive_dir = Path("D:/D/Google Drive/RSI_Pretraining_Lab")

    md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else "# Research Journal"
    threads = json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else []

    # Replace mermaid blocks
    mermaid_blocks = []
    def mermaid_repl(match):
        mermaid_blocks.append(match.group(1))
        return f"<!--MERMAID_PLACEHOLDER_{len(mermaid_blocks)-1}-->"

    cleaned_md = re.sub(r'```mermaid\s*\n(.*?)\n```', mermaid_repl, md_text, flags=re.DOTALL)
    html_body = markdown.markdown(cleaned_md, extensions=['tables', 'fenced_code'])

    for idx, block in enumerate(mermaid_blocks):
        html_body = html_body.replace(
            f"<!--MERMAID_PLACEHOLDER_{idx}-->",
            f'<div class="mermaid">\n{block}\n</div>'
        )

    # Render threaded conversation cards
    threads_html = ""
    for thread in threads:
        status_color = "var(--accent-green)" if thread.get("status") == "ANSWERED" else "var(--accent-amber)"
        messages_html = ""
        for msg in thread.get("messages", []):
            is_tutor = msg.get("role") == "tutor"
            bubble_class = "tutor-bubble" if is_tutor else "user-bubble"
            author_label = f"🎓 {msg.get('author')}" if is_tutor else f"👤 {msg.get('author')}"
            messages_html += f"""
            <div class="thread-msg {bubble_class}">
              <div class="msg-header">
                <strong>{author_label}</strong>
                <small>{msg.get('timestamp', '')}</small>
              </div>
              <div class="msg-text">{msg.get('text', '')}</div>
            </div>
            """

        threads_html += f"""
        <div class="thread-card" id="{thread.get('thread_id')}" data-thread-id="{thread.get('thread_id')}">
          <div class="thread-card-header">
            <span class="thread-section">📌 {thread.get('section', 'General')}</span>
            <span class="thread-status" style="border-color:{status_color}; color:{status_color};">{thread.get('status', 'OPEN')}</span>
          </div>
          <div class="thread-quote" onclick="scrollToHighlight('{thread.get('thread_id')}')">
            “{thread.get('highlighted_text', '')}”
          </div>
          <div class="thread-messages">
            {messages_html}
          </div>
          <div class="thread-reply-box">
            <input type="text" placeholder="Reply to this thread..." class="reply-input" id="reply-input-{thread.get('thread_id')}" onkeydown="if(event.key==='Enter') postReply('{thread.get('thread_id')}')">
            <button class="reply-btn" onclick="postReply('{thread.get('thread_id')}')">Reply</button>
          </div>
        </div>
        """

    full_html = f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🔬 Mirendil Research Lab Journal &amp; Inline Discussion Portal</title>
  
  <!-- Highlight.js for Code Highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>

  <!-- MathJax 3 for Crisp LaTeX -->
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
      --highlight-bg: rgba(251, 191, 36, 0.25);
      --highlight-border: #fbbf24;
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
      max-width: 900px;
      margin: 0 auto;
      padding: 3rem 2.5rem;
      user-select: text;
    }}
    #journal-drawer {{
      width: 440px;
      min-width: 440px;
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

    /* Highlight Styling in Document */
    mark.user-highlight {{
      background: var(--highlight-bg);
      color: inherit;
      border-bottom: 2px solid var(--highlight-border);
      border-radius: 2px;
      padding: 0 2px;
      cursor: pointer;
      transition: background 0.2s;
    }}
    mark.user-highlight:hover {{
      background: rgba(251, 191, 36, 0.45);
    }}
    mark.active-highlight {{
      background: rgba(56, 189, 248, 0.4) !important;
      border-bottom: 2px solid var(--accent-blue) !important;
    }}

    /* Floating Tooltip Button */
    #selection-tooltip {{
      display: none;
      position: absolute;
      background: var(--surface);
      border: 1px solid var(--accent-blue);
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      border-radius: 25px;
      padding: 0.35rem 0.85rem;
      z-index: 1000;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--accent-blue);
      align-items: center;
      gap: 0.4rem;
      transition: transform 0.15s ease;
    }}
    #selection-tooltip:hover {{
      transform: translateY(-2px);
      background: var(--card);
    }}

    /* Thread Cards */
    .thread-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.15rem;
      margin-bottom: 1.25rem;
      font-size: 0.88rem;
      transition: border-color 0.2s;
    }}
    .thread-card.focused {{
      border-color: var(--accent-blue);
      box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
    }}
    .thread-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.5rem;
    }}
    .thread-section {{ font-size: 0.75rem; font-weight: 700; color: var(--accent-blue); }}
    .thread-status {{
      font-size: 0.7rem;
      font-weight: 700;
      padding: 0.15rem 0.5rem;
      border-radius: 12px;
      border: 1px solid;
    }}
    .thread-quote {{
      background: rgba(0,0,0,0.3);
      border-left: 3px solid var(--accent-amber);
      padding: 0.5rem 0.75rem;
      border-radius: 0 6px 6px 0;
      font-size: 0.82rem;
      font-style: italic;
      color: var(--text-muted);
      margin-bottom: 0.75rem;
      cursor: pointer;
    }}
    .thread-quote:hover {{ color: #fff; }}
    .thread-messages {{
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
      margin-bottom: 0.75rem;
    }}
    .thread-msg {{
      padding: 0.6rem 0.8rem;
      border-radius: 6px;
      font-size: 0.85rem;
      line-height: 1.5;
    }}
    .user-bubble {{
      background: rgba(255,255,255,0.04);
      border-left: 3px solid #94a3b8;
    }}
    .tutor-bubble {{
      background: rgba(56, 189, 248, 0.08);
      border-left: 3px solid var(--accent-blue);
    }}
    .msg-header {{
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-bottom: 0.2rem;
    }}
    .thread-reply-box {{
      display: flex;
      gap: 0.5rem;
      margin-top: 0.5rem;
    }}
    .reply-input {{
      flex: 1;
      background: rgba(0,0,0,0.3);
      border: 1px solid var(--border);
      color: #fff;
      padding: 0.4rem 0.65rem;
      border-radius: 6px;
      font-size: 0.82rem;
    }}
    .reply-btn {{
      background: var(--accent-blue);
      color: #000;
      border: none;
      padding: 0.4rem 0.75rem;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 700;
      cursor: pointer;
    }}

    /* Main Typography */
    h1 {{ font-size: 2.1rem; color: #fff; margin-bottom: 0.75rem; }}
    h2 {{ font-size: 1.45rem; color: #fff; margin: 2.2rem 0 0.85rem; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; }}
    h3 {{ font-size: 1.15rem; color: var(--accent-blue); margin: 1.5rem 0 0.65rem; }}
    p {{ margin-bottom: 1.15rem; }}
    blockquote {{
      background: var(--surface);
      border-left: 4px solid var(--accent-purple);
      padding: 0.85rem 1.15rem;
      border-radius: 0 8px 8px 0;
      margin: 1.25rem 0;
      font-size: 1rem;
    }}
    table {{ width: 100%; border-collapse: collapse; margin: 1.5rem 0; font-size: 0.9rem; }}
    th, td {{ padding: 0.7rem 0.9rem; border-bottom: 1px solid var(--border); text-align: left; }}
    th {{ background: rgba(255,255,255,0.04); color: #fff; }}

    /* Unrestricted, Responsive Image Styling */
    #journal-main img {{
      max-width: 100%;
      width: auto;
      height: auto;
      display: block;
      margin: 1.75rem auto;
      border-radius: 8px;
      border: 1px solid var(--border);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      cursor: zoom-in;
      transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    #journal-main img:hover {{
      border-color: var(--accent-blue);
      box-shadow: 0 6px 28px rgba(56, 189, 248, 0.3);
    }}
    .image-figure {{
      margin: 2rem 0;
      text-align: center;
      position: relative;
      width: 100%;
    }}
    .image-figure img {{
      margin: 0 auto;
    }}
    .image-caption {{
      margin-top: 0.6rem;
      font-size: 0.85rem;
      color: var(--text-muted);
      font-style: italic;
    }}
    .image-zoom-badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      margin-top: 0.6rem;
      padding: 0.3rem 0.8rem;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--accent-blue);
      background: rgba(56, 189, 248, 0.1);
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 14px;
      cursor: pointer;
      transition: all 0.2s ease;
    }}
    .image-zoom-badge:hover {{
      background: rgba(56, 189, 248, 0.22);
      border-color: var(--accent-blue);
      color: #fff;
      transform: translateY(-1px);
    }}

    /* Lightbox Modal */
    .lightbox-overlay {{
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(4, 7, 13, 0.94);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      z-index: 99999;
      justify-content: center;
      align-items: center;
      padding: 1.5rem;
      opacity: 0;
      transition: opacity 0.2s ease;
    }}
    .lightbox-overlay.active {{
      display: flex;
      opacity: 1;
    }}
    .lightbox-content {{
      max-width: 96vw;
      max-height: 94vh;
      display: flex;
      flex-direction: column;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 25px 60px rgba(0,0,0,0.85);
      overflow: hidden;
      animation: lightboxPop 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    @keyframes lightboxPop {{
      from {{ transform: scale(0.96); opacity: 0; }}
      to {{ transform: scale(1); opacity: 1; }}
    }}
    .lightbox-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.75rem 1.25rem;
      background: rgba(255,255,255,0.02);
      border-bottom: 1px solid var(--border);
      gap: 1rem;
    }}
    .lightbox-caption {{
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 60vw;
    }}
    .lightbox-controls {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
      flex-shrink: 0;
    }}
    .lightbox-btn {{
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.4rem 0.85rem;
      border-radius: 6px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      transition: all 0.15s ease;
    }}
    .lightbox-btn:hover {{
      background: var(--border);
      border-color: var(--accent-blue);
      color: #fff;
    }}
    .lightbox-btn.close-btn {{
      background: rgba(244, 63, 94, 0.15);
      border-color: rgba(244, 63, 94, 0.4);
      color: var(--accent-rose);
    }}
    .lightbox-btn.close-btn:hover {{
      background: var(--accent-rose);
      color: #fff;
    }}
    .lightbox-img-wrapper {{
      overflow: auto;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 1.25rem;
      max-height: calc(94vh - 65px);
      background: #050811;
    }}
    .lightbox-img-wrapper img {{
      max-width: 90vw;
      max-height: 80vh;
      width: auto;
      height: auto;
      object-fit: contain;
      border-radius: 6px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.7);
      cursor: grab;
      transition: transform 0.2s ease;
      margin: 0;
    }}
    .lightbox-img-wrapper img.zoomed {{
      max-width: none;
      max-height: none;
      cursor: zoom-out;
    }}
  </style>
</head>
<body>

  <!-- FLOATING SELECTION TOOLTIP -->
  <div id="selection-tooltip" onclick="openCommentFromSelection()">
    <span>💬 Ask Question / Discuss</span>
  </div>

  <!-- MAIN DOCUMENT -->
  <main id="journal-main" onmouseup="handleTextSelection(event)">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem;">
      <div style="font-size:0.85rem; color:var(--accent-green); font-weight:700;">● LIVE RESEARCH LAB JOURNAL &bull; HIGHLIGHT &amp; DISCUSS ENABLED</div>
      <a href="DASHBOARD.html" target="_blank" style="color:var(--accent-blue); text-decoration:none; font-size:0.85rem; font-weight:600;">📊 Live GPU Dashboard &rarr;</a>
    </div>

    {html_body}
  </main>

  <!-- CONVERSATIONAL ANNOTATIONS DRAWER -->
  <aside id="journal-drawer">
    <div class="drawer-title">
      <span>💬 Discussions ({len(threads)})</span>
      <span style="font-size:0.75rem; color:var(--text-muted);">Highlight text to start new thread</span>
    </div>

    <div id="threads-container">
      {threads_html}
    </div>
  </aside>

  <!-- IMAGE LIGHTBOX MODAL -->
  <div id="image-lightbox" class="lightbox-overlay" onclick="closeLightbox(event)">
    <div class="lightbox-content" onclick="event.stopPropagation()">
      <div class="lightbox-header">
        <span id="lightbox-caption" class="lightbox-caption">Laboratory Visual Asset</span>
        <div class="lightbox-controls">
          <a id="lightbox-external-link" href="#" target="_blank" class="lightbox-btn" title="Open original image in new tab">
            <span>↗ Open in New Tab</span>
          </a>
          <button class="lightbox-btn" onclick="toggleLightboxZoom()" title="Toggle full zoom">
            <span id="lightbox-zoom-text">🔍 100%</span>
          </button>
          <button class="lightbox-btn close-btn" onclick="closeLightbox()" title="Close (Esc)">
            <span>✕ Close</span>
          </button>
        </div>
      </div>
      <div class="lightbox-img-wrapper">
        <img id="lightbox-img" src="" alt="Enlarged view" />
      </div>
    </div>
  </div>

  <script>
    hljs.highlightAll();

    let currentSelectedText = "";

    // --- TEXT SELECTION HANDLER ---
    function handleTextSelection(e) {{
      const selection = window.getSelection();
      const text = selection.toString().trim();
      const tooltip = document.getElementById("selection-tooltip");

      if (text.length > 3) {{
        currentSelectedText = text;
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        
        tooltip.style.left = (rect.left + window.scrollX + rect.width / 2 - 80) + "px";
        tooltip.style.top = (rect.top + window.scrollY - 42) + "px";
        tooltip.style.display = "flex";
      }} else {{
        setTimeout(() => {{
          if (window.getSelection().toString().trim().length === 0) {{
            tooltip.style.display = "none";
          }}
        }}, 200);
      }}
    }}

    // --- CREATE COMMENT THREAD FROM SELECTION ---
    function openCommentFromSelection() {{
      const text = currentSelectedText;
      if (!text) return;

      const userQuestion = prompt(`💬 Ask Question about selected text:\n\n"${{text}}"\n\nType your question or observation:`);
      if (!userQuestion || !userQuestion.trim()) return;

      const newThreadId = "thread_" + Date.now().toString().slice(-4);
      const newThread = {{
        thread_id: newThreadId,
        section: "User Highlight",
        highlighted_text: text,
        created_at: new Date().toLocaleString('en-US', {{ timeZone: 'America/Los_Angeles' }}) + ' PST',
        status: "OPEN",
        messages: [
          {{
            author: "Ramakant",
            role: "researcher",
            text: userQuestion.trim(),
            timestamp: new Date().toLocaleString('en-US', {{ timeZone: 'America/Los_Angeles' }}) + ' PST'
          }}
        ]
      }};

      // Save to localStorage
      let localThreads = JSON.parse(localStorage.getItem('research_threads') || '[]');
      localThreads.unshift(newThread);
      localStorage.setItem('research_threads', JSON.stringify(localThreads));

      alert('Question posted! Antigravity (Tutor) and Chronicler subagent will review it in annotations.json and provide answers.');
      location.reload();
    }}

    // --- POST REPLY TO THREAD ---
    function postReply(threadId) {{
      const input = document.getElementById('reply-input-' + threadId);
      const replyText = input.value.trim();
      if (!replyText) return;

      const replyMsg = {{
        author: "Ramakant",
        role: "researcher",
        text: replyText,
        timestamp: new Date().toLocaleString('en-US', {{ timeZone: 'America/Los_Angeles' }}) + ' PST'
      }};

      // Store reply
      let threadReplies = JSON.parse(localStorage.getItem('thread_replies_' + threadId) || '[]');
      threadReplies.push(replyMsg);
      localStorage.setItem('thread_replies_' + threadId, JSON.stringify(threadReplies));

      input.value = '';
      alert('Reply added to thread! Tutor is notified.');
      location.reload();
    }}

    function scrollToHighlight(threadId) {{
      const card = document.getElementById(threadId);
      if (card) {{
        card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        card.classList.add('focused');
        setTimeout(() => card.classList.remove('focused'), 2000);
      }}
    }}

    // --- LIGHTBOX ENGINE FOR HIGH-RES IMAGE INSPECTION ---
    function initLightbox() {{
      const images = document.querySelectorAll('#journal-main img');
      images.forEach((img, idx) => {{
        img.title = img.title || "Click to expand image in full resolution";
        img.style.cursor = "zoom-in";
        img.onclick = () => openLightbox(img.src, img.alt || img.title || `Figure ${{idx + 1}}`);

        if (!img.closest('.image-figure')) {{
          const figure = document.createElement('div');
          figure.className = 'image-figure';
          img.parentNode.insertBefore(figure, img);
          figure.appendChild(img);

          const badge = document.createElement('button');
          badge.type = 'button';
          badge.className = 'image-zoom-badge';
          badge.innerHTML = '🔍 Click to expand full size';
          badge.onclick = (e) => {{
            e.stopPropagation();
            openLightbox(img.src, img.alt || img.title || `Figure ${{idx + 1}}`);
          }};
          figure.appendChild(badge);

          if (img.alt && img.alt !== 'Figure') {{
            const cap = document.createElement('div');
            cap.className = 'image-caption';
            cap.textContent = img.alt;
            figure.appendChild(cap);
          }}
        }}
      }});
    }}

    let isZoomed = false;

    function openLightbox(src, caption) {{
      const overlay = document.getElementById('image-lightbox');
      const img = document.getElementById('lightbox-img');
      const cap = document.getElementById('lightbox-caption');
      const link = document.getElementById('lightbox-external-link');
      const zoomText = document.getElementById('lightbox-zoom-text');

      img.src = src;
      img.classList.remove('zoomed');
      isZoomed = false;
      zoomText.textContent = '🔍 100%';
      cap.textContent = caption || 'Laboratory Visual Asset';
      link.href = src;

      overlay.classList.add('active');
      document.body.style.overflow = 'hidden';
    }}

    function closeLightbox(e) {{
      if (e && e.target && e.target.closest && e.target.closest('.lightbox-content')) return;
      const overlay = document.getElementById('image-lightbox');
      overlay.classList.remove('active');
      document.body.style.overflow = '';
    }}

    function toggleLightboxZoom() {{
      const img = document.getElementById('lightbox-img');
      const zoomText = document.getElementById('lightbox-zoom-text');
      isZoomed = !isZoomed;
      if (isZoomed) {{
        img.classList.add('zoomed');
        zoomText.textContent = '🔍 Fit Window';
      }} else {{
        img.classList.remove('zoomed');
        zoomText.textContent = '🔍 100%';
      }}
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') {{
        closeLightbox();
      }}
    }});

    // Initialize lightbox on DOM load
    window.addEventListener('DOMContentLoaded', () => {{
      initLightbox();
    }});
    // Also run immediately
    initLightbox();
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
