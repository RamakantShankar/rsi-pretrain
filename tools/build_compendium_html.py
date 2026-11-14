#!/usr/bin/env python3
"""
Mirendil Laboratory - Publication HTML Compiler
Translates RESEARCH_COMPENDIUM.md and EXPERIMENT_LOG.md into standalone,
publication-grade HTML documents with interactive Mermaid vector diagrams,
MathJax LaTeX equations, high-contrast typography, and full widescreen layout.

Outputs to both local repository and Google Drive mirror.
"""

import sys
import os
import re
import html
import shutil
from pathlib import Path
import markdown

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  
  <!-- Highlight.js for Code Highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css" id="hljs-light" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css" id="hljs-dark" disabled />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/sql.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/json.min.js"></script>
  
  <!-- MathJax 3 for Publication Math / LaTeX -->
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <!-- Mermaid.js for Vector Architecture Diagrams -->
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    mermaid.initialize({{
      startOnLoad: true,
      theme: isDark ? 'dark' : 'default',
      securityLevel: 'loose',
      flowchart: {{
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      }}
    }});
  </script>

  <style>
    :root {{
      --body-bg: #f8fafc;
      --card-bg: #ffffff;
      --text-main: #0f172a;
      --text-muted: #475569;
      --border-color: #cbd5e1;
      --border-light: #e2e8f0;
      --table-header-bg: #f1f5f9;
      --table-row-alt: #f8fafc;
      --code-bg: #f1f5f9;
      --accent: #4338ca;
      --accent-badge: #e0e7ff;
      --accent-text: #3730a3;
      --banner-bg: #f8fafc;
    }}

    [data-theme="dark"] {{
      --body-bg: #090d16;
      --card-bg: #0f172a;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --border-color: #334155;
      --border-light: #1e293b;
      --table-header-bg: #1e293b;
      --table-row-alt: #131d2e;
      --code-bg: #1e293b;
      --accent: #6366f1;
      --accent-badge: #312e81;
      --accent-text: #c7d2fe;
      --banner-bg: #0f172a;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      min-width: 320px;
      width: 95%;
      max-width: 1560px;
      margin: 0 auto;
      padding: 32px 24px;
      background-color: var(--body-bg);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.65;
      font-size: 16px;
      transition: background-color 0.2s ease, color 0.2s ease;
    }}

    .lab-banner {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      border: 1px solid var(--border-light);
      border-radius: 10px;
      padding: 14px 24px;
      margin-bottom: 24px;
      font-size: 0.9rem;
      color: var(--text-muted);
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}

    .lab-badge {{
      display: inline-block;
      padding: 4px 12px;
      background: var(--accent-badge);
      color: var(--accent-text);
      border-radius: 9999px;
      font-weight: 700;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      margin-right: 8px;
    }}

    .theme-btn {{
      background: var(--table-header-bg);
      border: 1px solid var(--border-color);
      color: var(--text-main);
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}

    .theme-btn:hover {{
      background: var(--border-light);
    }}

    .markdown-body {{
      background-color: var(--card-bg);
      padding: 50px 60px;
      border-radius: 12px;
      border: 1px solid var(--border-light);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
      color: var(--text-main);
    }}

    /* Typography Overrides for Crystal Contrast */
    .markdown-body h1, .markdown-body h2, .markdown-body h3, 
    .markdown-body h4, .markdown-body h5, .markdown-body h6 {{
      color: var(--text-main);
      font-weight: 700;
      margin-top: 1.8em;
      margin-bottom: 0.8em;
      line-height: 1.3;
    }}
    .markdown-body h1 {{
      font-size: 2.2rem;
      border-bottom: 2px solid var(--border-color);
      padding-bottom: 0.4em;
    }}
    .markdown-body h2 {{
      font-size: 1.7rem;
      border-bottom: 1px solid var(--border-light);
      padding-bottom: 0.3em;
    }}
    .markdown-body h3 {{ font-size: 1.35rem; }}
    .markdown-body h4 {{ font-size: 1.15rem; }}

    .markdown-body p, .markdown-body li, .markdown-body blockquote {{
      color: var(--text-main);
      font-size: 1.02rem;
      line-height: 1.75;
    }}

    .markdown-body strong {{
      color: var(--text-main);
      font-weight: 700;
    }}

    .markdown-body a {{
      color: var(--accent);
      text-decoration: underline;
      text-underline-offset: 3px;
    }}

    .markdown-body hr {{
      height: 2px;
      background-color: var(--border-light);
      border: none;
      margin: 40px 0;
    }}

    /* Widescreen Responsive Tables */
    .markdown-body table {{
      width: 100% !important;
      border-collapse: collapse !important;
      margin: 28px 0 !important;
      display: table !important;
      overflow-x: auto;
      font-size: 0.95rem;
    }}

    .markdown-body table th {{
      background-color: var(--table-header-bg) !important;
      color: var(--text-main) !important;
      font-weight: 700 !important;
      border: 1px solid var(--border-color) !important;
      padding: 12px 16px !important;
      text-align: left;
    }}

    .markdown-body table td {{
      border: 1px solid var(--border-color) !important;
      padding: 11px 16px !important;
      color: var(--text-main) !important;
    }}

    .markdown-body table tr:nth-child(2n) {{
      background-color: var(--table-row-alt) !important;
    }}

    /* Code Blocks */
    .markdown-body pre {{
      background-color: var(--code-bg) !important;
      border: 1px solid var(--border-color) !important;
      border-radius: 8px;
      padding: 18px 22px;
      overflow-x: auto;
      font-size: 0.92rem;
    }}

    .markdown-body code {{
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
      padding: 2px 6px;
      border-radius: 4px;
      background-color: var(--code-bg);
      color: var(--text-main);
      font-size: 0.9em;
    }}

    .markdown-body pre code {{
      background-color: transparent !important;
      padding: 0;
      color: inherit;
    }}

    /* MathJax High Contrast Equations */
    mjx-container {{
      color: var(--text-main) !important;
      font-size: 1.15em !important;
      padding: 8px 0;
    }}
    mjx-container[jax="CHTML"][display="true"] {{
      margin: 1.2em 0 !important;
    }}

    /* Mermaid Diagrams Full Width */
    .mermaid {{
      background-color: var(--card-bg) !important;
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 28px;
      margin: 32px 0;
      display: flex;
      justify-content: center;
      overflow-x: auto;
      box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }}

    @media (max-width: 900px) {{
      body {{
        width: 100%;
        padding: 16px 12px;
      }}
      .markdown-body {{
        padding: 24px 20px;
      }}
      .lab-banner {{
        flex-direction: column;
        align-items: flex-start;
        gap: 12px;
      }}
    }}
  </style>
</head>
<body>
  <div class="lab-banner">
    <div>
      <span class="lab-badge">Executive Publication</span>
      <strong>Mirendil Laboratory</strong> &bull; Recursive Self-Improvement Research
    </div>
    <div style="display: flex; align-items: center; gap: 18px;">
      <span>Target: <strong>NVIDIA DGX Grace Blackwell GB10</strong></span>
      <button id="themeToggle" class="theme-btn" onclick="toggleTheme()">☀️ Light Mode</button>
    </div>
  </div>

  <article class="markdown-body">
    {content}
  </article>

  <script>
    function applyTheme(theme) {{
      document.documentElement.setAttribute('data-theme', theme);
      const btn = document.getElementById('themeToggle');
      const hljsLight = document.getElementById('hljs-light');
      const hljsDark = document.getElementById('hljs-dark');
      
      if (btn) {{
        btn.innerHTML = theme === 'dark' ? '🌙 Dark Mode' : '☀️ Light Mode';
      }}
      if (hljsLight && hljsDark) {{
        if (theme === 'dark') {{
          hljsLight.disabled = true;
          hljsDark.disabled = false;
        }} else {{
          hljsLight.disabled = false;
          hljsDark.disabled = true;
        }}
      }}
      localStorage.setItem('mirendil-theme', theme);
    }}

    function toggleTheme() {{
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      applyTheme(current === 'dark' ? 'light' : 'dark');
    }}

    // Default to crisp light paper theme unless user explicitly chose dark
    const savedTheme = localStorage.getItem('mirendil-theme') || 'light';
    applyTheme(savedTheme);

    document.addEventListener('DOMContentLoaded', (event) => {{
      document.querySelectorAll('pre code').forEach((block) => {{
        if (!block.classList.contains('language-mermaid')) {{
          hljs.highlightElement(block);
        }}
      }});
    }});
  </script>
</body>
</html>
"""

def compile_markdown_to_html(md_path: Path, output_path: Path):
    print(f"Reading: {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-process: protect LaTeX formulas from markdown parser
    math_blocks = []
    def save_math(match):
        idx = len(math_blocks)
        math_blocks.append(match.group(0))
        return f"%%MATHBLOCK_{idx}%%"

    # Replace display math first, then inline math
    md_text = re.sub(r"\$\$[\s\S]*?\$\$", save_math, md_text)
    md_text = re.sub(r"(?<!\\)\$(?!\$)[^\n]+?(?<!\\)\$", save_math, md_text)

    # Convert markdown to HTML
    extensions = ["tables", "fenced_code", "toc", "sane_lists"]
    body_html = markdown.markdown(md_text, extensions=extensions)

    # Restore math blocks
    for idx, block in enumerate(math_blocks):
        body_html = body_html.replace(f"%%MATHBLOCK_{idx}%%", block)

    # Convert mermaid code blocks into <div class="mermaid">
    def transform_mermaid(match):
        inner_code = match.group(1)
        unescaped = html.unescape(inner_code).strip()
        return f'<div class="mermaid">\n{unescaped}\n</div>'

    body_html = re.sub(
        r'<pre><code class="language-mermaid">([\s\S]*?)</code></pre>',
        transform_mermaid,
        body_html
    )

    # Extract title from first H1 if present
    h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", body_html)
    title = h1_match.group(1) if h1_match else md_path.stem.replace("_", " ").title()

    full_html = HTML_TEMPLATE.format(title=title, content=body_html)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Compiled successfully -> {output_path} ({len(full_html):,} bytes)")

def main():
    root = Path(__file__).resolve().parent.parent
    gdrive_dir = Path("D:/D/Google Drive/RSI_Pretraining_Lab")

    files_to_compile = [
        "RESEARCH_COMPENDIUM.md",
        "EXPERIMENT_LOG.md",
        "RESEARCH_JOURNAL.md",
    ]

    for fname in files_to_compile:
        src = root / fname
        if not src.exists():
            print(f"Warning: {src} not found, skipping.")
            continue

        html_name = src.stem + ".html"
        local_out = root / html_name
        compile_markdown_to_html(src, local_out)

        if gdrive_dir.exists():
            gdrive_out = gdrive_dir / html_name
            shutil.copy2(local_out, gdrive_out)
            print(f"Mirrored HTML to Google Drive -> {gdrive_out}")
            gdrive_md = gdrive_dir / fname
            shutil.copy2(src, gdrive_md)
            print(f"Mirrored Markdown to Google Drive -> {gdrive_md}")
            if fname == "EXPERIMENT_LOG.md":
                shutil.copy2(src, gdrive_dir / "EXPERIMENT_AND_ABLATION_LOG.md")
                print(f"Mirrored EXPERIMENT_AND_ABLATION_LOG.md to Google Drive")
        else:
            print(f"Google Drive mirror directory not found: {gdrive_dir}")

    # Mirror dashboard and annotations
    if gdrive_dir.exists():
        for extra in ["DASHBOARD.html", "annotations.json"]:
            extra_src = root / extra
            if extra_src.exists():
                shutil.copy2(extra_src, gdrive_dir / extra)
                print(f"Mirrored {extra} to Google Drive")

if __name__ == "__main__":
    main()
