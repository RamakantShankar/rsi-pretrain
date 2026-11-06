#!/usr/bin/env python3
"""
Mirendil Laboratory - Publication HTML Compiler
Translates RESEARCH_COMPENDIUM.md and EXPERIMENT_LOG.md into standalone,
publication-grade HTML documents with interactive Mermaid vector diagrams,
MathJax LaTeX equations, and GitHub Markdown typography.

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
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  
  <!-- GitHub Markdown CSS -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css" />
  
  <!-- Highlight.js for Code Highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css" />
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
    mermaid.initialize({{
      startOnLoad: true,
      theme: 'default',
      securityLevel: 'loose',
      flowchart: {{
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      }}
    }});
  </script>

  <style>
    body {{
      box-sizing: border-box;
      min-width: 200px;
      max-width: 1100px;
      margin: 0 auto;
      padding: 45px 30px;
      background-color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
    }}
    .markdown-body {{
      background-color: #ffffff;
      padding: 48px;
      border-radius: 12px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
    }}
    .mermaid {{
      background-color: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 20px;
      margin: 24px 0;
      display: flex;
      justify-content: center;
      overflow-x: auto;
    }}
    .lab-banner {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 2px solid #e2e8f0;
      padding-bottom: 16px;
      margin-bottom: 32px;
      font-size: 0.875rem;
      color: #64748b;
    }}
    .lab-badge {{
      display: inline-block;
      padding: 4px 12px;
      background: #e0e7ff;
      color: #3730a3;
      border-radius: 9999px;
      font-weight: 600;
      font-size: 0.75rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    @media (max-width: 767px) {{
      body {{
        padding: 15px;
      }}
      .markdown-body {{
        padding: 20px;
      }}
    }}
    @media print {{
      body {{
        background: none;
        max-width: 100%;
        padding: 0;
      }}
      .markdown-body {{
        border: none;
        box-shadow: none;
        padding: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class="lab-banner">
    <div>
      <span class="lab-badge">Publication Preview</span>
      <strong>Mirendil Laboratory</strong> &bull; Recursive Self-Improvement Research
    </div>
    <div>Target: NVIDIA DGX Grace Blackwell GB10</div>
  </div>

  <article class="markdown-body">
    {content}
  </article>

  <script>
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
    # Match display math $$...$$ and inline math $...$
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
        # Unescape HTML entities (e.g. &gt; -> >, &lt; -> <, &amp; -> &)
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
            print(f"Mirrored to Google Drive -> {gdrive_out}")
        else:
            print(f"Google Drive mirror directory not found: {gdrive_dir}")

if __name__ == "__main__":
    main()
