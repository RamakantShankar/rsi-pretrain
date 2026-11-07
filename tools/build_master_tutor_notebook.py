#!/usr/bin/env python3
"""
Mirendil Laboratory - Interactive Master Tutor Notebook Generator
Compiles an interactive, standalone HTML Master Learning Notebook covering
the entire Foundation Model & Recursive Self-Improvement curriculum.

Features:
  - Widescreen responsive layout with collapsible navigation and reading progress
  - Live interactive calculation widgets (RMSNorm sandbox, RoPE clock, SwiGLU gate, GQA memory, Binomial guard)
  - Pencil & paper exercises with tiny numbers and step-by-step arithmetic
  - Zero-KaTeX bulletproof HTML math and MathJax 3 rendering
  - Interactive self-check quizzes with instant revealable answers
  - Clickable links to all workspace source files and Jupyter notebooks
  - Auto-mirrors to Google Drive and local workspace
"""

import sys
import os
import shutil
from pathlib import Path

def generate_tutor_html() -> str:
    html_content = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🎓 RSI Master Tutor Notebook: Foundation Pretraining to Autonomous Self-Play</title>
  
  <!-- Highlight.js for Syntax Highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css" id="hljs-theme" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/json.min.js"></script>
  
  <!-- MathJax 3 for Crisp Mathematical Typesetting -->
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      },
      options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <!-- Mermaid.js for Vector Architecture Flowcharts -->
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({
      startOnLoad: true,
      theme: 'dark',
      securityLevel: 'loose',
      flowchart: { useMaxWidth: true, htmlLabels: true, curve: 'basis' }
    });
  </script>

  <style>
    :root {
      --bg: #0b0f19;
      --surface: #111827;
      --surface-card: #1e293b;
      --border: #334155;
      --border-light: #1e293b;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --text-bright: #ffffff;
      --accent-blue: #38bdf8;
      --accent-indigo: #6366f1;
      --accent-green: #34d399;
      --accent-purple: #c084fc;
      --accent-amber: #fbbf24;
      --accent-rose: #f43f5e;
      --code-bg: #0f172a;
    }

    [data-theme="light"] {
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-card: #f1f5f9;
      --border: #cbd5e1;
      --border-light: #e2e8f0;
      --text: #1e293b;
      --text-muted: #64748b;
      --text-bright: #0f172a;
      --accent-blue: #0284c7;
      --accent-indigo: #4f46e5;
      --accent-green: #059669;
      --accent-purple: #7c3aed;
      --accent-amber: #d97706;
      --accent-rose: #e11d48;
      --code-bg: #f8fafc;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.65;
      display: flex;
      min-height: 100vh;
    }

    /* Fixed Sidebar Navigation */
    #sidebar {
      width: 320px;
      min-width: 320px;
      background: var(--surface);
      border-right: 1px solid var(--border);
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
      padding: 1.5rem 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      z-index: 50;
    }

    .brand-title {
      font-size: 1.15rem;
      font-weight: 800;
      color: var(--text-bright);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .brand-subtitle {
      font-size: 0.8rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }

    .nav-group-title {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent-blue);
      font-weight: 700;
      margin-bottom: 0.5rem;
    }

    .nav-links {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .nav-links a {
      color: var(--text);
      text-decoration: none;
      font-size: 0.88rem;
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: all 0.15s ease;
    }

    .nav-links a:hover {
      background: var(--surface-card);
      color: var(--accent-blue);
      transform: translateX(3px);
    }

    .nav-links a.active {
      background: rgba(56, 189, 248, 0.12);
      color: var(--accent-blue);
      font-weight: 600;
      border-left: 3px solid var(--accent-blue);
    }

    .progress-box {
      background: var(--surface-card);
      padding: 1rem;
      border-radius: 8px;
      border: 1px solid var(--border);
    }

    .progress-bar-bg {
      background: rgba(255,255,255,0.08);
      height: 8px;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 0.5rem;
    }

    .progress-bar-fill {
      background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
      height: 100%;
      width: 17%;
      transition: width 0.3s;
    }

    /* Main Content Area */
    #main-content {
      flex: 1;
      padding: 2.5rem 3.5rem;
      max-width: 1100px;
      overflow-x: hidden;
    }

    /* Top Sticky Action Bar */
    .top-action-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2.5rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }

    .pill-links {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    .pill-btn {
      background: var(--surface-card);
      border: 1px solid var(--border);
      color: var(--text);
      text-decoration: none;
      padding: 0.45rem 0.9rem;
      border-radius: 20px;
      font-size: 0.82rem;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      transition: all 0.2s;
    }

    .pill-btn:hover {
      border-color: var(--accent-blue);
      color: var(--accent-blue);
      transform: translateY(-1px);
    }

    .theme-toggle-btn {
      background: var(--surface-card);
      border: 1px solid var(--border);
      color: var(--text);
      cursor: pointer;
      padding: 0.45rem 0.9rem;
      border-radius: 20px;
      font-size: 0.82rem;
      font-weight: 600;
    }

    /* Module Section Styling */
    .module-section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 2.5rem;
      margin-bottom: 3.5rem;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }

    .module-header {
      border-bottom: 2px solid var(--border);
      padding-bottom: 1.25rem;
      margin-bottom: 2rem;
    }

    .module-badge {
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      padding: 0.25rem 0.75rem;
      border-radius: 12px;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent-blue);
      margin-bottom: 0.75rem;
    }

    h1 { font-size: 2.2rem; color: var(--text-bright); margin-bottom: 0.75rem; }
    h2 { font-size: 1.6rem; color: var(--text-bright); margin: 2rem 0 1rem; }
    h3 { font-size: 1.25rem; color: var(--accent-blue); margin: 1.5rem 0 0.75rem; }
    h4 { font-size: 1.05rem; color: var(--accent-purple); margin: 1.25rem 0 0.5rem; }
    p { margin-bottom: 1.15rem; color: var(--text); font-size: 1rem; }

    /* Interactive Widget Containers */
    .widget-container {
      background: var(--surface-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.75rem;
      margin: 1.75rem 0;
    }

    .widget-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.75rem;
    }

    .widget-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-bright);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Hand-On Pencil & Paper Card */
    .pencil-paper-card {
      background: #0f172a;
      border: 2px dashed var(--accent-amber);
      border-radius: 10px;
      padding: 1.75rem;
      margin: 2rem 0;
    }

    .pencil-header {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: var(--accent-amber);
      font-size: 1.15rem;
      font-weight: 700;
      margin-bottom: 1rem;
    }

    .step-box {
      background: rgba(255, 255, 255, 0.03);
      border-left: 3px solid var(--accent-amber);
      padding: 0.85rem 1.25rem;
      margin-bottom: 0.85rem;
      border-radius: 0 6px 6px 0;
    }

    /* Proactive Trap / Gotcha Alert */
    .gotcha-box {
      background: rgba(244, 63, 94, 0.08);
      border: 1px solid rgba(244, 63, 94, 0.3);
      border-left: 4px solid var(--accent-rose);
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      margin: 1.75rem 0;
    }

    .gotcha-title {
      color: var(--accent-rose);
      font-weight: 700;
      font-size: 0.95rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.5rem;
    }

    /* Interactive Quiz Section */
    .quiz-box {
      background: rgba(99, 102, 241, 0.08);
      border: 1px solid rgba(99, 102, 241, 0.3);
      border-radius: 10px;
      padding: 1.75rem;
      margin: 2rem 0;
    }

    .quiz-title {
      color: var(--accent-indigo);
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    .quiz-question {
      font-weight: 600;
      margin-bottom: 1rem;
      color: var(--text-bright);
    }

    .quiz-btn {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.65rem 1.25rem;
      border-radius: 6px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.88rem;
      transition: all 0.2s;
    }

    .quiz-btn:hover {
      background: var(--accent-indigo);
      color: #ffffff;
      border-color: var(--accent-indigo);
    }

    .quiz-reveal {
      display: none;
      margin-top: 1.25rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem;
    }

    /* Sliders, Inputs, & Buttons */
    .input-row {
      display: flex;
      gap: 1rem;
      margin-bottom: 1rem;
      flex-wrap: wrap;
    }

    .input-field {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      flex: 1;
      min-width: 140px;
    }

    .input-field label {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
    }

    .input-field input {
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text-bright);
      padding: 0.55rem 0.85rem;
      border-radius: 6px;
      font-family: monospace;
      font-size: 0.95rem;
    }

    .action-btn {
      background: var(--accent-blue);
      color: #0b0f19;
      font-weight: 700;
      border: none;
      padding: 0.65rem 1.5rem;
      border-radius: 6px;
      cursor: pointer;
      transition: opacity 0.2s;
    }

    .action-btn:hover { opacity: 0.85; }

    .result-badge {
      background: #090d13;
      border: 1px solid var(--accent-green);
      color: var(--accent-green);
      font-family: monospace;
      padding: 1rem;
      border-radius: 8px;
      margin-top: 1rem;
      font-size: 0.95rem;
    }

    /* Comparison Tables */
    table {
      width: 100%;
      border-collapse: collapse;
      margin: 1.5rem 0;
      font-size: 0.92rem;
    }

    th, td {
      padding: 0.85rem 1rem;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }

    th {
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-bright);
      font-weight: 700;
    }

    /* Code Blocks */
    pre code {
      border-radius: 8px;
      font-family: 'Fira Code', Consolas, Monaco, monospace;
      font-size: 0.88rem;
    }

    .code-pill {
      font-family: monospace;
      background: rgba(255,255,255,0.08);
      padding: 0.15rem 0.4rem;
      border-radius: 4px;
      font-size: 0.88em;
    }
  </style>
</head>
<body>

  <!-- SIDEBAR NAVIGATION -->
  <nav id="sidebar">
    <div>
      <div class="brand-title">🎓 RSI Master Tutor</div>
      <div class="brand-subtitle">Interactive Pretraining &amp; Self-Play Lab</div>
    </div>

    <div class="progress-box">
      <div style="display:flex; justify-content:space-between; font-size:0.8rem; font-weight:700;">
        <span>Curriculum Mastery</span>
        <span id="progress-pct">Module 1 / 6</span>
      </div>
      <div class="progress-bar-bg">
        <div class="progress-bar-fill" id="progress-fill"></div>
      </div>
    </div>

    <div>
      <div class="nav-group-title">Interactive Modules</div>
      <ul class="nav-links">
        <li><a href="#saturation" class="active"><span>0. Saturation &amp; Retrospective</span> <span>🎯</span></a></li>
        <li><a href="#module1"><span>1. Transformer Architecture</span> <span>🏛️</span></a></li>
        <li><a href="#module2"><span>2. High-Density Data &amp; BPE</span> <span>📦</span></a></li>
        <li><a href="#module3"><span>3. Optimization &amp; WSD</span> <span>⚡</span></a></li>
        <li><a href="#module4"><span>4. Self-Play Engine</span> <span>🤖</span></a></li>
        <li><a href="#module5"><span>5. Regression Guard &amp; Replay</span> <span>🛡️</span></a></li>
        <li><a href="#module6"><span>6. Production Observability</span> <span>📡</span></a></li>
      </ul>
    </div>

    <div>
      <div class="nav-group-title">Primary Artifacts</div>
      <ul class="nav-links">
        <li><a href="DASHBOARD.html" target="_blank"><span>Live Dashboard</span> <span>📊</span></a></li>
        <li><a href="RESEARCH_COMPENDIUM.html" target="_blank"><span>Research Compendium</span> <span>📚</span></a></li>
        <li><a href="notebooks/" target="_blank"><span>12 Jupyter Notebooks</span> <span>📁</span></a></li>
      </ul>
    </div>
  </nav>

  <!-- MAIN LEARNING PORTAL -->
  <main id="main-content">
    
    <!-- Top Action Bar -->
    <div class="top-action-bar">
      <div class="pill-links">
        <a href="DASHBOARD.html" target="_blank" class="pill-btn">📊 Live Executive Dashboard</a>
        <a href="RESEARCH_COMPENDIUM.html" target="_blank" class="pill-btn">📚 Complete Research Compendium</a>
        <a href="notebooks/" target="_blank" class="pill-btn">📁 12 Runnable Jupyter Notebooks</a>
      </div>
      <button class="theme-toggle-btn" onclick="toggleTheme()">🌓 Switch Theme</button>
    </div>

    <!-- SECTION 0: SATURATION VERDICT & RETROSPECTIVE OVERVIEW -->
    <section id="saturation" class="module-section">
      <div class="module-header">
        <span class="module-badge">Scientific Verdict &amp; GPU State</span>
        <h1>0. The Saturation Plateau: Why We Halt &amp; Reflect</h1>
      </div>

      <p>
        Across <strong>158 consecutive generations</strong> of autonomous self-play executed on the NVIDIA DGX Grace Blackwell GB10 GPU, the model traversed a classic three-phase evolutionary trajectory:
      </p>

      <div class="mermaid">
      graph LR
          A["Gen 0 -> 20<br><b>Exponential Takeoff</b><br>1.8% -> 41.8% (+40.0%)<br><i>+2.00% / generation</i>"] --> B["Gen 20 -> 80<br><b>Linear Consolidation</b><br>41.8% -> 50.9% (+9.1%)<br><i>+0.15% / generation</i>"]
          B --> C["Gen 80 -> 158<br><b>Logarithmic Plateau</b><br>50.9% -> 58.2% (+7.3%)<br><i>+0.09% / generation</i>"]
          C --> D["<b>Frontier Capacity Barrier</b><br>17 Unsolved Tasks<br><i>Requires MCTS search / scale</i>"]
          
          style A fill:#064e3b,stroke:#059669,stroke-width:2px;
          style B fill:#1e1b4b,stroke:#4f46e5,stroke-width:2px;
          style C fill:#451a03,stroke:#d97706,stroke-width:2px;
          style D fill:#4c0519,stroke:#e11d48,stroke-width:2px;
      </div>

      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">📈 Final Autonomous Run Metrics ($M_0 \to M_{158}$)</div>
        </div>
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Baseline ($M_0$)</th>
              <th>Midpoint ($M_{50}$)</th>
              <th>Final Peak ($M_{156} - M_{158}$)</th>
              <th>Total Gain</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Deterministic Pass@1</strong></td>
              <td>1.8% (1/55)</td>
              <td>47.3% (26/55)</td>
              <td><strong>58.2% (32/55)</strong></td>
              <td style="color:var(--accent-green);"><strong>+56.4% (+31 tasks)</strong></td>
            </tr>
            <tr>
              <td><strong>Any-Valid Pass@5</strong></td>
              <td>18.2% (10/55)</td>
              <td>50.9% (28/55)</td>
              <td><strong>61.8% (34/55)</strong></td>
              <td style="color:var(--accent-green);"><strong>+43.6% (+24 tasks)</strong></td>
            </tr>
            <tr>
              <td><strong>Replay Buffer Gold Solutions</strong></td>
              <td>0</td>
              <td>12,410</td>
              <td><strong>42,738 verified programs</strong></td>
              <td style="color:var(--accent-green);"><strong>+42,738 gold units</strong></td>
            </tr>
            <tr>
              <td><strong>Active GPU Power Draw</strong></td>
              <td>89.4 W (100% Load)</td>
              <td>91.2 W (100% Load)</td>
              <td><strong>16.36 W (0% Compute Halted)</strong></td>
              <td style="color:var(--accent-blue);"><strong>$0.00 / hour spend during retro</strong></td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        <strong>Why Halting Now is the Right Decision:</strong> In the final 78 cycles, the rate of capability discovery slowed from $+2.0\%$ per cycle down to $+0.09\%$ per cycle. The remaining 17 "Frontier" tasks (nested matrix transformations, 2D geometry, array intersections, recursive graph traversal) hit the <em>procedural capacity limit</em> of a 152M-parameter model without test-time search. To unlock them, our next phase will implement <strong>Test-Time Compute (MCTS &amp; reasoning scratchpads)</strong>.
      </p>
    </section>

    <!-- SECTION 1: MODERN TRANSFORMER ARCHITECTURE -->
    <section id="module1" class="module-section">
      <div class="module-header">
        <span class="module-badge">Module 1 &bull; Core Architecture</span>
        <h1>1. Modern Transformer Architecture Foundations</h1>
      </div>

      <p>
        Our architecture in <a href="file:///M:/Rama/Study/llm/LLM/RSI/src/model/transformer.py" target="_blank" class="code-pill">src/model/transformer.py</a> incorporates all five foundational advancements powering modern models like Llama 3 and DeepSeek:
      </p>

      <table>
        <thead>
          <tr>
            <th>Component</th>
            <th>Classic 2017 Transformer</th>
            <th>Modern 2026 LLM (Our Model)</th>
            <th>The Direct Advantage</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Normalization</strong></td>
            <td>Post-LN / Classic LayerNorm</td>
            <td><strong>Pre-RMSNorm</strong></td>
            <td>Saves 10-50% memory bandwidth; prevents gradient explosion.</td>
          </tr>
          <tr>
            <td><strong>Position Encoding</strong></td>
            <td>Absolute Additive Embeddings</td>
            <td><strong>Rotary Position Embeddings (RoPE)</strong></td>
            <td>Geometric relative position; seamless sequence length generalization.</td>
          </tr>
          <tr>
            <td><strong>Activation Function</strong></td>
            <td>Standard ReLU or GELU</td>
            <td><strong>SwiGLU Multiplicative Gate</strong></td>
            <td>Learned soft-filtering gate; superior downstream reasoning.</td>
          </tr>
          <tr>
            <td><strong>Attention Key/Value</strong></td>
            <td>Multi-Head Attention (12 Q : 12 KV)</td>
            <td><strong>Grouped-Query Attention (12 Q : 4 KV)</strong></td>
            <td>Cuts KV-cache RAM by 66.7% during auto-regressive generation.</td>
          </tr>
          <tr>
            <td><strong>Output Projection</strong></td>
            <td>Independent LM Head</td>
            <td><strong>Tied Embedding Weights</strong></td>
            <td>Reduces parameter footprint from 230M down to 152M.</td>
          </tr>
        </tbody>
      </table>

      <!-- 1.1 RMSNORM -->
      <h2>1.1 RMSNorm: Why Mean-Centering is Obsolete</h2>
      <p>
        Classic LayerNorm centers the input vector around zero ($\mu$) and divides by standard deviation ($\sigma$). Zhang &amp; Sennrich (2019) demonstrated that <em>centering around zero has zero stabilizing benefit</em>—what stabilizes deep networks is strictly <strong>scaling the vector magnitude</strong>.
      </p>

      <div class="pencil-paper-card">
        <div class="pencil-header">✏️ Hands-On Pencil &amp; Paper Exercise: Calculating RMSNorm by Hand</div>
        <p>Let's manually compute RMSNorm on paper for a small 3-dimensional vector without a computer:</p>
        <p style="font-family:monospace; font-size:1.1rem; color:var(--accent-amber);">
          x = [2.0, &nbsp; 4.0, &nbsp; -2.0] &nbsp; &nbsp; (Dimension d = 3, Gain &gamma; = 1.0, &epsilon; = 0)
        </p>

        <div class="step-box">
          <strong>Step 1: Square every number in the vector</strong><br>
          $2.0^2 = 4.0$, &nbsp; $4.0^2 = 16.0$, &nbsp; $(-2.0)^2 = 4.0$ &rarr; Squared vector: $[4.0, \; 16.0, \; 4.0]$
        </div>

        <div class="step-box">
          <strong>Step 2: Calculate the mean of the squares</strong><br>
          $\text{Mean} = \frac{4.0 + 16.0 + 4.0}{3} = \frac{24.0}{3} = \mathbf{8.0}$
        </div>

        <div class="step-box">
          <strong>Step 3: Take the square root to get Root Mean Square (RMS)</strong><br>
          $\text{RMS}(x) = \sqrt{8.0} \approx \mathbf{2.8284}$
        </div>

        <div class="step-box">
          <strong>Step 4: Divide each original number by that RMS value</strong><br>
          $y_1 = 2.0 / 2.8284 \approx \mathbf{0.7071}$<br>
          $y_2 = 4.0 / 2.8284 \approx \mathbf{1.4142}$<br>
          $y_3 = -2.0 / 2.8284 \approx \mathbf{-0.7071}$<br>
          &rarr; <strong>Final Normalized Output:</strong> $\mathbf{y} = [0.7071, \; 1.4142, \; -0.7071]$
        </div>
        <p style="font-size:0.85rem; color:var(--text-muted); margin-top:0.5rem;">
          <em>Notice:</em> If you calculate the Root Mean Square of this output vector $y$, it equals exactly $1.0000$!
        </p>
      </div>

      <!-- INTERACTIVE RMSNORM WIDGET -->
      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">🧪 Interactive RMSNorm Sandbox: Test Any Vector Live</div>
        </div>
        <p style="font-size:0.9rem;">Type comma-separated numbers below and click <strong>"Compute Step-by-Step"</strong> to watch the arithmetic happen live:</p>
        
        <div class="input-row" style="margin-top:1rem;">
          <div class="input-field" style="flex:3;">
            <label>Input Vector (x):</label>
            <input type="text" id="rmsnorm-input" value="2.0, 4.0, -2.0">
          </div>
          <button class="action-btn" onclick="runInteractiveRMSNorm()">Compute Step-by-Step</button>
        </div>

        <div id="rmsnorm-result" class="result-badge">
          Ready. Click 'Compute Step-by-Step' to calculate.
        </div>
      </div>

      <div class="gotcha-box">
        <div class="gotcha-title">⚠️ Production Gotcha: The Silent Float16 Overflow Trap</div>
        <p>
          In <a href="file:///M:/Rama/Study/llm/LLM/RSI/src/model/norm.py#L26" class="code-pill">src/model/norm.py (line 26)</a>, notice the call: <br>
          <code>return self._norm(x.float()).type_as(x)</code>.<br>
          <strong>Why is <code>.float()</code> strictly mandatory?</strong> In <code>bfloat16</code> or <code>fp16</code>, numbers greater than $256$ square to numbers $>65,536$, overflowing standard 16-bit half precision into infinity or NaN. RMSNorm must <em>always</em> upcast to 32-bit float for the sum of squares and square root, then downcast back to half-precision!
        </div>
      </div>

      <!-- 1.2 ROPE -->
      <h2>1.2 Rotary Position Embeddings (RoPE)</h2>
      <p>
        Instead of adding a static position vector, RoPE <strong>rotates</strong> Query and Key vectors in 2D geometric planes. The attention score between token $m$ and token $n$ depends purely on the <strong>relative angle $(\theta \cdot (m - n))$</strong>.
      </p>

      <!-- ROPE CLOCK WIDGET -->
      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">⏰ Interactive RoPE Rotating Clock Visualizer</div>
        </div>
        <p style="font-size:0.9rem;">
          Drag the sliders to change token positions $m$ and $n$. Notice how shifting both tokens together spins the hands, but leaves the <strong>angular difference and attention score 100% identical</strong>!
        </p>

        <div style="display:grid; grid-template-columns: 280px 1fr; gap:2rem; align-items:center; margin-top:1.5rem;">
          <div style="display:flex; flex-direction:column; align-items:center;">
            <canvas id="ropeClock" width="240" height="240" style="background:#090d13; border-radius:50%; border:2px solid var(--border);"></canvas>
            <div style="display:flex; gap:1rem; margin-top:0.75rem; font-size:0.8rem; font-weight:700;">
              <span style="color:var(--accent-blue);">● Query Vector ($m$)</span>
              <span style="color:var(--accent-purple);">● Key Vector ($n$)</span>
            </div>
          </div>

          <div>
            <div class="input-field" style="margin-bottom:1rem;">
              <label>Query Token Position ($m$): <span id="posMVal" style="color:var(--accent-blue); font-weight:700;">10</span></label>
              <input type="range" id="sliderM" min="0" max="60" value="10" oninput="updateRoPEClock()">
            </div>

            <div class="input-field" style="margin-bottom:1rem;">
              <label>Key Token Position ($n$): <span id="posNVal" style="color:var(--accent-purple); font-weight:700;">12</span></label>
              <input type="range" id="sliderN" min="0" max="60" value="12" oninput="updateRoPEClock()">
            </div>

            <div class="result-badge" id="ropeBadge">
              Relative Distance: 2 tokens | Angular Offset: 60.0° | Attention Score: 0.500
            </div>

            <div style="display:flex; gap:0.5rem; margin-top:1rem;">
              <button class="pill-btn" onclick="setRoPEPositions(10, 12)">Pos (10, 12)</button>
              <button class="pill-btn" onclick="setRoPEPositions(40, 42)">Shift to (40, 42) [Invariance Demo]</button>
              <button class="pill-btn" onclick="setRoPEPositions(10, 25)">Distance = 15 Tokens</button>
            </div>
          </div>
        </div>
      </div>

      <!-- 1.3 SWIGLU -->
      <h2>1.3 SwiGLU: Multiplicative Gated Networks</h2>
      <p>
        In standard MLPs, activations simply pass through ReLU/GELU. In <strong>SwiGLU</strong> (<a href="file:///M:/Rama/Study/llm/LLM/RSI/src/model/mlp.py" class="code-pill">src/model/mlp.py</a>), the input branches into two parallel paths:
      </p>
      $$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$

      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">🚪 Interactive SwiGLU Gating Demonstration</div>
        </div>
        <p style="font-size:0.9rem;">
          Drag the input value $x$ to see how the $\text{SiLU}$ gate smoothly closes for negative inputs and opens for positive inputs:
        </p>

        <div class="input-row" style="margin-top:1rem;">
          <div class="input-field">
            <label>Input Value $x$: <span id="swiglu-x-val" style="color:var(--accent-blue); font-weight:700;">1.5</span></label>
            <input type="range" id="swiglu-slider" min="-40" max="40" value="15" oninput="updateSwiGLU()">
          </div>
        </div>

        <div id="swiglu-result" class="result-badge">
          Computing...
        </div>
      </div>

      <!-- 1.4 GQA -->
      <h2>1.4 Grouped-Query Attention (GQA) &amp; Memory Savings</h2>
      <p>
        Multi-Head Attention (MHA) allocates 12 Key and 12 Value heads for 12 Query heads. During generation, storing all KV heads in GPU RAM (the KV-cache) quickly exhausts memory. <strong>Grouped-Query Attention (GQA)</strong> shares 1 KV head across 3 Query heads (12 Q : 4 KV).
      </p>

      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">💾 Interactive KV-Cache Memory Savings Calculator</div>
        </div>
        <div class="input-row">
          <div class="input-field">
            <label>Batch Size:</label>
            <input type="number" id="gqa-batch" value="16" min="1" max="128" oninput="calculateGQAMemory()">
          </div>
          <div class="input-field">
            <label>Context Length (Tokens):</label>
            <input type="number" id="gqa-ctx" value="2048" min="512" max="32768" step="512" oninput="calculateGQAMemory()">
          </div>
          <div class="input-field">
            <label>Model Layers:</label>
            <input type="number" id="gqa-layers" value="12" readonly>
          </div>
        </div>

        <div id="gqa-memory-result" class="result-badge">
          Calculating KV-Cache footprint...
        </div>
      </div>

      <!-- SOCRATIC QUIZ MODULE 1 -->
      <div class="quiz-box">
        <div class="quiz-title">🎯 Module 1 Self-Check Socratic Quiz</div>
        
        <div class="quiz-question">
          1. Why does RMSNorm eliminate the mean-subtraction step $(\mu = \frac{1}{d}\sum x_i)$, and what hardware bottleneck does this relieve on our DGX Blackwell GPU?
        </div>
        <button class="quiz-btn" onclick="toggleQuizAnswer('q1-ans')">Reveal Tutor Answer &amp; Deep Dive</button>
        <div id="q1-ans" class="quiz-reveal">
          <p><strong>Answer:</strong> Zhang &amp; Sennrich (2019) demonstrated empirically and theoretically that shifting activations by subtracting their mean contributes essentially nothing to training stability. Stability is achieved strictly by scaling the vector's length (norm) so gradients don't vanish or explode.</p>
          <p>By eliminating mean subtraction, RMSNorm saves an entire memory pass over the hidden dimension tensor, reducing GPU High-Bandwidth Memory (HBM) traffic by <strong>10% to 50%</strong> during backpropagation!</p>
        </div>

        <div class="quiz-question" style="margin-top:1.5rem;">
          2. If an input vector has dimension $d=768$ and vocabulary size $V=100,262$, why does weight tying (`lm_head.weight = tok_embeddings.weight`) save nearly 77 million parameters?
        </div>
        <button class="quiz-btn" onclick="toggleQuizAnswer('q2-ans')">Reveal Tutor Answer &amp; Deep Dive</button>
        <div id="q2-ans" class="quiz-reveal">
          <p><strong>Answer:</strong> An independent output projection matrix (`lm_head`) requires $V \times d$ parameters: $100,262 \times 768 = \mathbf{77,001,216}$ parameters. In 16-bit precision, that single matrix alone consumes over 154 MB of GPU memory!</p>
          <p>By tying the output matrix to the input embedding matrix, we eliminate that matrix entirely, dropping our model size from 230M down to 152M while keeping input token representations and output target predictions in the exact same mathematical geometry.</p>
        </div>
      </div>
    </section>

    <!-- MODULE 2: DATA PIPELINE -->
    <section id="module2" class="module-section">
      <div class="module-header">
        <span class="module-badge">Module 2 &bull; Data Engine</span>
        <h1>2. High-Density Data Pipeline &amp; BPE Tokenization</h1>
      </div>
      <p>
        Foundation coding models live or die by their data density. Feeding raw, uncurated code or padding batches with <code>&lt;pad&gt;</code> tokens wastes up to 75% of your GPU FLOPS.
      </p>

      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">📉 The Data Density Breakthrough (Validation Perplexity Drop)</div>
        </div>
        <table>
          <thead>
            <tr>
              <th>Dataset Configuration</th>
              <th>Validation Loss</th>
              <th>Validation Perplexity (PPL)</th>
              <th>Deterministic Pass@1</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Baseline StarCoder Synthetic Shards</td>
              <td>2.695</td>
              <td>14.81</td>
              <td>0.0%</td>
            </tr>
            <tr>
              <td><strong>High-Density Tri-Mix (flytech + TokenBender + codeparrot)</strong></td>
              <td><strong>2.326</strong></td>
              <td><strong>10.24</strong></td>
              <td><strong>1.8% (Initial seed)</strong></td>
            </tr>
          </tbody>
        </table>
        <p style="margin-top:1rem; font-size:0.9rem; color:var(--accent-green);">
          <strong>Result:</strong> Simply switching to high-density, memory-mapped binary sharding (<a href="file:///M:/Rama/Study/llm/LLM/RSI/src/data/dataset.py" class="code-pill">src/data/dataset.py</a>) dropped validation perplexity by <strong>30.9%</strong> before self-play even started!
        </p>
      </div>

      <div class="gotcha-box">
        <div class="gotcha-title">⚠️ The Tokenizer Trap: Unmatched Vocabularies</div>
        <p>
          Our vocabulary size is <strong>100,262 tokens</strong> (derived from StarCoder2 BPE). If your model configuration sets <code>vocab_size = 50,257</code> (standard GPT-2) but your tokenizer emits IDs up to 100,261, PyTorch will throw a silent CUDA device assert error during embedding lookup that crashes your GPU cluster.
        </p>
      </div>
    </section>

    <!-- MODULE 3: OPTIMIZATION & WSD -->
    <section id="module3" class="module-section">
      <div class="module-header">
        <span class="module-badge">Module 3 &bull; Optimization</span>
        <h1>3. Optimization Dynamics: The Warmup-Stable-Decay Advantage</h1>
      </div>

      <p>
        In traditional pretraining, practitioners use <strong>Cosine Annealing</strong>. While popular, Cosine is fatal for lifelong autonomous self-play!
      </p>

      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">⚡ WSD (Warmup-Stable-Decay) vs. Cosine Annealing</div>
        </div>
        <div class="mermaid">
        graph LR
            subgraph Cosine ["Cosine Schedule (The Trap)"]
                C1["Warmup"] --> C2["Continuous Decay to 0"] --> C3["Frozen Model (LR ~ 0)<br>Cannot continue training!"]
            end
            subgraph WSD ["WSD Schedule (Our Strategy)"]
                W1["Warmup (10%)"] --> W2["Infinite Stable Plateau (LR = 1.2e-3)<br>Continuous Self-Play Lifelong Learning"] --> W3["Decay (10% at release)"]
            end
            style Cosine fill:#4c0519,stroke:#e11d48,stroke-width:1px;
            style WSD fill:#064e3b,stroke:#059669,stroke-width:1px;
        </div>
      </div>

      <div class="gotcha-box">
        <div class="gotcha-title">⚠️ The Weight Decay Gotcha in AdamW</div>
        <p>
          In <a href="file:///M:/Rama/Study/llm/LLM/RSI/src/engine/optim.py" class="code-pill">src/engine/optim.py</a>, parameters are split into two groups:
          <br>1. <strong>2D Weights (Linear layers, attention projections):</strong> Weight decay = 0.1
          <br>2. <strong>1D Parameters (RMSNorm gamma gains, biases):</strong> Weight decay = <strong>0.0</strong>
          <br><em>Why?</em> Applying weight decay to normalizer gains $\gamma$ artificially shrinks activation scales over time, destabilizing the entire network!
        </p>
      </div>
    </section>

    <!-- MODULE 4: SELF-PLAY ENGINE -->
    <section id="module4" class="module-section">
      <div class="module-header">
        <span class="module-badge">Module 4 &bull; Autonomous Self-Play</span>
        <h1>4. The Autonomous Self-Play Engine &amp; Sandboxed Verification</h1>
      </div>

      <p>
        How did our model climb from 1.8% to 58.2% without human labeling? Through an autonomous loop that generates candidate solutions, tests them inside a secure Python AST sandbox, and filters gold programs:
      </p>

      <div class="mermaid">
      graph TD
          Task["Problem Prompt &amp; Signature"] --> Gen["Generator (src/rsi/generator.py)<br>Adaptive Sampling: K=18 Frontier, K=3 Mastered<br>Temp=0.6, Top-P=0.95"]
          Gen --> Candidates["K Candidate Programs"]
          Candidates --> Verifier["AST Sandboxed Verifier (src/rsi/verifier.py)<br>1. AST Node Inspection (Block OS, Sys, Sockets)<br>2. Subprocess Isolation with 2.0s Timeout<br>3. Unit Test Assertions"]
          Verifier --> Filter{"Unit Tests Passed?"}
          Filter -- YES --> Gold["Harvested Gold Solution<br>Pushed to Replay Buffer (42,738 Total)"]
          Filter -- NO --> Discard["Discarded Candidate"]
          Gold --> Distill["Distillation Trainer<br>90% Pretrain + 10% Gold Replay"]
          
          style Gold fill:#064e3b,stroke:#059669,stroke-width:2px;
          style Discard fill:#4c0519,stroke:#e11d48,stroke-width:1px;
      </div>
    </section>

    <!-- MODULE 5: REGRESSION GUARD -->
    <section id="module5" class="module-section">
      <div class="module-header">
        <span class="module-badge">Module 5 &bull; Safety &amp; Guardrails</span>
        <h1>5. The Regression Guard &amp; Catastrophic Forgetting Defense</h1>
      </div>

      <p>
        In RL and self-play, models frequently suffer from <em>catastrophic forgetting</em>: learning one new algorithm causes them to forget 5 existing ones. How do we statistically detect real regressions versus random sampling noise?
      </p>

      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">📐 Derivation of Binomial Variance for $N=55$ Benchmarks</div>
        </div>
        <p>
          For a benchmark of $N=55$ problems, accuracy $\hat{p}$ is a binomial random variable with variance:
        </p>
        $$\sigma_{\hat{p}} = \sqrt{\frac{p(1 - p)}{N}}$$
        <p>
          At $p \approx 0.50$ (50% accuracy):
          $$\sigma_{\hat{p}} = \sqrt{\frac{0.50 \times 0.50}{55}} = \sqrt{0.004545} \approx \mathbf{6.74\%}$$
        </p>
        <p>
          A single task flipping from pass to fail shifts accuracy by $\frac{1}{55} \approx \mathbf{1.82\%}$. Therefore, setting our regression guard tolerance to **$\Delta \ge -2.0\%$** allows single-problem binomial noise while strictly rejecting true multi-problem capability regressions!
        </p>
      </div>

      <!-- INTERACTIVE REGRESSION GUARD CALCULATOR -->
      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">🛡️ Interactive Regression Guard Sandbox</div>
        </div>
        <div class="input-row">
          <div class="input-field">
            <label>Parent Checkpoint Accuracy (%):</label>
            <input type="number" id="guard-parent" value="50.9" step="0.1" oninput="testRegressionGuard()">
          </div>
          <div class="input-field">
            <label>Candidate Checkpoint Accuracy (%):</label>
            <input type="number" id="guard-candidate" value="47.3" step="0.1" oninput="testRegressionGuard()">
          </div>
        </div>
        <div id="guard-result" class="result-badge">
          Ready to evaluate checkpoint promotion.
        </div>
      </div>
    </section>

    <!-- MODULE 6: SYSTEMS ENGINEERING -->
    <section id="module6" class="module-section">
      <div class="module-header">
        <span class="module-badge">Module 6 &bull; Systems &amp; Observability</span>
        <h1>6. Production Systems: 24/7 Daemons &amp; Observability</h1>
      </div>

      <p>
        How did our experiment run for 24+ consecutive hours across 158 generations without crashing when the laptop slept?
      </p>

      <div class="widget-container">
        <div class="widget-header">
          <div class="widget-title">📡 The Perpetual Supervisor Architecture (<a href="file:///M:/Rama/Study/llm/LLM/RSI/src/rsi/perpetual_supervisor.py" class="code-pill">src/rsi/perpetual_supervisor.py</a>)</div>
        </div>
        <ul style="padding-left:1.5rem; display:flex; flex-direction:column; gap:0.75rem;">
          <li><strong>Detached Process Tree:</strong> Spawned under <code>nohup ... &amp;</code> with redirected stdout/stderr to disk. When SSH connections drop (SIGHUP), the Linux kernel preserves the background process group.</li>
          <li><strong>State-File Polling Loop:</strong> The supervisor polls <code>data/rsi/problem_bank.json</code> and generation milestones every 30 seconds.</li>
          <li><strong>Zero-Idle Auto-Chaining:</strong> As soon as Generation $K$ finishes evaluating, the supervisor automatically launches Generation $K+1$ through $K+15$, ensuring the DGX Blackwell GB10 GPU never sits idle at 0% power.</li>
        </ul>
      </div>

      <!-- CONCLUSION & NEXT STEPS -->
      <h2>Next Horizon: Phase 5 (Test-Time Compute &amp; MCTS)</h2>
      <p>
        Now that you have mastered all 6 modules from Transformer mathematics to autonomous self-play loops, our next scientific milestone is unlocking the final 17 frontier tasks using <strong>Monte Carlo Tree Search (MCTS)</strong> and test-time reasoning compute!
      </p>
    </section>

  </main>

  <script>
    // --- SYNTAX HIGHLIGHTING ---
    hljs.highlightAll();

    // --- THEME TOGGLE ---
    function toggleTheme() {
      const html = document.documentElement;
      const current = html.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
    }

    // --- QUIZ ACCORDION REVEAL ---
    function toggleQuizAnswer(id) {
      const el = document.getElementById(id);
      if (el.style.display === 'block') {
        el.style.display = 'none';
      } else {
        el.style.display = 'block';
      }
    }

    // --- INTERACTIVE RMSNORM LIVE CALCULATOR ---
    function runInteractiveRMSNorm() {
      const inputStr = document.getElementById('rmsnorm-input').value;
      const resultBox = document.getElementById('rmsnorm-result');
      try {
        const nums = inputStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
        if (nums.length === 0) {
          resultBox.innerHTML = '<span style="color:var(--accent-rose)">Please enter at least one valid number!</span>';
          return;
        }
        const d = nums.length;
        const squares = nums.map(x => x * x);
        const meanSq = squares.reduce((a, b) => a + b, 0) / d;
        const rms = Math.sqrt(meanSq);
        const normalized = nums.map(x => x / rms);
        const normRms = Math.sqrt(normalized.map(y => y * y).reduce((a, b) => a + b, 0) / d);

        resultBox.innerHTML = `
          <strong>Input Vector (x):</strong> [${nums.map(n => n.toFixed(2)).join(', ')}] (Dimension d = ${d})<br>
          <strong>Squared Elements (x²):</strong> [${squares.map(s => s.toFixed(2)).join(', ')}]<br>
          <strong>Mean of Squares:</strong> ${meanSq.toFixed(4)}<br>
          <strong>Root Mean Square (RMS):</strong> √${meanSq.toFixed(4)} = <strong>${rms.toFixed(4)}</strong><br>
          <hr style="border:0; border-top:1px solid var(--border); margin:0.75rem 0;">
          <strong>Normalized Vector y = x / RMS:</strong> [${normalized.map(y => y.toFixed(4)).join(', ')}]<br>
          <span style="color:var(--accent-green);"><strong>Output Verification:</strong> RMS of y = ${normRms.toFixed(4)} (Exactly 1.0000!)</span>
        `;
      } catch (err) {
        resultBox.innerHTML = '<span style="color:var(--accent-rose)">Error parsing input vector.</span>';
      }
    }

    // --- ROPE CLOCK VISUALIZER ---
    const canvas = document.getElementById("ropeClock");
    const ctx = canvas.getContext("2d");
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 95;
    const stepDeg = 30; // 30 degrees per token

    function updateRoPEClock() {
      const m = parseInt(document.getElementById("sliderM").value);
      const n = parseInt(document.getElementById("sliderN").value);
      document.getElementById("posMVal").textContent = m;
      document.getElementById("posNVal").textContent = n;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Outer dial
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.strokeStyle = "#334155";
      ctx.lineWidth = 3;
      ctx.stroke();

      // Hour ticks
      for (let i = 0; i < 12; i++) {
        const ang = (i * 30 - 90) * (Math.PI / 180);
        const x1 = centerX + Math.cos(ang) * (radius - 8);
        const y1 = centerY + Math.sin(ang) * (radius - 8);
        const x2 = centerX + Math.cos(ang) * radius;
        const y2 = centerY + Math.sin(ang) * radius;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = "#64748b";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      const angleM = (m * stepDeg - 90) * (Math.PI / 180);
      const angleN = (n * stepDeg - 90) * (Math.PI / 180);

      // Query Hand (Blue)
      drawClockHand(angleM, radius * 0.82, "#38bdf8", 4);
      // Key Hand (Purple)
      drawClockHand(angleN, radius * 0.68, "#c084fc", 4);

      // Center pivot
      ctx.beginPath();
      ctx.arc(centerX, centerY, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "#ffffff";
      ctx.fill();

      // Calculations
      const diffTokens = Math.abs(m - n);
      const angDiffDeg = (diffTokens * stepDeg) % 360;
      const score = Math.cos(angDiffDeg * Math.PI / 180).toFixed(3);

      document.getElementById("ropeBadge").innerHTML = `
        Relative Distance: <strong>${diffTokens} tokens</strong> | Angular Difference: <strong>${angDiffDeg.toFixed(1)}°</strong> | Attention Score: <strong>${score}</strong>
      `;
    }

    function drawClockHand(angle, length, color, width) {
      const x = centerX + Math.cos(angle) * length;
      const y = centerY + Math.sin(angle) * length;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = "round";
      ctx.stroke();
    }

    function setRoPEPositions(m, n) {
      document.getElementById("sliderM").value = m;
      document.getElementById("sliderN").value = n;
      updateRoPEClock();
    }

    // --- SWIGLU LIVE DEMO ---
    function updateSwiGLU() {
      const rawVal = parseInt(document.getElementById('swiglu-slider').value);
      const x = rawVal / 10;
      document.getElementById('swiglu-x-val').textContent = x.toFixed(1);
      
      const silu = x / (1 + Math.exp(-x));
      const up = x * 1.5;
      const gatedProduct = silu * up;

      document.getElementById('swiglu-result').innerHTML = `
        <strong>Input (x):</strong> ${x.toFixed(2)}<br>
        <strong>SiLU Gate Projection (σ(x) · x):</strong> ${silu.toFixed(3)} ${x < 0 ? '(Valve Closed: Damped)' : '(Valve Open: Passing)'}<br>
        <strong>Up Projection Candidate Features:</strong> ${up.toFixed(3)}<br>
        <strong>Gated Element-Wise Product:</strong> ${silu.toFixed(3)} × ${up.toFixed(3)} = <strong>${gatedProduct.toFixed(3)}</strong>
      `;
    }

    // --- GQA MEMORY CALCULATOR ---
    function calculateGQAMemory() {
      const batch = parseInt(document.getElementById('gqa-batch').value) || 16;
      const ctxLen = parseInt(document.getElementById('gqa-ctx').value) || 2048;
      const layers = 12;
      const headDim = 64; // 768 / 12
      const bytesPerElem = 2; // FP16/BF16

      // Formula: 2 (K and V) * batch * layers * kv_heads * seq_len * head_dim * bytes
      const mhaBytes = 2 * batch * layers * 12 * ctxLen * headDim * bytesPerElem;
      const gqaBytes = 2 * batch * layers * 4 * ctxLen * headDim * bytesPerElem;
      const mqaBytes = 2 * batch * layers * 1 * ctxLen * headDim * bytesPerElem;

      const mhaMB = (mhaBytes / (1024 * 1024)).toFixed(1);
      const gqaMB = (gqaBytes / (1024 * 1024)).toFixed(1);
      const mqaMB = (mqaBytes / (1024 * 1024)).toFixed(1);

      document.getElementById('gqa-memory-result').innerHTML = `
        <strong>Multi-Head Attention (12 KV Heads):</strong> ${mhaMB} MB RAM<br>
        <strong>Grouped-Query Attention (4 KV Heads - Our Model):</strong> <span style="color:var(--accent-green); font-weight:700;">${gqaMB} MB RAM (66.7% Memory Saved!)</span><br>
        <strong>Multi-Query Attention (1 KV Head):</strong> ${mqaMB} MB RAM (91.7% Memory Saved)
      `;
    }

    // --- REGRESSION GUARD SANDBOX ---
    function testRegressionGuard() {
      const parent = parseFloat(document.getElementById('guard-parent').value) || 50.9;
      const cand = parseFloat(document.getElementById('guard-candidate').value) || 47.3;
      const delta = (cand - parent).toFixed(1);
      const resultBox = document.getElementById('guard-result');

      const isPass = parseFloat(delta) >= -2.0;

      if (isPass) {
        resultBox.innerHTML = `
          Delta = <strong>${delta > 0 ? '+' + delta : delta}%</strong> &ge; -2.0% threshold.<br>
          <span style="color:var(--accent-green); font-weight:700;">VERDICT: PROMOTED &check; (Checkpoint Accepted &amp; Replay Buffer Distilled)</span>
        `;
        resultBox.style.borderColor = "var(--accent-green)";
      } else {
        resultBox.innerHTML = `
          Delta = <strong>${delta}%</strong> &lt; -2.0% threshold.<br>
          <span style="color:var(--accent-rose); font-weight:700;">VERDICT: REJECTED (GUARDED) &cross; (Automatic Rollback to Parent Checkpoint to Protect Buffer)</span>
        `;
        resultBox.style.borderColor = "var(--accent-rose)";
      }
    }

    // Initialize all widgets on load
    window.onload = function() {
      runInteractiveRMSNorm();
      updateRoPEClock();
      updateSwiGLU();
      calculateGQAMemory();
      testRegressionGuard();
    };
  </script>
</body>
</html>
"""
    return html_content

def main():
    print("Building Interactive Master Tutor Notebook...")
    html_code = generate_tutor_html()
    
    local_path = Path("M:/Rama/Study/llm/LLM/RSI/RSI_MASTER_TUTOR_NOTEBOOK.html")
    drive_path = Path("D:/D/Google Drive/RSI_Pretraining_Lab/RSI_MASTER_TUTOR_NOTEBOOK.html")
    
    # Save local
    local_path.write_text(html_code, encoding="utf-8")
    print(f"Saved locally: {local_path} ({len(html_code):,} bytes)")
    
    # Mirror to Google Drive
    drive_dir = drive_path.parent
    if drive_dir.exists():
        drive_path.write_text(html_code, encoding="utf-8")
        print(f"Mirrored to Google Drive: {drive_path}")
    else:
        print(f"Google Drive folder not found at {drive_dir}")

if __name__ == "__main__":
    main()
