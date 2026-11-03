# Tutor Pedagogy & Teaching Guidelines (RSI Pretraining)

This repository follows a strict, student-first teaching methodology. All lessons, explanations, and notebooks must adhere to this exact 5-step pedagogical framework:

---

## The 5-Step Teaching Framework

### 1. Intuition & Real-World Metaphor First
* **Never start with abstract math, raw equations, or tensor dimensions.**
* Start with an intuitive analogy from everyday life or real software engineering (e.g., *the spacecraft landing for LR schedules*, *the clock face for RoPE*).
* Clearly state **why** older architectures failed and **what problem** this modern component solves.

### 2. Publication-Quality Vector Diagrams (NO ASCII)
* **Never use text/ASCII art diagrams.**
* Use **presentation-grade SVG vector diagrams** with:
  - Clean rounded cards, drop shadows, and modern color themes (slate blue, emerald green, purple).
  - Clear directional arrows showing data flow and phase transitions.
  - Crisp vector rendering that looks professional and publication-ready.

### 3. Hands-On Pencil & Paper Exercise (Tiny Numbers)
* Break the concept down into small, friendly numbers (dimension 2, 3, or 4).
* Walk the student through manual arithmetic step-by-step:
  - No matrices larger than size 4.
  - Show each intermediate calculation explicitly (squares, averages, dot products).
* Allow the student to calculate and verify the results on paper to demystify the "black box".

### 4. Zero-KaTeX Notebook Standard
* **Do not use fragile LaTeX (`$...$`, `\text{}`, `\sqrt{}`) in Jupyter markdown cells.**
* Use **100% clean, bulletproof HTML styling and Unicode characters** (`x'₁`, `θ`, `√`, `cos`, `sin`, `⊙`) to ensure 100% error-free rendering across VS Code, JupyterLab, and web viewers.
* **Keep all exercises for a given topic inside a single, unified notebook**.

### 5. Runnable PyTorch Code Verification
* Immediately follow the paper calculation with real, modular PyTorch code from our actual `src/` codebase.
* Show the model executing on the exact same numbers, proving that the Python implementation matches the student's handwritten math to the exact decimal point.

### 6. Proactive Blindspots & Traps ("What You Haven't Asked")
* **Never wait for the student to encounter a disaster.**
* For every topic, architecture choice, and training phase, **proactively surface the unasked questions, hidden traps, silent failure modes, and production gotchas** that practitioners usually only discover the hard way (e.g. SIGHUP terminal disconnections, Step-0 $\ln(V)$ checks, AdamW momentum loss on checkpoint restarts, softmax FP32 underflows).
* Break down the exact mathematical and physical reasons behind these gotchas.

---
*Created and adhered to for the RSI Pretraining Tutorial.*
