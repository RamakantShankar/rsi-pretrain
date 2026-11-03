# Tutor Pedagogy & Teaching Guidelines (RSI Pretraining)

This repository follows a strict, student-first teaching methodology. All lessons, explanations, and notebooks must adhere to this exact 5-step pedagogical framework:

---

## The 5-Step Teaching Framework

### 1. Intuition & Real-World Metaphor First
* **Never start with abstract math, raw equations, or tensor dimensions.**
* Start with an intuitive analogy from everyday life or real software engineering (e.g., *the clock face for RoPE*, *the consultant & fact-checker for SwiGLU*).
* Clearly state **why** older architectures failed and **what problem** this modern component solves.

### 2. Clear Visual ASCII / Flow Diagram
* Provide an unambiguous visual flow diagram showing the data lifecycle:
  - What inputs enter the block?
  - Where does context come from?
  - How do the signals split, transform, and recombine?
* Use clean ASCII boxes with clear directional arrows (`↓`, `→`).

### 3. Hands-On Pencil & Paper Exercise (Tiny Numbers)
* Break the concept down into small, friendly numbers (dimension 2, 3, or 4).
* Walk the student through manual arithmetic step-by-step:
  - No matrices larger than size 4.
  - Show each intermediate calculation explicitly (squares, averages, dot products).
* Allow the student to calculate and verify the results on paper to demystify the "black box".

### 4. Zero-KaTeX Notebook Standard
* **Do not use fragile LaTeX (`$...$`, `\text{}`, `\sqrt{}`) in Jupyter markdown cells.**
* Use **100% clean, bulletproof HTML styling and Unicode characters** (`x'₁`, `θ`, `√`, `cos`, `sin`, `⊙`, `[16px pre blocks]`) to ensure 100% error-free rendering across VS Code, JupyterLab, and web viewers.
* **Keep all exercises for a given topic inside a single, unified notebook** (e.g., `01_rope_intuition.ipynb`, `02_swiglu_explained.ipynb`, `03_gqa_explained.ipynb`).

### 5. Runnable PyTorch Code Verification
* Immediately follow the paper calculation with real, modular PyTorch code from our actual `src/` codebase.
* Show the model executing on the exact same numbers, proving that the Python implementation matches the student's handwritten math to the exact decimal point.

---
*Created and adhered to for the RSI Pretraining Tutorial.*
