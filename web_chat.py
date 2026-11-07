#!/usr/bin/env python3
"""
Mirendil-152M RSI Champion (M_156) Web Chat Server.
Runs on NVIDIA DGX Blackwell GB10 GPU (Port 7860).
Provides an interactive web chat portal accessible across the local network.
"""

import os
import sys
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.model.transformer import TransformerDecoder, ModelConfig
from src.tokenizer.bpe import CodeTokenizer
from chat import load_champion_model, generate_response, CHAT_PROMPT_HEADER

app = FastAPI(title="Mirendil-152M RSI Chat Portal")

# Global model and tokenizer
model = None
tokenizer = None
device = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🤖 Mirendil-152M RSI Champion Live Chat (DGX Blackwell)</title>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <style>
    :root {
      --bg: #090d16;
      --surface: #111827;
      --card: #1f2937;
      --border: #374151;
      --accent-blue: #38bdf8;
      --accent-green: #34d399;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 1rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand h1 { font-size: 1.25rem; font-weight: 700; color: #fff; }
    .badge {
      font-size: 0.75rem;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent-blue);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 0.2rem 0.6rem;
      border-radius: 20px;
      font-weight: 600;
    }
    .device-badge {
      font-size: 0.8rem;
      color: var(--accent-green);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    #chat-container {
      flex: 1;
      overflow-y: auto;
      padding: 2rem;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
      max-width: 950px;
      margin: 0 auto;
      width: 100%;
    }
    .message {
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      padding: 1.25rem;
      border-radius: 10px;
      max-width: 85%;
      line-height: 1.6;
    }
    .message.user {
      align-self: flex-end;
      background: #1e3a8a;
      border: 1px solid #2563eb;
      color: #fff;
    }
    .message.assistant {
      align-self: flex-start;
      background: var(--surface);
      border: 1px solid var(--border);
    }
    .message-role {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }
    pre code {
      border-radius: 8px;
      font-family: 'Fira Code', Consolas, Monaco, monospace;
      font-size: 0.9rem;
    }
    #controls {
      background: var(--surface);
      border-top: 1px solid var(--border);
      padding: 1rem 2rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .presets-row {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      max-width: 950px;
      margin: 0 auto;
      width: 100%;
    }
    .preset-btn {
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 0.3rem 0.7rem;
      border-radius: 15px;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .preset-btn:hover {
      color: var(--accent-blue);
      border-color: var(--accent-blue);
    }
    .input-row {
      display: flex;
      gap: 1rem;
      max-width: 950px;
      margin: 0 auto;
      width: 100%;
    }
    #prompt-input {
      flex: 1;
      background: var(--card);
      border: 1px solid var(--border);
      color: #fff;
      padding: 0.85rem 1.25rem;
      border-radius: 8px;
      font-size: 1rem;
      outline: none;
    }
    #prompt-input:focus { border-color: var(--accent-blue); }
    .send-btn {
      background: var(--accent-blue);
      color: #000;
      border: none;
      padding: 0.85rem 1.75rem;
      border-radius: 8px;
      font-weight: 700;
      font-size: 0.95rem;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    .send-btn:hover { opacity: 0.85; }
    .settings-bar {
      display: flex;
      gap: 1.5rem;
      font-size: 0.8rem;
      color: var(--text-muted);
      max-width: 950px;
      margin: 0 auto;
      width: 100%;
      align-items: center;
    }
    .setting-item { display: flex; align-items: center; gap: 0.5rem; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>🤖 Mirendil-152M RSI Champion (M_156)</h1>
      <span class="badge">58.2% Pass@1 &bull; 42,738 Gold Programs</span>
    </div>
    <div class="device-badge">
      <span>●</span> Active on NVIDIA Grace Blackwell GB10 GPU
    </div>
  </header>

  <div id="chat-container">
    <div class="message assistant">
      <div class="message-role">Champion Assistant (M_156)</div>
      <div>
        Hello! I am your 152M-parameter autonomous self-play model running on the DGX Grace Blackwell GB10 GPU.<br>
        Ask me to write any Python function, or test my algorithmic capabilities!
      </div>
    </div>
  </div>

  <div id="controls">
    <div class="presets-row">
      <span style="font-size:0.75rem; color:var(--text-muted); padding:0.3rem 0;">Quick Presets:</span>
      <button class="preset-btn" onclick="usePreset('Write a function to check if a string is a palindrome.')">Palindrome Check</button>
      <button class="preset-btn" onclick="usePreset('Write a function to reverse a string.')">Reverse String</button>
      <button class="preset-btn" onclick="usePreset('Write a function to calculate the factorial of n.')">Factorial</button>
      <button class="preset-btn" onclick="usePreset('def is_even(n):')">Raw: is_even()</button>
      <button class="preset-btn" onclick="usePreset('def count_vowels(s):')">Raw: count_vowels()</button>
      <button class="preset-btn" onclick="usePreset('def binary_search(arr, target):')">Raw: binary_search()</button>
    </div>

    <div class="input-row">
      <input type="text" id="prompt-input" placeholder="Type a Python instruction or function signature (e.g. 'def is_palindrome(s):')..." onkeydown="if(event.key==='Enter') sendPrompt()">
      <button class="send-btn" id="send-button" onclick="sendPrompt()">Generate 🚀</button>
    </div>

    <div class="settings-bar">
      <div class="setting-item">
        <label>Temperature: <span id="temp-val" style="color:#fff;">0.3</span></label>
        <input type="range" id="temp-slider" min="0" max="10" value="3" oninput="document.getElementById('temp-val').textContent=(this.value/10).toFixed(1)">
      </div>
      <div class="setting-item">
        <label>Max Tokens: <span id="tokens-val" style="color:#fff;">96</span></label>
        <input type="range" id="tokens-slider" min="32" max="256" step="16" value="96" oninput="document.getElementById('tokens-val').textContent=this.value">
      </div>
    </div>
  </div>

  <script>
    function usePreset(text) {
      document.getElementById('prompt-input').value = text;
      sendPrompt();
    }

    async function sendPrompt() {
      const input = document.getElementById('prompt-input');
      const text = input.value.trim();
      if (!text) return;

      input.value = '';
      const chatBox = document.getElementById('chat-container');

      // Append user message
      const userMsg = document.createElement('div');
      userMsg.className = 'message user';
      userMsg.innerHTML = '<div class="message-role">You</div><div>' + escapeHtml(text) + '</div>';
      chatBox.appendChild(userMsg);

      // Append assistant placeholder
      const assistMsg = document.createElement('div');
      assistMsg.className = 'message assistant';
      assistMsg.innerHTML = '<div class="message-role">Champion Assistant (M_156)</div><div class="content" style="color:var(--text-muted);">Generating on DGX Blackwell GB10...</div>';
      chatBox.appendChild(assistMsg);
      chatBox.scrollTop = chatBox.scrollHeight;

      const temp = parseFloat(document.getElementById('temp-slider').value) / 10;
      const maxTokens = parseInt(document.getElementById('tokens-slider').value);

      try {
        const resp = await fetch('/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: text, temperature: temp, max_tokens: maxTokens })
        });
        const data = await resp.json();
        
        assistMsg.querySelector('.content').innerHTML = '<pre><code class="language-python">' + escapeHtml(data.response) + '</code></pre>';
        hljs.highlightAll();
      } catch (err) {
        assistMsg.querySelector('.content').innerHTML = '<span style="color:#f87171;">Error generating response: ' + err.message + '</span>';
      }
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    function escapeHtml(text) {
      return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }
  </script>
</body>
</html>
"""

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.3
    max_tokens: int = 96

@app.on_event("startup")
def startup_event():
    global model, tokenizer, device
    print("Initializing Champion Model on DGX GPU...")
    model, tokenizer, device = load_champion_model()
    print(f"Server ready on device [{device}]!")

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

@app.post("/generate")
def generate_api(req: GenerateRequest):
    global model, tokenizer, device
    p = req.prompt.strip()

    # Determine if conversational or raw code
    if p.startswith("def ") or p.startswith("class ") or p.startswith("#"):
        full_prompt = p
        stop_tokens = ["<|endoftext|>"]
    else:
        full_prompt = CHAT_PROMPT_HEADER.format(query=p)
        stop_tokens = ["# User:", "<|endoftext|>"]

    ans = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=full_prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        stop_tokens=stop_tokens,
        device=device
    )

    if not (p.startswith("def ") or p.startswith("class ")):
        final_code = ans.strip()
    else:
        final_code = p + ans

    return JSONResponse({"prompt": req.prompt, "response": final_code})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting Web Chat Server on 0.0.0.0:{port}...")
    uvicorn.run("web_chat:app", host="0.0.0.0", port=port, log_level="info")
