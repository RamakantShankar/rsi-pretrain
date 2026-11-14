"""
CodeAct Agent: Closed-Loop Tool Execution Engine for Rama-84M.
Allows the model to propose Python code, execute it in a sandboxed REPL,
observe execution outputs via <|tool_output|>, and synthesize final answers.
"""

import sys
import subprocess
import torch
from typing import Dict, Any, Optional
from tokenizers import Tokenizer
from src.model.looped_transformer import LoopedTransformerDecoder


class CodeActAgent:
    def __init__(self, model: LoopedTransformerDecoder, tokenizer: Tokenizer, device: torch.device):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.im_start_id = tokenizer.token_to_id("<|im_start|>")
        self.im_end_id = tokenizer.token_to_id("<|im_end|>")
        self.tool_call_id = tokenizer.token_to_id("<|tool_call|>")
        self.tool_output_id = tokenizer.token_to_id("<|tool_output|>")
        self.eot_id = tokenizer.token_to_id("<|endoftext|>")
        self.stop_ids = {self.im_end_id, self.eot_id}

    def extract_code(self, text: str) -> str:
        """Extracts executable python statements from generated text."""
        lines = []
        in_code = False
        for line in text.splitlines():
            s = line.strip()
            if not s:
                if in_code:
                    lines.append(line)
                continue
            first = s.split()[0].split("(")[0].split("=")[0]
            if first in {"def", "import", "from", "for", "while", "if", "print", "result", "primes", "fib", "a", "b", "c", "x", "y", "n", "return", "total", "count"}:
                in_code = True
            if in_code:
                # Stop at bullet points or non-code summary
                if s[0].isdigit() and ("." in s[:3] or "," in s[:3]):
                    break
                if s.startswith("The ") or s.startswith("Therefore"):
                    if not any(k in s for k in ["=", "(", ")", "+", "*"]):
                        break
                lines.append(line)
        return "\n".join(lines).strip()

    def execute_python_code(self, code_str: str, timeout: float = 3.0) -> str:
        """Executes Python code in an isolated subprocess with timeout."""
        clean_code = self.extract_code(code_str)
        if not clean_code:
            return "(No executable code detected)"

        try:
            res = subprocess.run(
                [sys.executable, "-c", clean_code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            out = res.stdout.strip()
            err = res.stderr.strip()
            if err:
                return f"Error: {err.splitlines()[-1]}"
            return out if out else "(Executed successfully with no output)"
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out (3.0s)"
        except Exception as e:
            return f"Error: {str(e)}"

    def solve(
        self,
        query: str,
        max_action_tokens: int = 96,
        max_answer_tokens: int = 64,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Closed-Loop Agentic Cycle:
        1. Query Priming -> Assistant starts generating thought + code.
        2. Detection & Interception -> Code snippet extracted and executed in REPL.
        3. Environment Observation -> <|tool_output|>\n{observation}\n appended to context.
        4. Final Synthesis -> Model conditioned on live observation outputs final answer.
        """
        # We prime with assistant initiation
        prompt = f"<|im_start|>user\n{query.strip()}<|im_end|>\n<|im_start|>assistant\n"
        enc = self.tok.encode(prompt).ids
        x = torch.tensor([enc], device=self.device)

        # 1. Action Generation Phase
        gen_action = []
        action_stops = {self.tool_output_id, self.im_end_id, self.eot_id}
        with torch.no_grad():
            for _ in range(max_action_tokens):
                logits, _ = self.model(x, num_loops=2)
                tok_id = int(torch.argmax(logits[:, -1, :]).item())
                if tok_id in action_stops:
                    break
                gen_action.append(tok_id)
                x = torch.tensor([enc + gen_action], device=self.device)

        raw_action = self.tok.decode(gen_action)
        code_snippet = self.extract_code(raw_action)
        
        # Fallback if assistant stopped early on math query
        if not code_snippet:
            # Prime with explicit tool call initiation
            tool_prompt = prompt + "<|tool_call|>\n"
            enc_tp = self.tok.encode(tool_prompt).ids
            x_tp = torch.tensor([enc_tp], device=self.device)
            gen_tp = []
            with torch.no_grad():
                for _ in range(max_action_tokens):
                    logits, _ = self.model(x_tp, num_loops=2)
                    tok_id = int(torch.argmax(logits[:, -1, :]).item())
                    if tok_id in action_stops:
                        break
                    gen_tp.append(tok_id)
                    x_tp = torch.tensor([enc_tp + gen_tp], device=self.device)
            raw_action = self.tok.decode(gen_tp)
            code_snippet = self.extract_code(raw_action)

        if verbose:
            print("=" * 65)
            print(f"QUERY: {query}")
            print("-" * 65)
            print(f"[1. MODEL GENERATED CODE]:\n{code_snippet}")

        # 2. Execution Phase: Local Python REPL
        observation = self.execute_python_code(code_snippet)
        if verbose:
            print(f"\n[2. ENVIRONMENT OBSERVATION]:\n{observation}")

        # 3. Context Update: Observation Injection
        followup_context = prompt + f"<|tool_call|>\n{code_snippet}\n<|tool_output|>\n{observation}\n"
        enc_followup = self.tok.encode(followup_context).ids
        x_followup = torch.tensor([enc_followup], device=self.device)

        # 4. Final Answer Phase
        gen_answer = []
        with torch.no_grad():
            for _ in range(max_answer_tokens):
                logits, _ = self.model(x_followup, num_loops=2)
                tok_id = int(torch.argmax(logits[:, -1, :]).item())
                if tok_id in self.stop_ids:
                    break
                gen_answer.append(tok_id)
                x_followup = torch.tensor([enc_followup + gen_answer], device=self.device)

        final_answer = self.tok.decode(gen_answer).replace("<|im_end|>", "").strip()

        if verbose:
            print(f"\n[3. FINAL SYNTHESIZED ANSWER]:\n{final_answer}")
            print("=" * 65 + "\n")

        return {
            "query": query,
            "generated_code": code_snippet,
            "observation": observation,
            "final_answer": final_answer,
            "full_transcript": followup_context + final_answer + "<|im_end|>"
        }
