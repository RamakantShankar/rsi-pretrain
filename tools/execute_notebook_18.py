import json
import io
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

cur = Path.cwd().resolve()
if (cur / "notebooks").exists():
    nb_path = cur / "notebooks/18_codeact_agentic_tool_use.ipynb"
elif cur.name == "notebooks":
    nb_path = cur / "18_codeact_agentic_tool_use.ipynb"
else:
    nb_path = Path("notebooks/18_codeact_agentic_tool_use.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

global_ns = {}
execution_count = 1

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        cell["execution_count"] = execution_count
        execution_count += 1
        print(f"Executing cell {cell['execution_count']}...")
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, global_ns)
            out_text = stdout_buf.getvalue()
            err_text = stderr_buf.getvalue()
            outputs = []
            if out_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_text.splitlines(keepends=True)
                })
            if err_text:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": err_text.splitlines(keepends=True)
                })
            cell["outputs"] = outputs
            print(f" Cell {cell['execution_count'] - 1} finished successfully.")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f" ERROR in cell {cell['execution_count'] - 1}:\n{tb}")
            cell["outputs"] = [{
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": tb.splitlines(keepends=True)
            }]
            break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("[+] Executed notebook 18 written back successfully.")
