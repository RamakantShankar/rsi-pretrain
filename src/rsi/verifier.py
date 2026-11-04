"""
Sandboxed Unit-Test Verifier for Recursive Self-Improvement (RSI).
Executes candidate code solutions in isolated Python subprocesses with
strict CPU execution timeouts and exception capture.
"""

import sys
import time
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    passed: bool
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    execution_time_sec: float = 0.0


class CodeVerifier:
    def __init__(self, timeout_sec: float = 2.0):
        self.timeout_sec = timeout_sec

    def verify(self, code_solution: str, unit_tests: List[str]) -> VerificationResult:
        """
        Executes code_solution + unit_tests in a temporary sandboxed Python subprocess.
        """
        full_code = code_solution.rstrip() + "\n\n" + "\n".join(unit_tests) + "\n"

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(full_code)
            temp_path = f.name

        start_t = time.time()
        try:
            # Execute in isolated subprocess with timeout
            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )
            elapsed = time.time() - start_t

            if proc.returncode == 0:
                return VerificationResult(passed=True, execution_time_sec=elapsed)

            # Analyze stderr failure
            stderr = proc.stderr
            if "SyntaxError" in stderr:
                err_type = "SYNTAX_ERROR"
            elif "AssertionError" in stderr:
                err_type = "ASSERTION_ERROR"
            elif "NameError" in stderr:
                err_type = "NAME_ERROR"
            elif "TypeError" in stderr:
                err_type = "TYPE_ERROR"
            else:
                err_type = "RUNTIME_ERROR"

            # Grab the last relevant line of the traceback
            lines = [line.strip() for line in stderr.splitlines() if line.strip()]
            summary_msg = lines[-1] if lines else "Process failed with non-zero exit"

            return VerificationResult(
                passed=False,
                error_type=err_type,
                error_msg=summary_msg,
                execution_time_sec=elapsed
            )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                error_type="TIMEOUT",
                error_msg=f"Execution exceeded {self.timeout_sec}s limit (infinite loop)",
                execution_time_sec=self.timeout_sec
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                error_type="SYSTEM_ERROR",
                error_msg=str(e),
                execution_time_sec=time.time() - start_t
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


if __name__ == "__main__":
    verifier = CodeVerifier(timeout_sec=1.0)
    print("=" * 60)
    print("TESTING SANDBOXED CODE VERIFIER")
    print("=" * 60)

    # 1. Test Passing Solution
    res1 = verifier.verify(
        "def add(a, b):\n    return a + b",
        ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]
    )
    print(f"1. Passing Test:        {res1.passed} (Time: {res1.execution_time_sec:.3f}s)")
    assert res1.passed

    # 2. Test Assertion Failure
    res2 = verifier.verify(
        "def add(a, b):\n    return a * b",
        ["assert add(2, 3) == 5"]
    )
    print(f"2. Assertion Failure:   Passed={res2.passed} | Error={res2.error_type} | Msg={res2.error_msg}")
    assert not res2.passed and res2.error_type == "ASSERTION_ERROR"

    # 3. Test Syntax Error
    res3 = verifier.verify(
        "def broken(\n  return",
        ["assert True"]
    )
    print(f"3. Syntax Error:        Passed={res3.passed} | Error={res3.error_type} | Msg={res3.error_msg}")
    assert not res3.passed and res3.error_type == "SYNTAX_ERROR"

    # 4. Test Infinite Loop Timeout
    res4 = verifier.verify(
        "def loop():\n    while True:\n        pass\nloop()",
        ["assert True"]
    )
    print(f"4. Infinite Loop:       Passed={res4.passed} | Error={res4.error_type} | Msg={res4.error_msg}")
    assert not res4.passed and res4.error_type == "TIMEOUT"

    print("=" * 60)
    print("ALL VERIFIER SANDBOX CHECKS PASSED PERFECTLY!")
    print("=" * 60)
