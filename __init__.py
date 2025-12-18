# explain_lines.py
import ast, sys, json, runpy, types, argparse, io, contextlib, linecache, builtins, time
import tkinter as tk
from tkinter import filedialog
import subprocess, textwrap, shutil
from trace import Trace

def browse_files():
    """
    Opens a file selection dialog and returns the selected file's path.
    """
    filename = filedialog.askopenfilename(
        title="Select a File",
        initialdir="/",  # Start directory (use "C:\\" for Windows root)
        filetypes=(
            ("All files", "*.*"),  # Correctly formatted file type
        )
    )

    if filename:
        print("Selected file:", filename)
    return filename


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def ast_facts(src):
    tree = ast.parse(src)
    facts = {}  # line -> list of strings
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", None)
        if not isinstance(ln, int): continue
        items = facts.setdefault(ln, [])
        if isinstance(node, ast.Assign):
            targets = [ast.unparse(t) if hasattr(ast, "unparse") else type(t).__name__ for t in node.targets]
            items.append(f"assign: {', '.join(targets)} = {ast.get_source_segment(src, node.value) or '...'}")
        elif isinstance(node, ast.AugAssign):
            items.append(f"augassign: {ast.get_source_segment(src, node.target)} {type(node.op).__name__}= {ast.get_source_segment(src, node.value)}")
        elif isinstance(node, ast.Call):
            callee = ast.get_source_segment(src, node.func) or 'call'
            items.append(f"call: {callee}(...)")
        elif isinstance(node, ast.Return):
            items.append(f"return: {ast.get_source_segment(src, node.value) if node.value else 'None'}")
        elif isinstance(node, ast.If):
            test = ast.get_source_segment(src, node.test)
            items.append(f"branch-if: {test}")
        elif isinstance(node, ast.For):
            it = ast.get_source_segment(src, node.iter)
            tgt = ast.get_source_segment(src, node.target)
            items.append(f"loop-for: {tgt} in {it}")
        elif isinstance(node, ast.While):
            items.append("loop-while: condition")
        elif isinstance(node, ast.Import):
            names = ", ".join([a.name for a in node.names])
            items.append(f"import: {names}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or "."
            names = ", ".join([a.name for a in node.names])
            items.append(f"from {mod} import {names}")
        elif isinstance(node, ast.With):
            items.append("with: context manager")
        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            items.append(f"def: {node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            items.append(f"class: {node.name}")
    return facts

def run_with_trace(path, entry=None, args=None, kwargs=None, stdin_data=""):
    """
    Safe execution:
      - Load file as a module with run_name="__not_main__" (skips if __name__=="__main__" blocks)
      - Stub input() with provided stdin_data (or empty)
      - Clamp time.sleep() to avoid long waits
      - Trace ONLY the entry() call (not import-time code)
    """
    args = args or []
    kwargs = kwargs or {}

    # 1) Import without triggering __main__ blocks
    globs = runpy.run_path(path, run_name="__not_main__")

    # 2) Build stubs
    fake_stdin = io.StringIO(stdin_data)
    original_input = builtins.input
    original_sleep = time.sleep

    def stub_input(prompt=None):
        line = fake_stdin.readline()
        return line.rstrip("\n") if line else ""

    def fast_sleep(s):
        # cap to 10ms so "waits" don't stall tracing
        original_sleep(min(float(s), 0.01))

    # 3) Trace only the entry call
    tracer = Trace(count=True, trace=False, ignoremods=set())

    out_buf = io.StringIO()
    try:
        builtins.input = stub_input
        time.sleep = fast_sleep

        if entry:
            func = globs.get(entry)
            if not callable(func):
                counts = {}
                return {}, ""
            with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(out_buf), contextlib.redirect_stdin(fake_stdin):
                tracer.runfunc(func, *args, **kwargs)
        else:
            counts = {}
            return {}, out_buf.getvalue()

    finally:
        builtins.input = original_input
        time.sleep = original_sleep

    results = tracer.results()
    counts = {ln: hits for (fname, ln), hits in results.counts.items() if fname.endswith(path)}
    return counts, out_buf.getvalue()

def explain_file(path, entry=None, args=None, kwargs=None):
    src = read_file(path)
    facts = ast_facts(src)
    hits, out = run_with_trace(path, entry, args, kwargs)

    # Precompute facts and hits for faster access
    fact_lines = set(facts.keys())
    hit_lines = set(hits.keys())

    explanations = []
    lines = src.splitlines()

    for i, text in enumerate(lines, start=1):
        if not text.strip():
            continue  # Skip blank lines early

        line_info = []
        if i in fact_lines:
            line_info.extend(facts[i])
        if i in hit_lines:
            line_info.append(f"runtime: executed {hits[i]}x")

        if not line_info:
            line_info = ["(no AST events; likely comment/blank or simple expression)"]

        explanations.append({
            "line": i,
            "code": text.rstrip(),
            "facts": line_info
        })

    return explanations, out

# ---- CONFIG ----
MODEL = "mymodel:latest"  # Ollama tag
USE_OLLAMA = shutil.which("ollama") is not None

SYSTEM_PROMPT = """You are a code explainer.
You will be given, for each line: the original code and a list of FACTS derived from AST/runtime.
RULES:
- ONLY restate those facts. Do not invent behavior or values not present in facts.
- If something is unknown, say 'unknown from provided facts'.
- Keep explanations short (1–2 sentences per line).
- Include the line number.
- If a potential issue is EVIDENT from facts (e.g., never executed, risky IO, untrusted network, high fail rate), emit a 'red_flag' with a one-line reason.
- Output strict JSON: { "lines": [ { "line": <int>, "explanation": "<string>", "red_flags": ["..."] } ] }"""

USER_TEMPLATE = """CODE FACTS:
{items}
Now produce the JSON as specified.
"""

def call_local_llm(prompt):
    if USE_OLLAMA:
        retries = 2  # Number of retries
        for attempt in range(retries + 1):
            try:
                # Use Ollama with a longer timeout to prevent hanging
                proc = subprocess.run(
                    ["ollama", "run", MODEL],
                    input=prompt.encode("utf-8"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=600  # Timeout after 60 seconds
                )
                if proc.returncode != 0:
                    print("Error running Ollama:", proc.stderr.decode("utf-8"))
                    raise RuntimeError("Ollama command failed. Check the error above.")
                return proc.stdout.decode("utf-8")
            except subprocess.TimeoutExpired:
                if attempt < retries:
                    print(f"Attempt {attempt + 1} failed due to timeout. Retrying...")
                else:
                    raise SystemExit("Ollama command timed out after multiple attempts. Ensure the model is available and try again.")
    else:
        raise SystemExit("No local LLM runner found (expected Ollama). Install and retry.")

def build_user_prompt(facts):
    blocks = []
    for it in facts:
        block = {
            "line": it["line"],
            "code": it["code"],
            "facts": it["facts"]
        }
        blocks.append(json.dumps(block, ensure_ascii=False))
    return USER_TEMPLATE.format(items="\n".join(blocks))

def explain_with_llm(facts):
    # Pre-flagging (deterministic warnings) → append to facts for LLM to surface
    for it in facts:
        ff = []
        if not any("runtime: executed" in f for f in it["facts"]):
            ff.append("never_executed_in_trace")
        if any("writes_to_network" in f or "scrape_untrusted" in f for f in it["facts"]):
            ff.append("risky_io_or_network")
        if any("execute_code" in f for f in it["facts"]):
            ff.append("dynamic_code_execution")
        if ff:
            it["facts"].append(f"precheck_flags: {','.join(ff)}")

    user = build_user_prompt(facts)
    full_prompt = f"system\n{SYSTEM_PROMPT}\n\nuser\n{user}\n"

    raw = call_local_llm(full_prompt)
    # naive clean-up: find first JSON block
    start = raw.find("{")
    end = raw.rfind("}")
    cleaned = raw[start:end+1] if start != -1 and end != -1 else raw
    data = json.loads(cleaned)  # will raise if malformed; okay—fail loud

    return data

if __name__ == "__main__":
    print("Please select a file to analyze.")
    path = browse_files()
    if not path:
        print("No file selected. Exiting.")
        sys.exit(1)

    entry = None  # Default entry point
    args = []     # Default positional arguments
    kwargs = {}   # Default keyword arguments

    expl, captured = explain_file(path, entry, args, kwargs)
    for item in expl:
        print(f"[L{item['line']}] {item['code']}")
        for f in item["facts"]:
            print(f"   - {f}")
    if captured.strip():
        print("\n--- program output ---")
        print(captured.strip())

    # Optionally use LLM for enhanced explanations
    use_llm = input("Would you like to use the LLM for enhanced explanations? (y/n): ").strip().lower()
    if use_llm == 'y':
        print("Using LLM for enhanced explanations...")
        llm_explanations = explain_with_llm(expl)
        print(json.dumps(llm_explanations, indent=2, ensure_ascii=False))
