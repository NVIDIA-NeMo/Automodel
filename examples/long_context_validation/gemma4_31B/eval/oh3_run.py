# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenHands-tool-surface SWE-bench agent (approach B).

Gives the model the EXACT 3-tool surface OpenHands v0.52.1 uses — `execute_bash`,
`str_replace_editor`, `finish` (schemas + names copied verbatim from OpenHands 0.52.1) —
which is what the CoderForge SFT data was generated with. This removes the tool-surface
mismatch that crippled the SFT model under mini-swe-agent's single `bash` tool, without the
cost/risk of running the full OpenHands runtime inside each container.

Execution reuses the proven Docker-less enroot backend (`enroot_env.EnrootEnvironment`): each
instance gets its official SWE-bench image container; `execute_bash` runs bash in `/testbed`
(conda `testbed` active via BASH_ENV), and `str_replace_editor` runs a self-contained editor
(view/create/str_replace/insert/undo) in-container. Output is a `preds.json` in the standard
SWE-bench format, graded by the same local grader (`grade_enroot.py`).
"""

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
import urllib.request

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enroot_env import EnrootEnvironment  # noqa: E402

DATASET_MAP = {
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "full": "princeton-nlp/SWE-Bench",
}

# --- OpenHands 0.52.1 tool schemas (verbatim names/params) ---
EXECUTE_BASH = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": "Execute a bash command in the terminal within a persistent shell session. One command at a time; chain with && or ;. Commands run in /testbed with the project's environment active.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute."},
                "is_input": {
                    "type": "string",
                    "enum": ["true", "false"],
                    "description": "If true, send input to a running process.",
                },
                "timeout": {"type": "number", "description": "Optional hard timeout in seconds."},
            },
            "required": ["command"],
        },
    },
}
STR_REPLACE_EDITOR = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": (
            "Custom editing tool for viewing, creating and editing files.\n"
            "* `view` shows `cat -n` for a file (optional view_range) or a 2-level listing for a directory.\n"
            "* `create` creates a new file (fails if it exists) with `file_text`.\n"
            "* `str_replace` replaces `old_str` (must match EXACTLY one unique location, whitespace included) with `new_str`.\n"
            "* `insert` inserts `new_str` after line `insert_line`.\n"
            "* `undo_edit` reverts the last edit to `path`.\n"
            "Always use absolute paths (e.g. /testbed/pkg/mod.py)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "description": "The command to run.",
                },
                "path": {"type": "string", "description": "Absolute path to file or directory."},
                "file_text": {"type": "string", "description": "Content for `create`."},
                "old_str": {"type": "string", "description": "String to replace (for `str_replace`)."},
                "new_str": {
                    "type": "string",
                    "description": "Replacement string (`str_replace`) or text to insert (`insert`).",
                },
                "insert_line": {"type": "integer", "description": "Line after which to insert (`insert`)."},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [start, end] line range for `view`.",
                },
            },
            "required": ["command", "path"],
        },
    },
}
FINISH = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Signals completion of the task. Call this once you have made and verified the fix.",
        "parameters": {
            "type": "object",
            "required": ["message"],
            "properties": {"message": {"type": "string", "description": "Final summary."}},
        },
    },
}
# CoderForge SFT trajectories were generated with a 4th tool, `think` (verbatim OpenHands 0.52.1
# schema), and the model's trained first action is almost always a `think` call. Omitting it forces
# the think-then-act model to route reasoning into an action's arg value on turn>=2 -> spurious
# tokens (`mimeTypes`, CJK) prepended to the command/path -> every post-turn-1 call fails. Restoring
# `think` gives the reasoning a home so action args stay clean. Result: "Your thought has been logged."
THINK = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Use the tool to think about something. It will not obtain new information or make any "
            "changes to the repository, but just log the thought. Use it when complex reasoning or "
            "brainstorming is needed.\n\n"
            "Common use cases:\n"
            "1. When exploring a repository and discovering the source of a bug, call this tool to "
            "brainstorm several unique ways of fixing the bug, and assess which change(s) are likely "
            "to be simplest and most effective.\n"
            "2. After receiving test results, use this tool to brainstorm ways to fix failing tests.\n"
            "3. When planning a complex refactoring, use this tool to outline different approaches and "
            "their tradeoffs.\n"
            "4. When designing a new feature, use this tool to think through architecture decisions and "
            "implementation details.\n"
            "5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.\n\n"
            "The tool simply logs your thought process for better transparency and does not execute any "
            "code or make changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {"thought": {"type": "string", "description": "The thought to log."}},
            "required": ["thought"],
        },
    },
}
TOOLS = [EXECUTE_BASH, STR_REPLACE_EDITOR, FINISH, THINK]

# System + user prompts copied VERBATIM from the CoderForge SFT data (the OpenHands 0.52.1 prompts the
# trajectories were generated with). Using the exact train-time prompts removes the distribution shift
# that the earlier minimal oh3 prompts introduced (the model already follows this phased workflow — it
# emits "## Phase 1. READING" straight from the user template below).
SYSTEM_PROMPT = """You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
* If the user asks a question, like "why is X happening", don't try to fix the problem. Just give an answer to the question.
</ROLE>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.
</CODE_QUALITY>

<VERSION_CONTROL>
* When configuring git credentials, use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.
* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.
* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.
* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.
</VERSION_CONTROL>

<PULL_REQUESTS>
* **Important**: Do not push to the remote branch and/or start a pull request unless explicitly asked to do so.
* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.
* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.
* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.
</PULL_REQUESTS>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING:
   * For bug fixes: Create tests to verify issues before implementing fixes
   * For new features: Consider test-driven development when appropriate
   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure
   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies
4. IMPLEMENTATION: Make focused, minimal changes to address the problem
5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.
</PROBLEM_SOLVING_WORKFLOW>

<SECURITY>
* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.
* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.
</SECURITY>

<ENVIRONMENT_SETUP>
* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.
* If you encounter missing dependencies:
  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)
  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)
  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed
* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.
</ENVIRONMENT_SETUP>

<TROUBLESHOOTING>
* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:
  1. Step back and reflect on 5-7 different possible sources of the problem
  2. Assess the likelihood of each possible cause
  3. Methodically address the most likely causes, starting with the highest probability
  4. Document your reasoning process
* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.
</TROUBLESHOOTING>
"""

USER_TEMPLATE = """<uploaded_files>
/testbed
</uploaded_files>

I've uploaded a python code repository in the directory /testbed. Consider the following issue description:

<issue_description>
{problem}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /testbed directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Hightlight any best practices to take into account when testing and fixing the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme
   2.2 Install the environment and anything needed
   2.2 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   4.1 Look at existing test files in the repository to understand the test format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.


8. FINAL REVIEW: Carefully re-read the problem description and verify your changes address all requirements.

   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
"""

# --- Backtick-isolation toggles (for the -it re-eval) ---
# The verbatim CoderForge SYSTEM_PROMPT is markdown-heavy (backticks around `command`/paths); the
# -it model mirrors that and emits backtick-wrapped tool args (`{"`command`": "`view`"}`) that the
# gemma4 parser rejects. OH3_MINIMAL_PROMPT swaps in a plain, backtick-free prompt (close to the
# original oh3 config that gave the 25%). OH3_NO_THINK drops the think tool. Base-SFT is unaffected
# (it needs neither; leave both unset for base runs).
MINIMAL_SYSTEM_PROMPT = (
    "You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.\n"
    "You are working in a repository checked out at /testbed. Use the tools to explore, edit, and test.\n"
    "* Make focused, minimal changes to the source files to fix the issue; do not modify tests.\n"
    "* Do NOT modify build/config files (pyproject.toml, setup.py, setup.cfg, tox.ini) as they are not part of the fix.\n"
    "* First reproduce or understand the issue, then fix it, then verify.\n"
    "* Every response must include at least one tool call. When the fix is complete and verified, call finish.\n"
    "* When you pass arguments to a tool, use plain values with no surrounding backticks or markdown formatting."
)
if os.environ.get("OH3_MINIMAL_PROMPT") == "1":
    SYSTEM_PROMPT = MINIMAL_SYSTEM_PROMPT
if os.environ.get("OH3_NO_THINK") == "1":
    TOOLS = [EXECUTE_BASH, STR_REPLACE_EDITOR, FINISH]

# Self-contained str_replace_editor executed inside the container (stdlib only).
EDITOR_PY = r"""
import json, os, sys, hashlib
a = json.load(open(sys.argv[1]))
def out(s): sys.stdout.write(s)
# Validate required args and return CLEAN, actionable errors (never a raw traceback) so the
# model can self-correct. Malformed tool-call JSON (e.g. wrong delimiters -> missing keys) was
# causing unrecoverable retry loops when the editor crashed with KeyError.
cmd = a.get("command")
if cmd not in ("view", "create", "str_replace", "insert", "undo_edit"):
    out("Error: missing/invalid 'command'. Received keys=%s. Send valid JSON args, e.g. "
        '{"command":"str_replace","path":"/testbed/f.py","old_str":"...","new_str":"..."}' % list(a.keys())); sys.exit(1)
path = a.get("path")
if not path:
    out("Error: 'path' is required (absolute, e.g. /testbed/pkg/mod.py). Received keys=%s. "
        "Your previous tool call likely had malformed JSON; resend valid JSON." % list(a.keys())); sys.exit(1)
_req = {"str_replace": ["old_str"], "insert": ["insert_line", "new_str"], "create": ["file_text"]}.get(cmd, [])
_missing = [k for k in _req if a.get(k) in (None, "")]
if _missing:
    out("Error: '%s' command requires %s but got keys=%s. Resend valid JSON args." % (cmd, _missing, list(a.keys()))); sys.exit(1)
# Keep the undo backup OUTSIDE the repo (/tmp) so it never leaks into the git diff / patch.
bak = "/tmp/oh3bak_" + hashlib.md5(path.encode()).hexdigest()
try:
    # Observation strings match the CoderForge train-time OpenHands 0.52.1 editor format EXACTLY
    # (prefixes "Here's the result of running `cat -n`...", "File created successfully at:", "The file
    # X has been edited...") to avoid the turn>=2 distribution shift that corrupts arg-value tokens.
    if cmd == "view":
        if os.path.isdir(path):
            r = []
            for root, dirs, files in os.walk(path):
                depth = root[len(path):].count(os.sep)
                if depth >= 2: dirs[:] = []
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    if not f.startswith("."): r.append(os.path.join(root, f))
            out("Here's the files and directories up to 2 levels deep in %s, excluding hidden items:\n%s"
                % (path, "\n".join(sorted(r)[:400]))); sys.exit(0)
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
        vr = a.get("view_range")
        s, e = (1, len(lines))
        if vr: s = vr[0]; e = len(lines) if vr[1] == -1 else vr[1]
        body = "\n".join("%6d\t%s" % (i, lines[i-1]) for i in range(s, min(e, len(lines)) + 1))
        out("Here's the result of running `cat -n` on %s:\n%s" % (path, body))
    elif cmd == "create":
        if os.path.exists(path): out("ERROR: file exists"); sys.exit(1)
        open(path, "w", encoding="utf-8").write(a.get("file_text", "")); out("File created successfully at: " + path)
    elif cmd == "str_replace":
        data = open(path, encoding="utf-8", errors="replace").read()
        old = a["old_str"]; n = data.count(old)
        if n == 0: out("ERROR: old_str not found"); sys.exit(1)
        if n > 1: out("ERROR: old_str not unique (%d matches)" % n); sys.exit(1)
        open(bak, "w", encoding="utf-8").write(data)
        new_str = a.get("new_str", "")
        new_data = data.replace(old, new_str)
        open(path, "w", encoding="utf-8").write(new_data)
        pos = data.find(old); start_line = data[:pos].count("\n") + 1
        nl = new_data.split("\n"); s2 = max(1, start_line - 2); e2 = min(len(nl), start_line + new_str.count("\n") + 3)
        snip = "\n".join("%6d\t%s" % (i, nl[i-1]) for i in range(s2, e2 + 1))
        out("The file %s has been edited. Here's the result of running `cat -n` on a snippet of %s:\n%s" % (path, path, snip))
    elif cmd == "insert":
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
        il = a["insert_line"]
        open(bak, "w", encoding="utf-8").write("\n".join(lines))
        new_str = a.get("new_str", "")
        lines.insert(il, new_str); new_data = "\n".join(lines); open(path, "w", encoding="utf-8").write(new_data)
        nl = new_data.split("\n"); s2 = max(1, il - 2); e2 = min(len(nl), il + new_str.count("\n") + 3)
        snip = "\n".join("%6d\t%s" % (i, nl[i-1]) for i in range(s2, e2 + 1))
        out("The file %s has been edited. Here's the result of running `cat -n` on a snippet of %s:\n%s" % (path, path, snip))
    elif cmd == "undo_edit":
        if not os.path.exists(bak): out("ERROR: nothing to undo"); sys.exit(1)
        open(path, "w", encoding="utf-8").write(open(bak, encoding="utf-8").read()); out("Reverted " + path)
    else:
        out("ERROR: unknown command"); sys.exit(1)
except Exception as e:
    out("ERROR: %s" % e); sys.exit(1)
"""


def run_editor(env, args, timeout):
    """Execute an OpenHands ``str_replace_editor`` action inside the instance container."""
    b64py = base64.b64encode(EDITOR_PY.encode()).decode()
    b64a = base64.b64encode(json.dumps(args).encode()).decode()
    cmd = (
        f"echo {b64py} | base64 -d > /tmp/oh3_ed.py && echo {b64a} | base64 -d > /tmp/oh3_args.json && "
        f"python3 /tmp/oh3_ed.py /tmp/oh3_args.json"
    )
    return env.execute({"command": cmd}, timeout=timeout)


def chat(endpoint, model, messages, max_tokens):
    """Send one chat-completions request to the served model and return the assistant message."""
    # tool_choice: "auto" needs vLLM's --enable-auto-tool-choice. When serving WITHOUT the gemma4
    # parser (NOPARSER, to route raw `call:...` through content for the JSON-aware fallback), set
    # OH3_TOOL_CHOICE=none so vLLM still renders the tools in the prompt but doesn't parse/force them.
    tool_choice = os.environ.get("OH3_TOOL_CHOICE", "auto")
    # Sampling: temperature 0 (greedy) makes a single corrupted/looping call fatal — identical input
    # -> identical broken output -> the model re-emits the same command forever (aborted_stuck). It
    # also drives base-SFT greedy degeneration (repeated `writelines`, stray CJK tokens in arg values
    # at the gemma4 `<|"|>` value boundaries). temperature>0 + a light repetition_penalty breaks both.
    temperature = float(os.environ.get("OH3_TEMPERATURE", "0.4"))
    top_p = float(os.environ.get("OH3_TOP_P", "0.95"))
    rep_pen = float(os.environ.get("OH3_REPETITION_PENALTY", "1.05"))
    # frequency/presence penalties: extra brakes on the deep-turn degeneration (repeated tokens like
    # "roommates roommates...", stray CJK/Devanagari) that corrupts base-SFT arg values on turn 2+.
    freq_pen = float(os.environ.get("OH3_FREQUENCY_PENALTY", "0.0"))
    pres_pen = float(os.environ.get("OH3_PRESENCE_PENALTY", "0.0"))
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    # top_k (vLLM sampling extension): Qwen3 thinking-mode preset wants top_k=20. Off by default (-1).
    _top_k = int(os.environ.get("OH3_TOP_K", "-1"))
    if _top_k > 0:
        payload["top_k"] = _top_k
    # stop_token_ids: gemma4 forces [1,106,50] (its base ckpt's generation_config lacks the turn-stop
    # 106). Other models (Qwen3, ...) use their own eos from generation_config -> set
    # OH3_STOP_TOKEN_IDS="" to omit. Default keeps the gemma4 behavior for backward compat.
    _stop_ids = [int(x) for x in os.environ.get("OH3_STOP_TOKEN_IDS", "1,106,50").split(",") if x.strip()]
    if _stop_ids:
        payload["stop_token_ids"] = _stop_ids
    # chat_template_kwargs {enable_thinking}: MODEL-DEPENDENT. gemma4: the default add_generation_prompt
    # injects a `<|channel>thought` channel CoderForge never used -> base-SFT needs enable_thinking=true
    # (skips it), -it needs false (Google-tuned to tool-call after the channel). Qwen3: controls its
    # native <think> mode. OH3_ENABLE_THINKING = "1"|"0" -> send true/false; "none" -> omit the kwarg
    # entirely (for models whose template doesn't accept it).
    _et = os.environ.get("OH3_ENABLE_THINKING", "1")
    if _et != "none":
        payload["chat_template_kwargs"] = {"enable_thinking": _et == "1"}
    if rep_pen and rep_pen != 1.0:
        payload["repetition_penalty"] = rep_pen  # vLLM sampling extension (not std OpenAI)
    if freq_pen:
        payload["frequency_penalty"] = freq_pen
    if pres_pen:
        payload["presence_penalty"] = pres_pen
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    except urllib.error.HTTPError as he:
        # Surface vLLM's actual error body (e.g. the 400 reason) instead of a bare "HTTP Error 400".
        try:
            detail = he.read().decode("utf-8", "replace")[:600]
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {he.code}: {detail}") from None


def trunc(s, n=None):
    """Truncate a string to ``n`` characters (env-configurable) with an elision marker."""
    # Max observation chars fed back to the model. OH3_MAX_OBS_CHARS shrinks this for the
    # context-trim experiment (test whether deep-turn degeneration is triggered by long tool
    # outputs growing the context, vs raw undertraining). Keeps head+tail around an elision marker.
    if n is None:
        n = int(os.environ.get("OH3_MAX_OBS_CHARS", "10000"))
    s = s or ""
    if len(s) <= n:
        return s
    half = max(1, n // 2)
    return s[:half] + f"\n...[{len(s) - n} chars elided]...\n" + s[-half:]


# Known params per tool (used to slice unquoted args in the text-fallback path below).
_TOOL_PARAMS = {
    "execute_bash": ["command", "is_input", "timeout"],
    "str_replace_editor": ["command", "path", "file_text", "old_str", "new_str", "insert_line", "view_range"],
    "finish": ["message"],
    "think": ["thought"],
}
_INT_PARAMS = {"insert_line"}


def _extract_braced(s, start):
    """s[start] == '{' -> return (inner_text, index_after_matching_close)."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    return s[start + 1 :], len(s)  # unbalanced (e.g. truncated) -> take the rest


def _parse_arg_body(inner, params):
    """Parse the gemma4 arg body `key:value,key2:value2` into a dict.

    The wire format quotes each value with `<|"|>` (token 52), but a frozen-head base model that
    can't emit token 48/49 also can't emit 52, so values arrive UNQUOTED. We locate the known
    `param:` keys (only those valid for this tool) at a top-level boundary and slice each value up
    to the next key. Best-effort: values that themselves contain another param name verbatim can
    mis-split — fine for the dominant single-arg `execute_bash`/`finish`, weaker for multi-arg
    str_replace_editor (which the diagnostic is meant to expose).
    """
    inner = inner.replace('<|"|>', "")
    hits = []
    for p in params:
        for m in re.finditer(r"(?:^|[\{,\n])\s*(" + re.escape(p) + r")\s*:", inner):
            hits.append((m.start(1), p, m.end()))
    hits.sort()
    if not hits:
        return None
    args = {}
    for idx, (_pos, name, valstart) in enumerate(hits):
        end = hits[idx + 1][0] if idx + 1 < len(hits) else len(inner)
        val = inner[valstart:end].strip().rstrip(",").strip()
        if name in _INT_PARAMS:
            try:
                val = int(val)
            except ValueError:
                pass
        args[name] = val
    return args


def parse_text_tool_calls(content):
    """Recover gemma4 `call:name{...}` tool calls emitted as plain text (no <|tool_call> wrapper).

    Returns (synth_tool_calls, content_before_first_call). Empty list if none found. This mirrors
    the served `response_schema` regex (`call:(?P<name>\\w+)(?P<arguments>\\{.*\\})`) but runs
    host-side for checkpoints whose frozen tied head can't emit the 48/49 marker tokens.
    """
    calls = []
    first = None
    for m in re.finditer(r"call:(\w+)\s*\{", content):
        name = m.group(1)
        if name not in _TOOL_PARAMS:
            continue
        inner, _ = _extract_braced(content, m.end() - 1)
        # The base-SFT falls back to its pretrained JSON prior: it emits `call:name{{"k":"v"}}`
        # (gemma4 open-brace + a JSON object) instead of the gemma4-native `{k:<|"|>v<|"|>}`. The
        # balanced-brace extract then yields a valid JSON object -> json.loads it directly. Only if
        # that fails (true gemma4-native args, e.g. from the -it model) do we key-slice.
        args = None
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                args = parsed
        except Exception:
            pass
        if args is None:
            args = _parse_arg_body(inner, _TOOL_PARAMS[name]) or {}
        if first is None:
            first = m.start()
        calls.append(
            {
                "id": f"txt_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    head = content[:first] if first is not None else content
    return calls, head


def parse_hermes_tool_calls(content):
    """Recover Qwen3/Hermes `<tool_call>{"name":..,"arguments":{..}}</tool_call>` calls from content.

    Used when serving WITHOUT vLLM's --tool-call-parser hermes: its parser does a STRICT json.loads
    and rejects the ~1-4% of calls whose code args (str_replace old_str/new_str, file_text) contain
    RAW control characters (unescaped newlines) -> the whole tool call is dropped. Parsing host-side
    with json.loads(strict=False) tolerates those control chars, recovering the dropped actions.
    Returns a list of tool_call dicts (same shape vLLM would emit); empty if none.
    """
    calls = []
    for i, m in enumerate(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)):
        try:
            obj = json.loads(m.group(1), strict=False)  # strict=False allows control chars in strings
        except Exception:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args)
        calls.append({"id": f"herm_{i}", "type": "function", "function": {"name": obj["name"], "arguments": args}})
    return calls


def solve(inst, endpoint, model, max_iter, max_tokens, cmd_timeout):
    """Run the OpenHands 3-tool agent loop on one SWE-bench instance; return its predicted patch."""
    iid = inst["instance_id"]
    image = "docker://swebench/sweb.eval.x86_64." + iid.replace("__", "_1776_").lower() + ":latest"
    env = EnrootEnvironment(image=image, cwd="/testbed", env={"BASH_ENV": "/root/.bashrc"}, timeout=cmd_timeout)
    # Failure-mode stats (Check 3): how the run behaves, not just the patch.
    stats = {"turns": 0, "tools": {}, "finished": False, "no_toolcall_turns": 0, "unknown_tools": {}}
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(problem=inst["problem_statement"])},
        ]
        finished = False
        # Loop-break-on-repeat: when the model repeats the exact same tool call (or piles up
        # consecutive format-errors) it makes no progress. Rather than immediately aborting the
        # episode, first NUDGE it (inject a corrective user turn + reset the counters) so it can
        # escape the loop — combined with temperature>0 this usually recovers. Only hard-abort once
        # the nudge budget is spent or a much higher hard cap is hit. All thresholds env-tunable.
        stuck = False
        last_sig = None
        consec_sig = 0
        consec_err = 0
        nudges_used = 0
        pending_nudge = None
        REPEAT_NUDGE = int(os.environ.get("OH3_REPEAT_NUDGE", "3"))
        ERROR_NUDGE = int(os.environ.get("OH3_ERROR_NUDGE", "5"))
        HARD_REPEAT = int(os.environ.get("OH3_HARD_REPEAT", "8"))
        HARD_ERROR = int(os.environ.get("OH3_HARD_ERROR", "12"))
        MAX_NUDGES = int(os.environ.get("OH3_MAX_NUDGES", "3"))

        def track_stuck(fn, raw_args, content):
            nonlocal last_sig, consec_sig, consec_err, stuck, pending_nudge
            sig = (fn, raw_args)
            consec_sig = consec_sig + 1 if sig == last_sig else 1
            last_sig = sig
            consec_err = consec_err + 1 if str(content).startswith("Error:") else 0
            if consec_sig >= HARD_REPEAT or consec_err >= HARD_ERROR:
                stuck = True
                stats["aborted_stuck"] = f"consec_identical={consec_sig},consec_error={consec_err}"
            elif consec_sig >= REPEAT_NUDGE or consec_err >= ERROR_NUDGE:
                pending_nudge = (
                    f"You have repeated the same tool call {consec_sig}x / hit {consec_err} consecutive "
                    "errors with no progress. STOP repeating it. Re-read the last tool output, then try a "
                    "DIFFERENT command or approach (inspect the file with str_replace_editor view, run a "
                    "different search, etc.). Do NOT resend the previous command verbatim."
                )

        consec_llm_err = 0
        for _ in range(max_iter):
            stats["turns"] += 1
            try:
                resp = chat(endpoint, model, messages, max_tokens)
                consec_llm_err = 0
            except Exception as e:
                # Abort on repeated LLM/API errors (e.g. a context-length 400 that will only recur as
                # the history grows) instead of burning the whole turn budget on dead retries.
                consec_llm_err += 1
                stats["llm_errors"] = stats.get("llm_errors", 0) + 1
                if consec_llm_err >= 4:
                    stats["aborted_llm_error"] = str(e)[:300]
                    break
                messages.append({"role": "user", "content": f"[LLM error: {e}] Continue with one tool call."})
                continue
            msg = resp["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            # Text-fallback: if vLLM's gemma4 parser produced no structured tool_calls, recover any
            # `call:name{...}` the model emitted as plain text (checkpoints whose frozen tied head
            # can't emit the <|tool_call> 48/49 markers the parser regex needs). Harmless for models
            # that already emit structured calls (this only runs when tcs is empty).
            if not tcs and "call:" in content:
                synth, content = parse_text_tool_calls(content)
                if synth:
                    tcs = synth
                    stats["text_fallback_turns"] = stats.get("text_fallback_turns", 0) + 1
                    stats["text_fallback_calls"] = stats.get("text_fallback_calls", 0) + len(synth)
            # Hermes fallback (Qwen3): when serving WITHOUT the server-side hermes parser, the raw
            # <tool_call>{...}</tool_call> arrives in content; parse it host-side with strict=False to
            # recover the calls the strict server parser would drop on control-char code args.
            if not tcs and "<tool_call>" in content:
                synth = parse_hermes_tool_calls(content)
                if synth:
                    tcs = synth
                    stats["hermes_fallback_turns"] = stats.get("hermes_fallback_turns", 0) + 1
                    stats["hermes_fallback_calls"] = stats.get("hermes_fallback_calls", 0) + len(synth)
                    content = content.split("<tool_call>")[0]  # keep pre-call text as the content
            # Qwen3 thinking-mode: strip the private <think>...</think> chain before storing in
            # history — its multi-turn guidance is that prior reasoning must NOT be fed back (only the
            # final response), else later turns degrade. No-op when thinking is off (no <think> tags).
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            messages.append({"role": "assistant", "content": content, "tool_calls": tcs})
            if not tcs:
                stats["no_toolcall_turns"] += 1
                messages.append(
                    {
                        "role": "user",
                        "content": "Your response had no tool call. Every response MUST include at least one tool call (execute_bash, str_replace_editor, or finish).",
                    }
                )
                continue
            for tc in tcs:
                fn = tc["function"]["name"]
                stats["tools"][fn] = stats["tools"].get(fn, 0) + 1
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}", strict=False)
                    parse_ok = isinstance(args, dict)
                except Exception:
                    args = {}
                    parse_ok = False
                if not parse_ok:
                    stats["bad_json"] = stats.get("bad_json", 0) + 1
                    bad = (
                        "Error: tool-call arguments were not valid JSON (truncated or wrong delimiters). "
                        "Resend ONE tool call with valid JSON; keep old_str/new_str short and exact."
                    )
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": bad})
                    track_stuck(fn, tc["function"].get("arguments", ""), bad)
                    continue
                if fn == "think":
                    content = "Your thought has been logged."
                elif fn == "finish":
                    finished = True
                    content = "Task marked finished."
                elif fn == "execute_bash":
                    o = env.execute({"command": args.get("command", "")}, timeout=args.get("timeout") or cmd_timeout)
                    # Match the CoderForge train-time observation format EXACTLY (OpenHands 0.52.1 bash
                    # footer). The prior `<returncode>/<output>` XML wrapper was off-distribution and
                    # perturbed the model on turn>=2 (the first gen after an observation) -> spurious
                    # value-open tokens in the next call's args (FindAction:/DebugType:/CJK).
                    rc = o["returncode"]
                    content = (
                        f"{trunc(o['output'])}\n"
                        f"[The command completed with exit code {rc}.]\n"
                        f"[Current working directory: /testbed]\n"
                        f"[Python interpreter: /opt/conda/envs/testbed/bin/python]\n"
                        f"[Command finished with exit code {rc}]"
                    )
                elif fn == "str_replace_editor":
                    o = run_editor(env, args, cmd_timeout)
                    content = trunc(o["output"])
                else:
                    stats["unknown_tools"][fn] = stats["unknown_tools"].get(fn, 0) + 1
                    content = f"Error: tool '{fn}' is not available. Use execute_bash, str_replace_editor, or finish."
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": content})
                track_stuck(fn, tc["function"].get("arguments", ""), content)
            if pending_nudge and not finished and not stuck:
                if nudges_used < MAX_NUDGES:
                    messages.append({"role": "user", "content": pending_nudge})
                    nudges_used += 1
                    stats["nudges"] = nudges_used
                    consec_sig = 0
                    consec_err = 0
                    last_sig = None
                else:
                    stuck = True
                    stats["aborted_stuck"] = f"nudges_exhausted after {nudges_used}"
                pending_nudge = None
            if finished or stuck:
                break
        # Produce the patch: changes to TRACKED files vs the instance base commit. This excludes
        # untracked files the agent created (reproduce scripts, editor backups) — matching the
        # SWE-bench gold-patch shape and what applied+resolved cleanly before.
        # Exclude build/config files: models sometimes edit them (e.g. pinning setuptools in
        # pyproject.toml), which makes `git apply` reject the whole patch at grading time and
        # breaks `pip install -e .`. SWE-bench gold patches only touch package source.
        base = inst.get("base_commit", "")
        excl = " ".join(
            f"':(exclude){f}'"
            for f in ["pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "MANIFEST.in", "conftest.py"]
        )
        patch = env.execute({"command": f"git diff {base} --no-color -- . {excl}"}, timeout=cmd_timeout)
        p = patch["output"] or ""
        stats["finished"] = finished
        stats["patch_len"] = len(p)
        stats["empty_reason"] = (
            ""
            if p.strip()
            else (
                "aborted_stuck"
                if stats.get("aborted_stuck")
                else (
                    "finished_no_edit"
                    if finished
                    else ("never_toolcalled" if not stats["tools"] else "hit_max_iter_no_fix")
                )
            )
        )
        # Step-0.2 diagnostics: final repo state — distinguishes untracked scratch files vs
        # reverted source edits vs a clean-but-empty tree. Untracked (??) with empty patch =>
        # the model only touched excluded/scratch files; no source edits => it reverted/never edited.
        gs = env.execute({"command": "git status --porcelain | head -40"}, timeout=cmd_timeout)["output"] or ""
        stats["git_status"] = gs
        stats["untracked"] = sum(1 for ln in gs.splitlines() if ln.startswith("??"))
        stats["modified_tracked"] = sum(1 for ln in gs.splitlines() if ln[:2].strip() in ("M", "MM", "AM"))
        return iid, p, stats, messages
    finally:
        env.cleanup()


def main():
    """CLI entry point: run the OpenHands 3-tool agent over a slice of SWE-bench instances."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--slice", default="")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--cmd-timeout", type=int, default=120)
    ap.add_argument("--stats", action="store_true", help="write per-instance failure-mode stats.json (Check 3)")
    ap.add_argument("--dump-traj", action="store_true", help="dump full per-instance message trajectories (Step 0.2)")
    args = ap.parse_args()

    out = os.path.join(args.output_dir, "preds.json")
    stats_out = os.path.join(args.output_dir, "stats.json")
    os.makedirs(args.output_dir, exist_ok=True)
    preds = json.load(open(out)) if os.path.exists(out) else {}
    all_stats = json.load(open(stats_out)) if (args.stats and os.path.exists(stats_out)) else {}

    inst_list = list(load_dataset(DATASET_MAP.get(args.subset, args.subset), split=args.split))
    if args.slice:
        a, b = (int(x) if x else None for x in args.slice.split(":"))
        inst_list = inst_list[a:b]
    todo = [i for i in inst_list if i["instance_id"] not in preds]
    print(f"OH3: {len(todo)} to run ({len(preds)} already done) of {len(inst_list)}", flush=True)

    import threading

    lock = threading.Lock()

    def worker(inst):
        iid, patch, stats, messages = solve(
            inst, args.endpoint, args.model, args.max_iter, args.max_tokens, args.cmd_timeout
        )
        with lock:
            preds[iid] = {"model_name_or_path": args.model, "instance_id": iid, "model_patch": patch}
            json.dump(preds, open(out, "w"), indent=2)
            if args.stats:
                all_stats[iid] = stats
                json.dump(all_stats, open(stats_out, "w"), indent=2)
            if args.dump_traj:
                tdir = os.path.join(args.output_dir, "traj")
                os.makedirs(tdir, exist_ok=True)
                json.dump(messages, open(os.path.join(tdir, f"{iid}.json"), "w"), indent=2)
        print(
            f"  done {iid} patch_len={len(patch)} turns={stats['turns']} finished={stats['finished']} untracked={stats.get('untracked')} modified={stats.get('modified_tracked')}",
            flush=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, i): i["instance_id"] for i in todo}
        for f in concurrent.futures.as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"  ERROR {futs[f]}: {type(e).__name__}: {e}", flush=True)
    print(f"OH3 DONE: {len(preds)} preds written to {out}", flush=True)


if __name__ == "__main__":
    main()
