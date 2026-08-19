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

"""In-distribution tool-call probe for a base-SFT gemma4 checkpoint.

Answers the pre-retrain fork: does the checkpoint emit the tool-call special token `<|tool_call>`
(id 48) when the prompt is rendered EXACTLY as the CoderForge SFT training data was, vs when it is
served through vLLM's chat/tool path? Two probes per sample:

  A) IN-DISTRIBUTION  — render messages[:first_assistant_toolcall] with the model's own chat
     template + tools + add_generation_prompt, POST to /v1/completions (raw, no server-side
     re-templating). Inspect whether the completion begins with `<|tool_call>call:`.
  B) SERVER CHAT PATH — POST the same messages+tools to /v1/chat/completions with
     tool_choice=auto. Inspect whether `tool_calls` is populated (structured) vs the call landing
     in `content` as text.

Verdict:
  - A emits <|tool_call> but B empty  -> prompt-format distribution shift (eval rendering != train).
  - neither emits it                  -> pure capability gap (more SFT tokens is the right fix).
  - both emit it                      -> checkpoint is fine; the oh3 0% was something else.
"""

import argparse
import json
import urllib.request

from transformers import AutoTokenizer

TOOLCALL_OPEN = "<|tool_call>"
TOOLCALL_OPEN_ID = 48  # <|tool_call> special token id (Gemma4)
# The Gemma4 assistant turn opens with these three tokens; a training-faithful decision
# prompt must end on them, i.e. the next token the model is trained to emit is <|tool_call>.
DECISION_TAIL = [105, 4368, 107]  # <|turn> , 'model' , '\n'


def _post(url, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def first_toolcall_prefix(messages):
    """Return messages up to (excluding) the first assistant turn that has tool_calls."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            return messages[:i]
    return None


def logprob_probe(prompt, prompt_ids, tok, endpoint, model, topk):
    """Path C: at the exact training decision point, what is P(<|tool_call>) for the FIRST token?

    Renders the prompt to its final tokens (must end on ``<|turn>model\\n`` to be
    training-faithful), then requests a single-token completion with top-``topk`` logprobs
    and reports the rank + logprob of ``<|tool_call>`` (id 48) vs the argmax token. Unlike
    the binary emit/no-emit checks, this shows whether emission is *forming* (48 climbing the
    candidate list across checkpoints) even when it is not yet the greedy pick.
    """
    tail = list(prompt_ids[-len(DECISION_TAIL) :])
    tail_ok = tail == DECISION_TAIL
    try:
        c = _post(
            endpoint + "/completions",
            {
                "model": model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": topk,
            },
        )
    except Exception as e:
        return {"available": False, "tail_ok": tail_ok, "tail": tail, "err": f"{type(e).__name__}: {e}"}
    lp = c["choices"][0].get("logprobs") or {}
    tops = (lp.get("top_logprobs") or [{}])[0] or {}
    argmax_tok = (lp.get("tokens") or [None])[0]
    argmax_lp = (lp.get("token_logprobs") or [None])[0]
    # rank <|tool_call> among the returned candidates (by logprob desc); None if outside top-k.
    ranked = sorted(tops.items(), key=lambda kv: kv[1], reverse=True)
    tc_rank, tc_lp = None, None
    for i, (t, v) in enumerate(ranked):
        if t == TOOLCALL_OPEN:
            tc_rank, tc_lp = i, v
            break
    return {
        "available": True,
        "tail_ok": tail_ok,
        "tail": tail,
        "argmax_tok": argmax_tok,
        "argmax_lp": argmax_lp,
        "tc_rank": tc_rank,
        "tc_lp": tc_lp,
        "topk": topk,
        "top5": ranked[:5],
    }


def probe_sample(tag, messages, tools, tok, endpoint, model, max_tokens, topk):
    """Probe one rendered sample: does the served checkpoint emit a structured tool call?"""
    prefix = first_toolcall_prefix(messages)
    if prefix is None:
        return None
    # --- A) in-distribution: exact training rendering, raw completion (best-effort; some vLLM
    # serves don't expose /completions for multimodal models -> fall back to B only). ---
    prompt = tok.apply_chat_template(prefix, tools=tools, tokenize=False, add_generation_prompt=True)
    a_text = None
    try:
        ca = _post(
            endpoint + "/completions",
            {
                "model": model,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": max_tokens,
                "stop": ["<tool_call|>"],
            },
        )
        a_text = ca["choices"][0]["text"]
    except Exception as e:
        print(f"  [A path /completions unavailable: {type(e).__name__}: {e}]")
    a_structured = bool(a_text) and TOOLCALL_OPEN in a_text
    a_text_call = bool(a_text) and "call:" in a_text
    a_emits = a_structured or a_text_call
    a_where = -1
    if a_text:
        tc_pos = a_text.find(TOOLCALL_OPEN)
        a_where = tc_pos if tc_pos >= 0 else a_text.find("call:")

    # --- B) server chat path with tool_choice=auto ---
    cb = _post(
        endpoint + "/chat/completions",
        {
            "model": model,
            "messages": prefix,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": max_tokens,
            "stop_token_ids": [1, 106, 50],
        },
    )
    bmsg = cb["choices"][0]["message"]
    b_tcs = bmsg.get("tool_calls") or []
    b_content = bmsg.get("content") or ""

    print(f"\n===== [{tag}] =====")
    print("  A in-distribution (/completions, training render):")
    if a_text is None:
        print("     (unavailable)")
    else:
        print(
            f"     emits <|tool_call> (structured): {a_structured} | text 'call:': {a_text_call} | "
            f"first-call char-offset: {a_where} | completion chars: {len(a_text)}"
        )
        print(f"     completion[:240]: {a_text[:240]!r}")
    print("  B server chat (/chat/completions, tool_choice=auto):")
    print(f"     structured tool_calls: {len(b_tcs)} | content[:160]: {b_content[:160]!r}")
    if b_tcs:
        print(f"     first tool_call: {b_tcs[0]['function']['name']}({b_tcs[0]['function']['arguments'][:120]})")

    # --- C) logprob of <|tool_call> at the decision point (first generated token) ---
    # return_dict=True so we get a plain List[int] of input_ids (tokenize=True alone can return
    # a fast-tokenizer Encoding on some tokenizers, which breaks the decision-tail check).
    prompt_ids = tok.apply_chat_template(
        prefix, tools=tools, tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    c = logprob_probe(prompt, prompt_ids, tok, endpoint, model, topk)
    print("  C decision-point logprob (first token after '<|turn>model\\n'):")
    if not c["available"]:
        print(f"     (unavailable: {c.get('err')})  tail_ok={c['tail_ok']} tail={c['tail']}")
    else:
        if not c["tail_ok"]:
            print(
                f"     WARNING: prompt tail {c['tail']} != training decision tail {DECISION_TAIL} "
                f"-> eval rendering differs from training (path-b/rendering issue)"
            )
        rank_str = "OUTSIDE top-%d" % c["topk"] if c["tc_rank"] is None else f"rank {c['tc_rank']}"
        print(
            f"     argmax={c['argmax_tok']!r} (lp={c['argmax_lp']}) | "
            f"<|tool_call> {rank_str}" + (f" (lp={c['tc_lp']:.3f})" if c["tc_lp"] is not None else "")
        )
        print(f"     top5: {[(t, round(v, 3)) for t, v in c['top5']]}")

    return {
        "tag": tag,
        "A_available": a_text is not None,
        "A_emits_toolcall_token": a_structured,
        "A_text_call": a_text_call,
        "A_starts_call": a_emits,
        "B_structured_count": len(b_tcs),
        "C_available": c["available"],
        "C_tail_ok": c["tail_ok"],
        "C_tc_rank": c.get("tc_rank"),
        "C_tc_lp": c.get("tc_lp"),
        "C_argmax": c.get("argmax_tok"),
        "C_argmax_lp": c.get("argmax_lp"),
    }


def main():
    """CLI entry point: probe a served checkpoint for tool-call emergence on CoderForge samples."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True, help="CoderForge training JSONL (in-distribution samples)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--topk", type=int, default=20, help="top-k logprobs for the decision-point probe (path C)")
    ap.add_argument("--swe-subset", default="lite")
    ap.add_argument("--swe-split", default="test")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # (1) In-distribution CoderForge training samples.
    results = []
    n = 0
    with open(args.data) as f:
        for line in f:
            if n >= args.n:
                break
            r = json.loads(line)
            msgs = r["messages"] if isinstance(r["messages"], list) else json.loads(r["messages"])
            tools = r.get("tools")
            tools = tools if isinstance(tools, (list, type(None))) else json.loads(tools)
            if first_toolcall_prefix(msgs) is None:
                continue
            res = probe_sample(
                f"coderforge#{n}", msgs, tools, tok, args.endpoint, args.model, args.max_tokens, args.topk
            )
            if res:
                results.append(res)
                n += 1

    a_avail = sum(1 for r in results if r["A_available"])
    a_hits = sum(1 for r in results if r["A_emits_toolcall_token"])
    a_textcall = sum(1 for r in results if r["A_text_call"])
    b_hits = sum(1 for r in results if r["B_structured_count"] > 0)
    print("\n================ VERDICT ================")
    print(f"in-distribution samples probed: {len(results)} (A path available on {a_avail})")
    print(
        f"  A) emits <|tool_call> under TRAINING rendering: {a_hits}/{a_avail}  "
        f"(plain-text 'call:' seen on {a_textcall}/{a_avail})"
    )
    print(f"  B) structured tool_calls under SERVER chat path: {b_hits}/{len(results)}")

    # C) decision-point logprob summary: is <|tool_call> the argmax, and if not, how close?
    c_avail = [r for r in results if r.get("C_available")]
    c_tail_ok = sum(1 for r in c_avail if r.get("C_tail_ok"))
    c_argmax_is_tc = sum(1 for r in c_avail if r.get("C_argmax") == TOOLCALL_OPEN)
    ranks = [r["C_tc_rank"] for r in c_avail if r.get("C_tc_rank") is not None]
    lps = [r["C_tc_lp"] for r in c_avail if r.get("C_tc_lp") is not None]
    print(
        f"  C) decision-point <|tool_call> logprob (n={len(c_avail)}, training-tail-matched {c_tail_ok}/{len(c_avail)}):"
    )
    print(
        f"       argmax==<|tool_call> on {c_argmax_is_tc}/{len(c_avail)} | "
        f"in-top-{args.topk} on {len(ranks)}/{len(c_avail)}"
        + (f" | ranks={ranks} | logprobs={[round(x, 3) for x in lps]}" if ranks else " | never in top-k")
    )
    print("  (track C across step_99/199/399/599: rank climbing / logprob rising => emission is installing.)")
    # switched-on = the model produces STRUCTURED tool calls the parser accepts (A token 48 or B tool_calls)
    if b_hits or a_hits:
        print("  => TOOL-CALLING SWITCHED ON: structured tool calls present. Retrain is working.")
    elif a_textcall:
        print("  => LEARNING BUT NOT THERE: emits the plain-text 'call:' approximation (not token 48 yet)")
        print("     -> the format is forming; keep training, expect the structured switch-on soon.")
    else:
        print("  => NOT YET: no tool call (structured or text) at this step. Watch the kill-check (~300-400).")


if __name__ == "__main__":
    main()
