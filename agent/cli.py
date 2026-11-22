"""Interactive CLI entrypoint for the Plate Coach agent.

Run with: python -m agent.cli     (or agent.exe)
"""

import json
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import openai
from langgraph.types import Command

from agent import llm
from agent.graph import (
    build_analysis_graph,
    build_export_graph,
    build_history_graph,
    build_question_graph,
    build_simulation_graph,
)
from agent.pricing import cost_breakdown
from agent.tools import HISTORY_PATH

VERSION = "1.0.0"
ERROR_LOG = Path("plate_coach_error.log")

GOALS = {
    "1": "weight_loss",
    "2": "maintain",
    "3": "balanced",
    "4": "muscle_gain",
}

HELP_TEXT = """
Plate Coach -- meal nutrition analysis agent (LangGraph + OpenRouter).

What it does: give it a photo of a meal and it identifies the dish, looks up its
nutrition, computes a 0-100 Healthy Score, and suggests small improvements --
each one re-scored by code, never guessed by the LLM. It answers nutrition
questions about foods and "what if" questions about your meal. Anything else is
politely refused.

Commands:
  <image path>          Analyze a meal photo, e.g.  C:\\photos\\lunch.jpg
                        (you can drag a file into the terminal; quotes are fine)
  <question>            Ask a nutrition question, e.g.  how much protein is in sushi?
  history               Weekly trend over up to 20 photos (or your saved meals)
  export-summary        Save a summary of this conversation to a text file
  model [name]          Show or switch the OpenRouter model, e.g.  model openai/gpt-4o-mini
  status                Show the model, API-key and LangSmith tracing status
  cost                  Show this session's total tokens and cost
  help, ?               Show this message
  quit, exit            Exit Plate Coach

After a photo analysis you can ask what-if questions ("what if I add a side
salad?"); press Enter on an empty line to go back to the main prompt.
Before anything is written to disk you are asked to confirm [Y/N].
"""


class Session:
    """Per-process CLI state: goal, model, last analysis, cost and transcript."""

    def __init__(self):
        self.goal = "balanced"
        self.model = llm.DEFAULT_MODEL
        self.last_state: dict = {}
        self.usage_log: list = []
        self.transcript: list = []

    def log(self, who: str, text: str):
        self.transcript.append((datetime.now().strftime("%H:%M:%S"), who, text))


# ---------------------------------------------------------------- printing


def info(msg: str):
    print(f"[INFO] {msg}", flush=True)


def error(msg: str):
    print(f"[ERROR] {msg}", flush=True)


def print_help():
    print(HELP_TEXT)


def print_status(session: Session):
    key = "set" if llm.OPENROUTER_API_KEY else "MISSING (add OPENROUTER_API_KEY to .env)"
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))
    project = os.getenv("LANGSMITH_PROJECT", "default")
    print(f"  Model:            {session.model}")
    print(f"  OpenRouter key:   {key}")
    print(f"  LangSmith traces: {'ON, project ' + project if tracing else 'off (set LANGSMITH_API_KEY + LANGSMITH_TRACING=true)'}")
    print(f"  Session goal:     {session.goal.replace('_', ' ')}")
    print("  Extensions:       RAG nutrition database, Code-Exec scoring, HITL file writes")


def print_cost(usage_log: list, title: str = "Cost of this task"):
    if not usage_log:
        return
    report = cost_breakdown(usage_log)
    print(f"\n[COST] {title}:")
    for step in report["steps"]:
        print(
            f"  {step['node']:<20} {step['prompt_tokens']:>6} in /{step['completion_tokens']:>5} out"
            f"   ${step['actual_cost_usd']:.6f}  (theoretical ${step['theoretical_cost_usd']:.6f})"
        )
    t = report["total"]
    print(
        f"  {'TOTAL':<20} {t['prompt_tokens']:>6} in /{t['completion_tokens']:>5} out"
        f"   ${t['actual_cost_usd']:.6f}  (theoretical ${t['theoretical_cost_usd']:.6f})"
        f"   [{t['llm_calls']} LLM calls]"
    )


def print_analysis(result: dict):
    final_output = result.get("final_output")
    if not final_output:
        if result.get("gate_reason"):
            print(f"\n[GATE] {result['gate_reason']}")
        return

    score = final_output.get("healthy_score") or {}
    conf = final_output.get("confidence")
    conf_str = f" ({conf:.0%} confident)" if isinstance(conf, (int, float)) else ""
    print(f"\nIdentified: {str(final_output.get('food', '?')).replace('_', ' ')}{conf_str}")
    print(f"Healthy Score: {score.get('score')}/100")
    print(f"Breakdown: {score.get('breakdown')}")
    print(f"Profile: {final_output.get('qualitative_profile')}")
    if final_output.get("unmatched_foods"):
        print(f"Note: {', '.join(final_output['unmatched_foods'])} is not in the nutrition database; typical values were used.")
    print("\nSuggestions:")
    for s in final_output.get("suggestions", []):
        projected = s.get("projected_score")
        projected_str = f" (projected: {projected}/100)" if projected is not None else ""
        print(f"  - {s['text']}{projected_str}")
    print(f"\n{final_output.get('disclaimer', '')}")


# ------------------------------------------------------------- graph running


def stream_graph(graph, graph_input, config=None, resuming: str | None = None) -> None:
    """Run a graph, printing [INFO] Entering Node: <name>... as each node starts.

    Uses LangGraph's stream_mode="tasks", which emits an event when a node's task
    starts (with "input") and when it finishes (with "result"/"error").
    """
    for event in graph.stream(graph_input, config, stream_mode="tasks"):
        if "input" in event and "result" not in event:
            if resuming:
                info(f"Resuming Node: {event['name']} ({resuming})...")
                resuming = None
            else:
                info(f"Entering Node: {event['name']}...")
        elif event.get("error"):
            raise RuntimeError(f"node {event['name']} failed: {event['error']}")


def run_with_hitl(graph, graph_input, config) -> dict:
    """Stream a checkpointed graph; on each interrupt() ask [Y/N] and resume."""
    stream_graph(graph, graph_input, config)
    while True:
        snapshot = graph.get_state(config)
        interrupts = [i for task in snapshot.tasks for i in task.interrupts]
        if not interrupts:
            return snapshot.values
        payload = interrupts[0].value.get("payload", {})
        print(f"\n[HITL] The agent wants to {payload.get('description', 'write a file')}:")
        print(f"       {Path(payload.get('target', '?')).resolve()}")
        approved = ask_yes_no("Proceed? [Y/N]: ")
        stream_graph(graph, Command(resume=approved), config, resuming="approved" if approved else "declined")


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "yes", "כן"):
            return True
        if answer in ("n", "no", "לא", ""):
            return False
        print("Please answer Y or N.")


def new_config() -> dict:
    # A fresh thread per run: the checkpointer would otherwise carry the last
    # run's state (retry count, usage log) into the next analysis.
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


# ------------------------------------------------------------------ commands


def clean_path(text: str) -> str:
    return text.strip().strip('"').strip("'").strip()


def is_image_path(text: str) -> bool:
    return Path(clean_path(text)).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def prompt_goal() -> str:
    print("\nWhat's your goal for this session?")
    print("  1) Weight loss\n  2) Maintain weight\n  3) Balanced nutrition\n  4) Muscle gain")
    while True:
        choice = input("Choose 1-4 (Enter = balanced): ").strip()
        if not choice:
            return "balanced"
        if choice in GOALS:
            return GOALS[choice]
        print("Please enter 1, 2, 3, or 4.")


def run_analysis(session: Session, graphs: dict, raw_path: str):
    path = clean_path(raw_path)
    if not Path(path).is_file():
        error(f"Could not find an image at: {path}")
        return
    session.log("user", f"[photo] {path}")
    state = {"image_path": path, "goal": session.goal, "model": session.model, "save_history": True}
    try:
        result = run_with_hitl(graphs["analysis"], state, new_config())
    except ValueError as exc:  # unreadable image
        error(f"That file isn't a readable image (.jpg/.png/etc): {exc}")
        return
    session.usage_log += result.get("usage_log", [])

    print_analysis(result)
    if result.get("write_result"):
        info(result["write_result"])
    if result.get("error"):
        error(result["error"])
    print_cost(result.get("usage_log", []))

    out = result.get("final_output")
    if out:
        session.last_state = result
        session.log("agent", f"{out.get('food')}: Healthy Score {(out.get('healthy_score') or {}).get('score')}/100; "
                    + " | ".join(s["text"] for s in out.get("suggestions", [])))
        whatif_loop(session, graphs)
    elif result.get("gate_reason"):
        session.log("agent", result["gate_reason"])


def whatif_loop(session: Session, graphs: dict):
    print("\nAsk a what-if question about this meal, or press Enter to go back.")
    while True:
        try:
            query = input("what-if> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            return
        if query.lower() in ("quit", "exit"):
            raise SystemExit(0)
        if query.lower() in ("help", "?"):
            print_help()
            continue
        guarded(session, lambda: run_whatif(session, graphs, query))


def run_whatif(session: Session, graphs: dict, query: str):
    session.log("user", f"[what-if] {query}")
    last = session.last_state
    state = {
        "user_query": query,
        "model": session.model,
        "nutrition_lookup": last.get("nutrition_lookup", {}),
        "vision_result": last.get("vision_result", {}),
    }
    stream_events_state = {}
    for event in graphs["simulation"].stream(state, stream_mode="tasks"):
        if "input" in event and "result" not in event:
            info(f"Entering Node: {event['name']}...")
        elif event.get("result"):
            stream_events_state.update(event["result"])
    result = stream_events_state
    session.usage_log += result.get("usage_log", [])

    if result.get("final_output", {}).get("refused"):
        print(f"\n{result['final_output']['message']}")
        session.log("agent", result["final_output"]["message"])
    elif result.get("error"):
        error(result["error"])
    else:
        d = result["simulation_result"]
        print(f"  Before: score {d['before']['score']}, {d['before']['calories']} kcal")
        print(f"  After:  score {d['after']['score']}, {d['after']['calories']} kcal")
        print(f"  Change: score {d['delta']['score']:+}, calories {d['delta']['calories']:+}, "
              f"protein {d['delta']['protein_g']:+} g, sugar {d['delta']['sugar_g']:+} g")
        if d.get("used_default"):
            print("  Note: the added food is not in the nutrition database; typical values were used.")
        session.log("agent", f"what-if: score {d['before']['score']} -> {d['after']['score']}")
    print_cost(result.get("usage_log", []))


def run_question(session: Session, graphs: dict, question: str):
    session.log("user", question)
    state = {"user_query": question, "model": session.model,
             "nutrition_lookup": session.last_state.get("nutrition_lookup", {})}
    result = {}
    for event in graphs["question"].stream(state, stream_mode="tasks"):
        if "input" in event and "result" not in event:
            info(f"Entering Node: {event['name']}...")
        elif event.get("result"):
            result.update(event["result"])
    session.usage_log += result.get("usage_log", [])

    out = result.get("final_output") or {}
    print(f"\n{out.get('message', '')}")
    if not out.get("refused") and out.get("disclaimer"):
        if out.get("sources"):
            print(f"\n(Sources: Plate Coach nutrition database -- {', '.join(out['sources'])})")
        print(out["disclaimer"])
    session.log("agent", out.get("message", ""))
    print_cost(result.get("usage_log", []))


def run_history(session: Session, graphs: dict):
    print("Enter up to 20 image paths, one per line. Blank line to finish.")
    if HISTORY_PATH.exists():
        print(f"(Or press Enter right away to use your {HISTORY_PATH.name} saved meals.)")
    photos = []
    while len(photos) < 20:
        path = clean_path(input(f"photo {len(photos) + 1}: "))
        if not path:
            break
        photos.append(path)
    if not photos and HISTORY_PATH.exists():
        try:
            saved = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            photos = [m["image_path"] for m in saved if m.get("image_path")][-20:]
        except (OSError, json.JSONDecodeError, TypeError):
            error(f"{HISTORY_PATH} is not readable; give the photo paths instead.")
            return
    if not photos:
        print("No photos provided.")
        return

    session.log("user", f"[history] {len(photos)} photos")
    result = {}
    for event in graphs["history"].stream({"history_photos": photos, "model": session.model}, stream_mode="tasks"):
        if "input" in event and "result" not in event:
            info(f"Entering Node: {event['name']}...")
        elif event.get("result"):
            result.update(event["result"])
    session.usage_log += result.get("usage_log", [])

    for skipped in result.get("history_skipped", []):
        error(f"Skipped {skipped}")
    for m in result.get("history_meals", []):
        print(f"  {m['food']:<24} score {m['score']:>3}")
    print(f"\n{result.get('history_summary')}")
    session.log("agent", str(result.get("history_summary")))
    print_cost(result.get("usage_log", []))


def build_summary_text(session: Session) -> str:
    report = cost_breakdown(session.usage_log)["total"]
    lines = [
        "Plate Coach -- conversation summary",
        f"Date: {datetime.now():%Y-%m-%d %H:%M}",
        f"Goal: {session.goal.replace('_', ' ')}   Model: {session.model}",
        "",
    ]
    for ts, who, text in session.transcript:
        lines.append(f"[{ts}] {'You' if who == 'user' else 'Plate Coach'}: {text}")
    lines += [
        "",
        f"Usage: {report['llm_calls']} LLM calls, {report['prompt_tokens']} prompt + "
        f"{report['completion_tokens']} completion tokens, ${report['actual_cost_usd']:.6f} "
        f"(theoretical ${report['theoretical_cost_usd']:.6f})",
        "",
        "General lifestyle information only, not professional medical or dietary advice.",
    ]
    return "\n".join(lines) + "\n"


def run_export(session: Session, graphs: dict):
    if not session.transcript:
        print("Nothing to export yet -- analyze a photo or ask a question first.")
        return
    target = Path(f"plate_coach_summary_{datetime.now():%Y%m%d_%H%M%S}.txt")
    pending = {
        "target": str(target),
        "mode": "text",
        "content": build_summary_text(session),
        "description": "export the conversation summary to a text file",
    }
    result = run_with_hitl(graphs["export"], {"pending_write": pending}, new_config())
    if result.get("error"):
        error(result["error"])
    else:
        info(result.get("write_result") or "Not saved.")


# ------------------------------------------------------------ error handling


def explain_api_error(exc: Exception) -> str:
    """Turn OpenRouter/OpenAI client exceptions into a message a user can act on."""
    if isinstance(exc, openai.AuthenticationError):
        return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY in your .env file."
    if isinstance(exc, openai.NotFoundError):
        return (f"OpenRouter does not know the model '{llm_model_hint()}'. Pick a current one at "
                "https://openrouter.ai/models and switch with:  model <name>")
    if isinstance(exc, openai.RateLimitError):
        return ("Rate limit reached. Free models allow only a limited number of requests per day/minute; "
                "wait a minute, or switch to a paid model with:  model openai/gpt-4o-mini")
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 402:
        return "Your OpenRouter account has no credits left for this paid model. Add credits or switch to a ':free' model."
    if isinstance(exc, openai.APITimeoutError):
        return "OpenRouter did not answer in time. Check your internet connection and try again."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not reach OpenRouter. Check your internet connection (see RUNNING_AGENT.md for proxy/SSL issues)."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenRouter returned an error ({exc.status_code}). Try again, or switch model."
    return ""


_current_session: Session | None = None


def llm_model_hint() -> str:
    return _current_session.model if _current_session else llm.DEFAULT_MODEL


def guarded(session: Session, action):
    """Run one command; turn any failure into a clear message instead of a traceback."""
    try:
        action()
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Cancelled.")
    except SystemExit:
        raise
    except openai.OpenAIError as exc:
        error(explain_api_error(exc) or f"LLM request failed: {exc}")
    except FileNotFoundError as exc:
        error(str(exc))
    except Exception as exc:  # last resort: still no raw traceback on screen
        try:
            with ERROR_LOG.open("a", encoding="utf-8") as f:
                f.write(f"\n--- {datetime.now()} ---\n{traceback.format_exc()}")
            where = f" Details were written to {ERROR_LOG.resolve()}."
        except OSError:
            where = ""
        error(f"Something went wrong: {type(exc).__name__}: {exc}.{where}")


# ---------------------------------------------------------------------- main


def main():
    global _current_session
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Welcome to Plate Coach v{VERSION}.")
    print_help()
    session = Session()
    _current_session = session
    if not llm.OPENROUTER_API_KEY:
        error("OPENROUTER_API_KEY is not set. Copy .env.example to .env (next to the program) and add your key.")
    try:
        session.goal = prompt_goal()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        return 0

    graphs = {
        "analysis": build_analysis_graph(),
        "question": build_question_graph(),
        "simulation": build_simulation_graph(),
        "history": build_history_graph(),
        "export": build_export_graph(),
    }

    while True:
        try:
            cmd = input("\nplate-coach> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not cmd:
            continue
        lower = cmd.lower()

        try:
            if lower in ("help", "?"):
                print_help()
            elif lower in ("quit", "exit"):
                print("Goodbye.")
                break
            elif lower == "status":
                print_status(session)
            elif lower == "cost":
                print_cost(session.usage_log, "Session total")
                if not session.usage_log:
                    print("No LLM calls yet this session.")
            elif lower == "model" or lower.startswith("model "):
                name = cmd[5:].strip()
                if name:
                    session.model = name
                    info(f"Model switched to {name}")
                else:
                    print(f"Current model: {session.model}")
            elif lower == "history":
                guarded(session, lambda: run_history(session, graphs))
            elif lower in ("export-summary", "export"):
                guarded(session, lambda: run_export(session, graphs))
            elif is_image_path(cmd):
                guarded(session, lambda: run_analysis(session, graphs, cmd))
            else:
                guarded(session, lambda: run_question(session, graphs, cmd))
        except SystemExit:
            print("Goodbye.")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
