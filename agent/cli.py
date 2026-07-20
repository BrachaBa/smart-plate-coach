"""Interactive CLI entrypoint for the Plate Coach agent.

Run with: python -m agent.cli
"""

import json
import sys
import uuid
from pathlib import Path

from langgraph.types import Command
from PIL import UnidentifiedImageError

from agent.graph import build_analysis_graph, build_history_graph, build_simulation_graph
from agent.llm import DEFAULT_MODEL
from agent.nodes import scope_guard_node
from agent.pricing import estimate_cost

GOALS = {
    "1": "weight_loss",
    "2": "maintain",
    "3": "balanced",
    "4": "muscle_gain",
}

HELP_TEXT = """
Plate Coach -- meal nutrition analysis agent.

Commands:
  <image path>       Analyze a meal photo (e.g. data/food41/images/sushi/1.jpg)
  <what-if question> After an analysis, ask free-form what-if questions
                      (e.g. "what if I remove the dessert?")
  history             Run the weekly trend analyzer over up to 20 saved photos
  export-summary       Export the last analysis to a local text file
  help, ?              Show this message
  quit, exit            Exit Plate Coach

Plate Coach only handles meal-photo nutrition analysis, healthy scoring, and
what-if simulations -- it will decline anything outside that scope.
"""


def print_help():
    print(HELP_TEXT)


def prompt_goal() -> str:
    print("\nWhat's your goal for this session?")
    print("  1) Weight loss\n  2) Maintain weight\n  3) Balanced nutrition\n  4) Muscle gain")
    while True:
        choice = input("Choose 1-4: ").strip()
        if choice in GOALS:
            return GOALS[choice]
        print("Please enter 1, 2, 3, or 4.")


def is_image_path(text: str) -> bool:
    suffix = Path(text).suffix.lower()
    return suffix in (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def print_token_cost(token_usage: dict):
    if not token_usage:
        return
    cost = estimate_cost(token_usage, model=DEFAULT_MODEL)
    print(
        f"[cost] {token_usage.get('prompt_tokens', 0)} prompt + "
        f"{token_usage.get('completion_tokens', 0)} completion tokens "
        f"(~${cost['actual_cost_usd']:.4f}, theoretical ${cost['theoretical_cost_usd']:.4f})"
    )


def stream_analysis(graph, state, config) -> dict:
    """Run the analysis graph, printing [INFO] Entering Node: ... as each node
    completes (LangGraph's stream_mode="updates" -- the idiomatic per-node
    streaming mode). Returns the final state once the graph pauses or ends."""
    final_state = {}
    for update in graph.stream(state, config, stream_mode="updates"):
        for node_name, node_output in update.items():
            if node_name == "__interrupt__":
                continue
            print(f"[INFO] Entering Node: {node_name}...")
            final_state.update(node_output or {})
    return final_state


def handle_hitl_interrupt(graph, config) -> bool:
    """If the graph paused on an interrupt, prompt [Y/N] and resume. Returns
    True if an interrupt was handled (resumed), False if there was none."""
    snapshot = graph.get_state(config)
    if not snapshot.tasks or not snapshot.tasks[0].interrupts:
        return False

    payload = snapshot.tasks[0].interrupts[0].value
    print(f"\n[HITL] About to write: {payload['payload'].get('target', 'a local file')}")
    answer = input("Proceed? [Y/N]: ").strip().lower()
    approved = answer in ("y", "yes")

    for update in graph.stream(Command(resume=approved), config, stream_mode="updates"):
        for node_name in update:
            if node_name != "__interrupt__":
                print(f"[INFO] Entering Node: {node_name}...")
    return True


def print_final_output(result: dict):
    final_output = result.get("final_output")
    if not final_output:
        if result.get("gate_reason"):
            print(f"\n[!] {result['gate_reason']}")
        return

    if final_output.get("refused"):
        print(f"\n{final_output['message']}")
        return

    score = final_output.get("healthy_score", {})
    print(f"\nHealthy Score: {score.get('score')}/100")
    print(f"Breakdown: {score.get('breakdown')}")
    print(f"Profile: {final_output.get('qualitative_profile')}")
    print("\nSuggestions:")
    for s in final_output.get("suggestions", []):
        projected = s.get("projected_score")
        projected_str = f" (projected: {projected}/100)" if projected is not None else ""
        print(f"  - {s['text']}{projected_str}")
    print(f"\n{final_output.get('disclaimer', '')}")


def run_analysis_flow(graph, config, image_path: str, goal: str) -> dict:
    try:
        result = stream_analysis(graph, {"image_path": image_path, "goal": goal}, config)
    except FileNotFoundError:
        print(f"[!] Could not find an image at: {image_path}")
        return {}
    except (UnidentifiedImageError, ValueError):
        print(f"[!] That file isn't a readable image (.jpg/.png/etc): {image_path}")
        return {}

    if handle_hitl_interrupt(graph, config):
        result = graph.get_state(config).values

    print_final_output(result)
    print_token_cost(result.get("token_usage"))
    return result


def enter_whatif_loop(sim_graph, last_state: dict) -> bool:
    """Returns False if the user asked to quit/exit the whole session (so main()
    can stop), True otherwise (including a blank line, which just ends this loop)."""
    print("\nAsk a what-if question, or press Enter to move on ('quit'/'exit' to leave Plate Coach).")
    while True:
        try:
            query = input("what-if> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return False

        if not query:
            return True
        if query.lower() in ("quit", "exit"):
            print("Goodbye.")
            return False

        scope_result = scope_guard_node({"user_query": query})
        if not scope_result.get("in_scope"):
            print(f"\n{scope_result['refusal_message']}")
            print_token_cost(scope_result.get("token_usage"))
            continue

        sim_state = {
            "nutrition_lookup": last_state.get("nutrition_lookup", {}),
            "vision_result": last_state.get("vision_result", {}),
            "simulation_query": query,
        }
        result = sim_graph.invoke(sim_state)
        if result.get("error"):
            print(f"[!] {result['error']}")
            continue
        delta = result.get("simulation_result", {})
        print(f"  Before: score {delta['before']['score']}, {delta['before']['calories']} kcal")
        print(f"  After:  score {delta['after']['score']}, {delta['after']['calories']} kcal")
        print(f"  Delta:  {delta['delta']}")
        print_token_cost(result.get("token_usage"))


def run_history_flow(hist_graph):
    print("Enter up to 20 image paths, one per line. Blank line to finish.")
    photos = []
    while len(photos) < 20:
        path = input(f"photo {len(photos) + 1} (or blank to finish): ").strip()
        if not path:
            break
        photos.append(path)
    if not photos:
        print("No photos provided.")
        return
    result = hist_graph.invoke({"history_photos": photos})
    print(f"\n{result.get('history_summary')}")
    print_token_cost(result.get("token_usage"))


def export_summary(last_state: dict):
    if not last_state or not last_state.get("final_output"):
        print("No analysis to export yet.")
        return
    out_path = Path("plate_coach_summary.txt")
    out_path.write_text(json.dumps(last_state["final_output"], indent=2))
    print(f"Summary exported to {out_path.resolve()}")


def main():
    print("Welcome to Plate Coach.")
    print_help()
    goal = prompt_goal()

    analysis_graph = build_analysis_graph()
    sim_graph = build_simulation_graph()
    hist_graph = build_history_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    last_state = {}
    while True:
        try:
            cmd = input("\nplate-coach> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not cmd:
            continue
        if cmd.lower() in ("help", "?"):
            print_help()
        elif cmd.lower() in ("quit", "exit"):
            print("Goodbye.")
            break
        elif cmd.lower() == "history":
            run_history_flow(hist_graph)
        elif cmd.lower() == "export-summary":
            export_summary(last_state)
        elif is_image_path(cmd):
            last_state = run_analysis_flow(analysis_graph, config, cmd, goal)
            if last_state.get("final_output") and not last_state["final_output"].get("refused"):
                if not enter_whatif_loop(sim_graph, last_state):
                    break
        else:
            # Free text that isn't a known command or image path -- route through
            # the scope guard so out-of-domain questions get a proper in-character
            # refusal instead of a generic "unrecognized command" message.
            scope_result = scope_guard_node({"user_query": cmd})
            if scope_result.get("in_scope"):
                print("[!] Not a recognized command or image path. Type 'help' for usage.")
            else:
                print(f"\n{scope_result['refusal_message']}")
            print_token_cost(scope_result.get("token_usage"))


if __name__ == "__main__":
    sys.exit(main())
