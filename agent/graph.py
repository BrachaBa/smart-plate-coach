"""LangGraph workflow definitions for the Plate Coach agent.

Five compiled graphs share the node module (agent.nodes):

- build_analysis_graph(): the main photo flow
      vision_intake -> input_gate -(gate)-> nutrition_analysis -> scoring
      -> improvement_planner -> goal_adjuster -> evaluator -(retry once)-> improvement_planner
                                                          -(ok)-> propose_save -> hitl_gate
  input_gate stops at END on a low-confidence photo; hitl_gate interrupts for [Y/N]
  before saving the meal to the local history file.
- build_question_graph(): scope_guard -(in scope)-> nutrition_qa   (RAG Q&A)
- build_simulation_graph(): scope_guard -(in scope)-> simulation_handler   (what-if)
- build_history_graph(): history_analyzer   (trend over up to 20 photos)
- build_export_graph(): hitl_gate   (export the conversation summary, after [Y/N])

The analysis and export graphs are compiled with a checkpointer, which
interrupt() needs to pause and resume across the CLI's [Y/N] prompt.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    evaluator_node,
    goal_adjuster_node,
    history_analyzer_node,
    hitl_gate_node,
    improvement_planner_node,
    input_gate_node,
    nutrition_analysis_node,
    nutrition_qa_node,
    scope_guard_node,
    scoring_node,
    simulation_handler_node,
    vision_intake_node,
)
from agent.state import PlateCoachState
from agent.tools import HISTORY_PATH

MAX_EVALUATOR_RETRIES = 1


def propose_save_node(state: PlateCoachState) -> dict:
    """Stage the "save this meal to history" write for the HITL gate.

    Only stages it -- the write itself happens in hitl_gate, after a yes.
    """
    if not state.get("save_history"):
        return {}
    out = state.get("final_output") or {}
    entry = {
        "image_path": state.get("image_path"),
        "food": out.get("food"),
        "healthy_score": (out.get("healthy_score") or {}).get("score"),
        "goal": state.get("goal"),
    }
    return {
        "pending_write": {
            "target": str(HISTORY_PATH),
            "mode": "append_json",
            "content": entry,
            "description": f"save this meal ({entry['food']}, score {entry['healthy_score']}) to your meal history",
        }
    }


def _route_scope(next_node: str):
    def route(state: PlateCoachState) -> str:
        return next_node if state.get("in_scope") else END

    return route


def _route_gate(state: PlateCoachState) -> str:
    return "nutrition_analysis" if state.get("gate_ok") else END


def _route_evaluator(state: PlateCoachState) -> str:
    return "propose_save" if state.get("evaluator_ok") else "improvement_planner"


def build_analysis_graph(checkpointer=None):
    """The main photo-analysis flow (see module docstring)."""
    graph = StateGraph(PlateCoachState)

    graph.add_node("vision_intake", vision_intake_node)
    graph.add_node("input_gate", input_gate_node)
    graph.add_node("nutrition_analysis", nutrition_analysis_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("improvement_planner", improvement_planner_node)
    graph.add_node("goal_adjuster", goal_adjuster_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("propose_save", propose_save_node)
    graph.add_node("hitl_gate", hitl_gate_node)

    graph.add_edge(START, "vision_intake")
    graph.add_edge("vision_intake", "input_gate")
    graph.add_conditional_edges("input_gate", _route_gate, {"nutrition_analysis": "nutrition_analysis", END: END})
    graph.add_edge("nutrition_analysis", "scoring")
    graph.add_edge("scoring", "improvement_planner")
    graph.add_edge("improvement_planner", "goal_adjuster")
    graph.add_edge("goal_adjuster", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _route_evaluator,
        {"propose_save": "propose_save", "improvement_planner": "improvement_planner"},
    )
    graph.add_edge("propose_save", "hitl_gate")
    graph.add_edge("hitl_gate", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def build_question_graph():
    """Free-text nutrition question: scope guard, then the RAG-grounded answer."""
    graph = StateGraph(PlateCoachState)
    graph.add_node("scope_guard", scope_guard_node)
    graph.add_node("nutrition_qa", nutrition_qa_node)
    graph.add_edge(START, "scope_guard")
    graph.add_conditional_edges("scope_guard", _route_scope("nutrition_qa"), {"nutrition_qa": "nutrition_qa", END: END})
    graph.add_edge("nutrition_qa", END)
    return graph.compile()


def build_simulation_graph():
    """What-if question against the last analyzed meal: scope guard, then simulation."""
    graph = StateGraph(PlateCoachState)
    graph.add_node("scope_guard", scope_guard_node)
    graph.add_node("simulation_handler", simulation_handler_node)
    graph.add_edge(START, "scope_guard")
    graph.add_conditional_edges(
        "scope_guard", _route_scope("simulation_handler"), {"simulation_handler": "simulation_handler", END: END}
    )
    graph.add_edge("simulation_handler", END)
    return graph.compile()


def build_history_graph():
    """Single-node graph for the up-to-20-photo weekly trend analysis command."""
    graph = StateGraph(PlateCoachState)
    graph.add_node("history_analyzer", history_analyzer_node)
    graph.add_edge(START, "history_analyzer")
    graph.add_edge("history_analyzer", END)
    return graph.compile()


def build_export_graph(checkpointer=None):
    """Export the conversation summary: the HITL gate asks before the file is written."""
    graph = StateGraph(PlateCoachState)
    graph.add_node("hitl_gate", hitl_gate_node)
    graph.add_edge(START, "hitl_gate")
    graph.add_edge("hitl_gate", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
