"""LangGraph workflow definitions for the Plate Coach agent.

Three compiled graphs share the same node module (agent.nodes), since
simulation/history don't belong on the linear analysis path:

- build_analysis_graph(): the main 10-node flow (scope guard -> vision intake
  -> gate -> nutrition analysis -> scoring -> planner -> goal adjuster ->
  evaluator, with conditional loop-backs on the gate and evaluator).
- build_simulation_graph(): a single-node graph wrapping simulation_handler_node,
  invoked per what-if question against the last known analysis state.
- build_history_graph(): a single-node graph wrapping history_analyzer_node.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    evaluator_node,
    goal_adjuster_node,
    hitl_gate_node,
    history_analyzer_node,
    improvement_planner_node,
    input_gate_node,
    nutrition_analysis_node,
    scope_guard_node,
    scoring_node,
    simulation_handler_node,
    vision_intake_node,
)
from agent.state import PlateCoachState

MAX_EVALUATOR_RETRIES = 1


def _route_scope(state: PlateCoachState) -> str:
    return "vision_intake" if state.get("in_scope", True) else END


def _route_gate(state: PlateCoachState) -> str:
    return "nutrition_analysis" if state.get("gate_ok") else END


def _route_evaluator(state: PlateCoachState) -> str:
    if state.get("evaluator_ok"):
        return "hitl_gate"
    return "improvement_planner"


def build_analysis_graph():
    """The main linear analysis flow, compiled with a checkpointer so the
    hitl_gate node's interrupt() can pause and resume across CLI turns."""
    graph = StateGraph(PlateCoachState)

    graph.add_node("scope_guard", scope_guard_node)
    graph.add_node("vision_intake", vision_intake_node)
    graph.add_node("input_gate", input_gate_node)
    graph.add_node("nutrition_analysis", nutrition_analysis_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("improvement_planner", improvement_planner_node)
    graph.add_node("goal_adjuster", goal_adjuster_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("hitl_gate", hitl_gate_node)

    graph.add_edge(START, "scope_guard")
    graph.add_conditional_edges("scope_guard", _route_scope, {"vision_intake": "vision_intake", END: END})

    graph.add_edge("vision_intake", "input_gate")
    graph.add_conditional_edges(
        "input_gate", _route_gate, {"nutrition_analysis": "nutrition_analysis", END: END}
    )

    graph.add_edge("nutrition_analysis", "scoring")
    graph.add_edge("scoring", "improvement_planner")
    graph.add_edge("improvement_planner", "goal_adjuster")
    graph.add_edge("goal_adjuster", "evaluator")
    graph.add_conditional_edges(
        "evaluator", _route_evaluator, {"hitl_gate": "hitl_gate", "improvement_planner": "improvement_planner"}
    )
    graph.add_edge("hitl_gate", END)

    return graph.compile(checkpointer=InMemorySaver())


def build_simulation_graph():
    """Single-node graph for free-form what-if questions, invoked repeatedly
    against the last known analysis state (not part of the linear analysis path)."""
    graph = StateGraph(PlateCoachState)
    graph.add_node("simulation_handler", simulation_handler_node)
    graph.add_edge(START, "simulation_handler")
    graph.add_edge("simulation_handler", END)
    return graph.compile()


def build_history_graph():
    """Single-node graph for the up-to-20-photo weekly trend analysis command."""
    graph = StateGraph(PlateCoachState)
    graph.add_node("history_analyzer", history_analyzer_node)
    graph.add_edge(START, "history_analyzer")
    graph.add_edge("history_analyzer", END)
    return graph.compile()
