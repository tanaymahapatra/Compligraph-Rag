from langgraph.graph import StateGraph, END

from src.state import GraphState

from src.nodes import (
    input_guardrail,
    reject_request,
    query_rewriter,
    retrieve_docs,
    grade_documents,
    web_search_fallback,
    generate_answer,
    compliance_auditor,
    MAX_AUDIT_ATTEMPTS,
)

# ============================================================
# CONDITIONAL ROUTERS
# ============================================================


def guardrail_router(state: GraphState):
    """
    Route safe queries into retrieval.
    Reject unsafe/unrelated queries.
    """

    if state.is_safe:
        return "query_rewriter"

    return "reject_request"


def grade_docs_router(state: GraphState):
    """
    If vector retrieval produced no sufficiently relevant
    documents, fall back to Tavily.

    Otherwise go directly to generation.
    """

    if state.web_search_used:
        return "web_search_fallback"

    return "generate_answer"


def audit_router(state: GraphState):
    """
    Route the answer based on the compliance auditor.

    APPROVED:
        END

    REJECTED:
        regenerate, unless the maximum number of attempts
        has already been reached.
    """

    if state.is_compliant is True:
        print("\n✅ AUDIT PASSED → END")
        return END

    loop_count = state.loop_count or 0

    if loop_count >= MAX_AUDIT_ATTEMPTS:
        print("\n⚠️ MAXIMUM AUDIT ATTEMPTS REACHED → END")

        return END

    print(
        f"\n🔄 AUDIT FAILED → REGENERATING "
        f"(attempt {loop_count + 1}/"
        f"{MAX_AUDIT_ATTEMPTS})"
    )

    return "generate_answer"


# ============================================================
# GRAPH BUILDER
# ============================================================


workflow = StateGraph(GraphState)


# ============================================================
# NODES
# ============================================================

workflow.add_node(
    "input_guardrail",
    input_guardrail,
)

workflow.add_node(
    "reject_request",
    reject_request,
)

workflow.add_node(
    "query_rewriter",
    query_rewriter,
)

workflow.add_node(
    "retrieve_docs",
    retrieve_docs,
)

workflow.add_node(
    "grade_documents",
    grade_documents,
)

workflow.add_node(
    "web_search_fallback",
    web_search_fallback,
)

workflow.add_node(
    "generate_answer",
    generate_answer,
)

workflow.add_node(
    "compliance_auditor",
    compliance_auditor,
)


# ============================================================
# ENTRY POINT
# ============================================================

workflow.set_entry_point("input_guardrail")


# ============================================================
# GUARDRAIL ROUTING
# ============================================================

workflow.add_conditional_edges(
    "input_guardrail",
    guardrail_router,
)


# ============================================================
# REJECT PATH
# ============================================================

workflow.add_edge(
    "reject_request",
    END,
)


# ============================================================
# RETRIEVAL PIPELINE
# ============================================================

workflow.add_edge(
    "query_rewriter",
    "retrieve_docs",
)

workflow.add_edge(
    "retrieve_docs",
    "grade_documents",
)


# ============================================================
# DOCUMENT ROUTING
# ============================================================

workflow.add_conditional_edges(
    "grade_documents",
    grade_docs_router,
)


# ============================================================
# WEB FALLBACK
# ============================================================

workflow.add_edge(
    "web_search_fallback",
    "generate_answer",
)


# ============================================================
# GENERATION → AUDIT
# ============================================================

workflow.add_edge(
    "generate_answer",
    "compliance_auditor",
)


# ============================================================
# AUDIT → END / REGENERATION
# ============================================================

workflow.add_conditional_edges(
    "compliance_auditor",
    audit_router,
)


# ============================================================
# COMPILE
# ============================================================

app = workflow.compile()
