"""
CrewAI Task definitions.
Tasks receive agents at build time and accept {user_query} as a runtime input.
"""

from crewai import Task, Agent


def build_tasks(agents: dict[str, Agent]) -> list[Task]:
    """
    Build all four research tasks in pipeline order.
    Returns list in execution order: plan → gather → verify → write.
    """

    create_research_plan = Task(
        description=(
            "Based on the user's query, break it down into specific topics and key questions, "
            "and create a focused research plan.\n"
            "The user's query is: {user_query}"
        ),
        expected_output=(
            "A research plan with main research topics to investigate, "
            "key questions for each topic, and success criteria for the research."
        ),
        agent=agents["research_planner"],
    )

    gather_research_data = Task(
        description=(
            "Using the research plan, collect information on all identified topics. "
            "Cite all sources used."
        ),
        expected_output=(
            "Comprehensive research data including: information for each "
            "research topic, and citations used along with source credibility notes."
        ),
        agent=agents["researcher"],
    )

    verify_information_quality = Task(
        description=(
            "Review all collected research. Identify any conflicting information, "
            "potential misinformation, or gaps that need addressing."
        ),
        expected_output=(
            "A report with all the original data you got plus any "
            "verified facts vs. questionable information, make sure this is as comprehensive "
            "as possible for final report generation."
        ),
        agent=agents["fact_checker"],
    )

    write_final_report = Task(
        description=(
            "Create a comprehensive report that answers the original query using all verified "
            "research data. Structure it with clear sections, include citations, and provide "
            "actionable insights."
        ),
        expected_output=(
            "A final research report containing: executive summary, detailed "
            "findings that answer the user query, supporting evidence and analysis, complete "
            "source citations."
        ),
        agent=agents["report_writer"],
    )

    return [
        create_research_plan,
        gather_research_data,
        verify_information_quality,
        write_final_report,
    ]
