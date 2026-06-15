"""
Task guardrails — validate CrewAI task outputs before they are accepted.

write_report_guardrail enforces that the final report always contains:
  - A Summary section  (## Summary ...)
  - An Insights or Recommendations section
  - A Citations or References section

CrewAI retries the task automatically when a guardrail returns (False, reason).
"""

import re


def write_report_guardrail(output) -> tuple[bool, str]:
    """
    Validate the final report output.

    Returns:
        (True, raw_output)  — output is accepted
        (False, reason)     — CrewAI will retry the task with the reason as feedback
    """
    try:
        raw = output if isinstance(output, str) else output.raw
    except Exception as e:
        return (False, f"Error retrieving output `.raw`: {e}")

    lower = raw.lower()

    if not re.search(r"#+.*summary", lower):
        return (
            False,
            "The report must include a Summary section with a header like '## Summary'.",
        )

    if not re.search(r"#+.*insights|#+.*recommendations", lower):
        return (
            False,
            "The report must include an Insights or Recommendations section "
            "with a header like '## Insights' or '## Recommendations'.",
        )

    if not re.search(r"#+.*citations|#+.*references", lower):
        return (
            False,
            "The report must include a Citations or References section "
            "with a header like '## Citations' or '## References'.",
        )

    return (True, raw)
