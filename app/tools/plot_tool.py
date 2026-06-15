"""
CustomPlotTool — CrewAI BaseTool that auto-generates matplotlib/seaborn charts
from research text using an LLM to extract quantifiable data.

Plots are saved as PNGs under the configured PLOTS_DIR (default: plots/).
The tool returns a summary string listing the saved filenames so the
report writer agent can reference them in the final report.
"""

import json
import os
import logging
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for server use
import matplotlib.pyplot as plt
import seaborn as sns

from crewai.tools import BaseTool
from crewai import LLM

from app.core.config import settings

logger = logging.getLogger(__name__)

PLOTS_DIR = os.environ.get("PLOTS_DIR", "plots")

EXTRACTION_PROMPT = """\
You are an expert data visualization assistant. Analyze the provided research text \
and identify meaningful, insightful charts that can be created to visualize quantifiable \
data supporting the research's key insights and findings. Only suggest charts for data \
that includes numerical values, measurable trends, comparisons, or categorical \
distributions that can be effectively plotted.

Focus on creating visualizations that highlight trends, comparisons, distributions, \
or relationships that add value to the research. Avoid suggesting charts for purely \
qualitative or non-quantifiable information.

For each chart, provide a JSON object with:
  - "chart_type" (string: choose from "line" for trends over time/continuous, \
"bar" for comparisons, "histogram" for distributions, "scatter" for relationships, \
"pie" for proportions)
  - "x_axis" (string: variable name for x-axis, e.g., "year", "category")
  - "y_axis" (string: variable name for y-axis, e.g., "value", "count")
  - "color" (string: optional variable for color grouping/hue, or null if not applicable)
  - "Title" (string: descriptive, insightful title that explains what the chart shows)
  - "data" (dictionary: keys matching x_axis, y_axis, and color variables; \
values as lists of extracted numerical/categorical data from the research)

Ensure data is accurately extracted and formatted as lists. If a variable has multiple \
series (e.g., for color), include all in the data dictionary.

If no quantifiable data suitable for meaningful visualization is present in the research, \
return an empty array [].

Text:
{research}

Example output (return valid JSON only):
[
  {{"chart_type": "line", "x_axis": "year", "y_axis": "funding_amount", "color": "sector", \
"Title": "AI Research Funding Trends by Sector", \
"data": {{"year": [2020, 2021, 2022], "funding_amount": [2.5, 3.8, 5.2], \
"sector": ["Healthcare", "Finance", "Tech"]}}}},
  {{"chart_type": "bar", "x_axis": "tool_name", "y_axis": "adoption_rate", "color": null, \
"Title": "Market Adoption Rates of AI Tools", \
"data": {{"tool_name": ["ToolA", "ToolB", "ToolC"], "adoption_rate": [45, 67, 23]}}}}
]

Return only the JSON array, no additional text or explanations.
"""


class CustomPlotTool(BaseTool):
    """
    Auto-generates charts from research text.

    Pass the full validated research as a string. The tool uses an LLM to
    extract quantifiable data, then renders charts with matplotlib/seaborn
    and saves them as PNGs under the plots/ directory.
    """

    name: str = "Create custom plots"
    description: str = (
        "This is a tool for automatically creating custom plots based on a research result. "
        "It automatically generates plots from a text input, which should have fact-checked "
        "information. Pass the full validated information gathered so far as a string."
    )

    def _run(self, research: str) -> str:
        try:
            # ── Step 1: Ask LLM to extract chart specs ─────────────────────
            prompt = EXTRACTION_PROMPT.format(research=research)
            llm = LLM(model=settings.MODEL)
            llm_response = llm.call([{"role": "user", "content": prompt}])

            # Clean potential markdown fences
            llm_response = llm_response.strip()
            if llm_response.startswith("```json"):
                llm_response = llm_response[7:]
            if llm_response.endswith("```"):
                llm_response = llm_response[:-3]
            llm_response = llm_response.strip()

            # ── Step 2: Parse JSON ──────────────────────────────────────────
            charts_data = json.loads(llm_response)

            if not isinstance(charts_data, list) or len(charts_data) == 0:
                return "No quantifiable data found in the research to visualize."

            os.makedirs(PLOTS_DIR, exist_ok=True)
            plots_created: list[str] = []

            # ── Step 3: Render each chart ───────────────────────────────────
            for i, chart_info in enumerate(charts_data):
                try:
                    chart_type = (chart_info.get("chart_type") or "bar").lower()
                    x_axis = chart_info.get("x_axis", "x")
                    y_axis = chart_info.get("y_axis", "y")
                    title = chart_info.get("Title", f"Chart {i + 1}")
                    hue = chart_info.get("color") or None
                    data = chart_info.get("data", {})

                    df = pd.DataFrame(data)
                    if df.empty:
                        continue

                    plt.figure(figsize=(10, 6))

                    if chart_type == "line":
                        sns.lineplot(data=df, x=x_axis, y=y_axis, marker="o", hue=hue)
                    elif chart_type in ("bar", "column"):
                        sns.barplot(data=df, x=x_axis, y=y_axis, hue=hue)
                    elif chart_type == "histogram":
                        plt.hist(df[y_axis], bins=10, alpha=0.7)
                        plt.xlabel(y_axis)
                        plt.ylabel("Frequency")
                    elif chart_type == "scatter":
                        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue)
                    elif chart_type == "pie":
                        plt.pie(df[y_axis], labels=df[x_axis], autopct="%1.1f%%", startangle=90)
                        plt.axis("equal")
                    else:
                        # Fallback to bar
                        sns.barplot(data=df, x=x_axis, y=y_axis, hue=hue)

                    plt.title(title)
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    # ── Step 4: Save ────────────────────────────────────────
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(PLOTS_DIR, f"plot_{i + 1}_{timestamp}.png")
                    plt.savefig(filename, dpi=150, bbox_inches="tight")
                    plt.close()
                    plots_created.append(filename)
                    logger.info(f"Plot saved: {filename}")

                except Exception as e:
                    logger.warning(f"Error creating chart {i + 1}: {e}")
                    plt.close()
                    continue

            if plots_created:
                return (
                    f"Successfully created {len(plots_created)} plot(s): "
                    + ", ".join(plots_created)
                )
            return "No plots could be created from the extracted data."

        except json.JSONDecodeError as e:
            return f"Error parsing LLM response as JSON: {e}"
        except Exception as e:
            return f"Error generating plots: {e}"
