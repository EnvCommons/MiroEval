"""
MiroEval Environment — Point-wise quality evaluation for deep research agents.

Implements the MiroEval benchmark (arXiv:2603.28407) as an OpenReward environment.
Agents receive a research query, use web search tools to gather information,
and submit a comprehensive research report graded by a 5-stage hierarchical
LLM pipeline using GPT-5.1 (faithful to the original MiroEval implementation).

Reference: https://github.com/MiroMindAI/MiroEval
"""

import json
import re
from typing import List

import openai
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

from grading import PointwiseGrader
from utils import load_tasks


# ── Pydantic schemas ──

class TaskSpec(BaseModel):
    id: str
    query: str
    domain: str
    language: str


class WebSearchInput(BaseModel, extra="forbid"):
    query: str = Field(..., description="Search query string")


class FetchUrlInput(BaseModel, extra="forbid"):
    url: str = Field(..., description="URL to fetch content from")


class ReportInput(BaseModel, extra="forbid"):
    report: str = Field(
        ...,
        description="Your comprehensive research report addressing the query.",
    )


# ── Bilingual instructions ──

CHINESE_INSTRUCTIONS = """您的任务是对以下研究问题进行深入调查并撰写一份全面的研究报告。

可用工具:
1. web_search(query: str) - 搜索网络信息，返回标题、URL和摘要
2. fetch_url(url: str) - 获取特定URL的完整内容
3. submit_report(report: str) - 提交最终研究报告（结束任务）

说明:
1. 使用 web_search 查找相关信息，从不同角度搜索以全面覆盖
2. 使用 fetch_url 获取有价值URL的完整内容
3. 准备好后，使用 submit_report 提交您的研究报告

重要提示: 这个问题需要深入研究。请充分调查后再提交报告。"""

ENGLISH_INSTRUCTIONS = """Your task is to conduct in-depth research on the question below and produce a comprehensive research report.

Available Tools:
1. web_search(query: str) - Search for information, returns titles, URLs, and snippets
2. fetch_url(url: str) - Fetch full content from a specific URL
3. submit_report(report: str) - Submit your final research report (ends the task)

Instructions:
1. Use web_search to find relevant information from multiple angles
2. Use fetch_url to get complete content from promising URLs
3. When ready, use submit_report with your comprehensive research report

Important: This question requires thorough research. Investigate fully before submitting."""


# ── Load data at module level ──

tasks, full_tasks = load_tasks()


# ── Environment class ──

class MiroEval(Environment):
    """
    MiroEval environment: deep research with point-wise quality grading.

    Agent workflow:
    1. Receives a research question (Chinese or English)
    2. Uses web_search to gather information (multiple searches allowed)
    3. Uses fetch_url to read detailed content from URLs
    4. Submits comprehensive research report via submit_report
    5. Report is graded by GPT-5.1 using MiroEval's 5-stage pipeline
    6. Receives reward (0-10) with dimension scores

    MiroEval Grading Dimensions:
    - Coverage: Breadth, depth, and relevance
    - Insight: Depth, originality, logic, value of analysis
    - Instruction Following: Meeting all requirements
    - Clarity: Readability, structure, ease of understanding
    - 1-3 query-specific dimensions (dynamically generated)

    Reference: https://arxiv.org/abs/2603.28407
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "openai_api_key required in secrets for LLM grading. "
                "Pass secrets={'openai_api_key': '...', 'tavily_api_key': '...'}"
            )

        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            raise ValueError(
                "tavily_api_key required in secrets for web search. "
                "Pass secrets={'openai_api_key': '...', 'tavily_api_key': '...'}"
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
        self.grader = PointwiseGrader(client=self.openai_client, model="gpt-5.1")

    async def get_prompt(self) -> List[TextBlock]:
        """Generate research prompt with bilingual support."""
        is_chinese = any("\u4e00" <= c <= "\u9fff" for c in self.validated.query)
        instructions = CHINESE_INSTRUCTIONS if is_chinese else ENGLISH_INSTRUCTIONS

        prompt_text = f"""Research Query:

{self.validated.query}

{instructions}"""

        return [TextBlock(type="text", text=prompt_text)]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "text":
            return tasks  # type: ignore
        raise ValueError(f"Unknown split: {split}. Available: text")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["text"]

    # ── Web tools ──

    @tool
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """Search the web for information. Returns titles, URLs, and snippets."""
        try:
            response = await self.tavily_client.search(
                query=params.query,
                search_depth="advanced",
                max_results=8,
            )
            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(type="text", text="No search results found.")],
                    metadata={"query": params.query, "results": []},
                    reward=0.0,
                    finished=False,
                )

            display_parts = [f"Search results for: {params.query}\n"]
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("content", "")
                display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

            return ToolOutput(
                blocks=[TextBlock(type="text", text="\n".join(display_parts))],
                metadata={"query": params.query, "results": results, "count": len(results)},
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Web search failed: {e}")],
                metadata={"query": params.query, "error": str(e)},
                reward=0.0,
                finished=False,
            )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """Fetch full text content from a URL."""
        try:
            response = await self.tavily_client.extract(urls=[params.url])
            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(type="text", text=f"No content extracted from {params.url}")],
                    metadata={"url": params.url, "results": []},
                    reward=0.0,
                    finished=False,
                )

            raw_content = results[0].get("raw_content", "")
            max_length = 12000
            if len(raw_content) > max_length:
                raw_content = raw_content[:max_length] + "...\n[Content truncated]"

            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Content from {params.url}:\n\n{raw_content}")],
                metadata={"url": params.url, "length": len(raw_content)},
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Failed to fetch URL: {e}")],
                metadata={"url": params.url, "error": str(e)},
                reward=0.0,
                finished=False,
            )

    # ── Report submission + grading ──

    @tool
    async def submit_report(self, params: ReportInput) -> ToolOutput:
        """Submit your final research report for evaluation."""
        result = await self.grader.grade_report(
            task_prompt=self.validated.query,
            report=params.report,
        )

        total_score = result["total_score"]

        # Format dimension scores for display
        dim_lines = []
        for dim_key, score in result["dimension_scores"].items():
            if score is not None:
                dim_lines.append(f"  {dim_key}: {score:.2f}/10")
            else:
                dim_lines.append(f"  {dim_key}: FAILED")
        dim_text = "\n".join(dim_lines)

        display = f"""Report Evaluated

Total Quality Score: {total_score:.2f}/10.0

Dimension Scores:
{dim_text}"""

        return ToolOutput(
            metadata={
                "task_id": self.validated.id,
                "total_score": total_score,
                "dimension_scores": result["dimension_scores"],
                "dimension_weights": result["dimension_weights"],
                "additional_dimensions": result["additional_dimensions"],
            },
            blocks=[TextBlock(type="text", text=display)],
            reward=total_score,
            finished=True,
        )
