"""
MiroEval Environment — Point-wise quality evaluation for deep research agents.

Implements the MiroEval benchmark (arXiv:2603.28407) as an OpenReward environment.
Agents receive a research query (with optional multimodal attachments), use web search
tools to gather information, and submit a comprehensive research report graded by a
5-stage hierarchical LLM pipeline using GPT-5.1 (faithful to original MiroEval).

Reference: https://github.com/MiroMindAI/MiroEval
"""

import base64
import logging
import mimetypes
from pathlib import Path
from typing import List, Optional, Union

import openai
from pydantic import BaseModel, Field

from openreward.environments import Environment, ImageBlock, JSONObject, TextBlock, ToolOutput, terminal, tool
from openreward.toolsets import WebToolset

from grading import PointwiseGrader
from utils import load_tasks, resolve_attachment_path

logger = logging.getLogger(__name__)


# ── Pydantic schemas ──

class TaskSpec(BaseModel):
    id: str
    query: str
    domain: str
    language: str
    files: list[dict] = []


class ViewAttachmentInput(BaseModel, extra="forbid"):
    filename: str = Field(..., description="Filename of the attachment to view (from the list in the prompt)")


class ReportInput(BaseModel, extra="forbid"):
    report: str = Field(
        ...,
        description="Your comprehensive research report addressing the query.",
    )


# ── Bilingual instructions ──

CHINESE_INSTRUCTIONS = """您的任务是对以下研究问题进行深入调查并撰写一份全面的研究报告。

可用工具:
1. web_search(query: str) - 搜索网络信息，返回标题、URL和摘要
2. web_fetch(url: str) - 获取特定URL的完整内容
3. view_attachment(filename: str) - 查看附件文件（图片、PDF等）

说明:
1. 如果有附件，先使用 view_attachment 查看附件内容
2. 使用 web_search 查找相关信息，从不同角度搜索以全面覆盖
3. 使用 web_fetch 获取有价值URL的完整内容
4. 准备好后，请直接以普通消息形式回复您的完整研究报告（不要调用任何工具）。您的整条回复将作为报告进行评分——请避免开场白和结束语。

重要提示: 这个问题需要深入研究。请充分调查后再提交报告。"""

ENGLISH_INSTRUCTIONS = """Your task is to conduct in-depth research on the question below and produce a comprehensive research report.

Available Tools:
1. web_search(query: str) - Search for information, returns titles, URLs, and snippets
2. web_fetch(url: str, prompt: str) - Fetch the content of a specific URL
3. view_attachment(filename: str) - View an attachment file (images, PDFs, etc.)

Instructions:
1. If attachments are provided, start by viewing them with view_attachment
2. Use web_search to find relevant information from multiple angles
3. Use web_fetch to get complete content from promising URLs
4. When ready, reply with your comprehensive research report as an ordinary message (no tool call). Your entire reply is graded as the report, so avoid preamble and closing remarks.

Important: This question requires thorough research. Investigate fully before submitting."""


# ── Attachment loading helpers ──

def _load_image_as_base64(path: Path) -> tuple[str, str]:
    """Load an image file and return (base64_data, mime_type)."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(str(path))
    return data, mime_type or "image/jpeg"


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pdfplumber."""
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def _render_pdf_pages_as_images(path: Path, max_pages: int = 20) -> list[tuple[str, str]]:
    """Render PDF pages as PNG images. Returns list of (base64_data, mime_type)."""
    from pdf2image import convert_from_path
    images = convert_from_path(str(path), dpi=150, first_page=1, last_page=max_pages)
    result = []
    for img in images:
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        result.append((b64, "image/png"))
    return result


# ── Load data at module level ──

tasks, full_tasks = load_tasks()


# ── Environment class ──

class MiroEval(Environment):
    """
    MiroEval environment: deep research with point-wise quality grading.

    Supports both text-only (70 tasks) and multimodal (30 tasks with attachments).
    100 total tasks in the 'test' split.

    Reference: https://arxiv.org/abs/2603.28407
    """

    # web_search / web_fetch come from the SDK rather than being hand-rolled here.
    # Which provider answers is process configuration (OPENREWARD_SEARCH_BACKEND,
    # default "backsearch"), so changing search provider needs no change here.
    #
    # The toolset owns the error split too: an unfetchable page stays tool output
    # the agent can act on, while a missing key or exhausted quota raises so the
    # rollout ends with a blank reward rather than a score that reads as a bad answer.
    toolsets = [WebToolset]

    # Search hits keep their snippets, as the prompt promises. Off in the SDK by
    # default, which would force a fetch per candidate just to triage results.
    web_include_snippets = True

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.validated = TaskSpec.model_validate(task_spec)

        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "openai_api_key required in secrets for LLM grading. "
                "Pass secrets={'openai_api_key': '...'}"
            )

        # Read live by WebToolset on every tool call, so the search backend takes its
        # credentials from the session rather than the server process. The configured
        # backend picks the key it needs: `api_key` for backsearch, `tavily_api_key`
        # for tavily. No up-front check — which key is required depends on the backend.
        self.search_secrets = secrets

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.grader = PointwiseGrader(client=self.openai_client, model="gpt-5.1")

        # Pre-load attachment content for grading (text representations)
        self.attachment_contents: list[dict[str, str]] = []
        if self.validated.files:
            self._load_attachment_contents()

    def _load_attachment_contents(self) -> None:
        """Load attachment content for the grader. Images as base64, PDFs/text as text."""
        for file_info in self.validated.files:
            filename = file_info["filename"]
            file_type = file_info.get("type", "unknown")
            path = resolve_attachment_path(self.validated.id, filename)
            if path is None:
                logger.warning(f"Attachment not found: {filename} for task {self.validated.id}")
                continue

            try:
                if file_type == "image" or path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
                    b64, mime = _load_image_as_base64(path)
                    self.attachment_contents.append({"image_base64": b64, "mime_type": mime})
                elif path.suffix.lower() == ".pdf":
                    text = _extract_pdf_text(path)
                    if text.strip():
                        self.attachment_contents.append({"text": f"[PDF: {filename}]\n{text}"})
                    else:
                        logger.warning(f"No text extracted from PDF: {filename}")
                else:
                    # Text/doc/other — read as text
                    text = path.read_text(encoding="utf-8", errors="replace")
                    self.attachment_contents.append({"text": f"[File: {filename}]\n{text}"})
            except Exception as e:
                logger.error(f"Failed to load attachment {filename}: {e}")

    @property
    def _is_multimodal(self) -> bool:
        return len(self.validated.files) > 0

    async def get_prompt(self) -> list[Union[TextBlock, ImageBlock]]:
        """Generate research prompt with bilingual support and attachment info."""
        is_chinese = any("\u4e00" <= c <= "\u9fff" for c in self.validated.query)
        instructions = CHINESE_INSTRUCTIONS if is_chinese else ENGLISH_INSTRUCTIONS

        prompt_text = f"Research Query:\n\n{self.validated.query}\n\n"

        if self._is_multimodal:
            if is_chinese:
                prompt_text += "附件文件（使用 view_attachment 工具查看）:\n"
            else:
                prompt_text += "Attachments (use view_attachment tool to view):\n"
            for f in self.validated.files:
                prompt_text += f"  - {f['filename']} ({f.get('type', 'file')})\n"
            prompt_text += "\n"

        prompt_text += instructions

        return [TextBlock(type="text", text=prompt_text)]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "test":
            return tasks  # type: ignore
        raise ValueError(f"Unknown split: {split}. Available: test")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["test"]

    # ── Web tools ──

    # ── Attachment viewing ──

    @tool
    async def view_attachment(self, params: ViewAttachmentInput) -> ToolOutput:
        """View an attachment file. Returns images as visual content, PDFs as rendered pages, text as text."""
        # Validate filename is in the task's file list
        valid_filenames = [f["filename"] for f in self.validated.files]
        if params.filename not in valid_filenames:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"File '{params.filename}' not found. Available: {valid_filenames}")],
                metadata={"error": "file_not_found", "available": valid_filenames},
                reward=0.0,
                finished=False,
            )

        path = resolve_attachment_path(self.validated.id, params.filename)
        if path is None:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Attachment file not available on disk: {params.filename}")],
                metadata={"error": "file_not_on_disk"},
                reward=0.0,
                finished=False,
            )

        try:
            suffix = path.suffix.lower()
            blocks: list[Union[TextBlock, ImageBlock]] = []

            if suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
                b64, mime = _load_image_as_base64(path)
                blocks.append(TextBlock(type="text", text=f"Image: {params.filename}"))
                blocks.append(ImageBlock(data=b64, mimeType=mime))

            elif suffix == ".pdf":
                blocks.append(TextBlock(type="text", text=f"PDF: {params.filename} (rendered as images)"))
                page_images = _render_pdf_pages_as_images(path)
                for i, (b64, mime) in enumerate(page_images):
                    blocks.append(TextBlock(type="text", text=f"Page {i+1}:"))
                    blocks.append(ImageBlock(data=b64, mimeType=mime))

            else:
                # Text/doc — read as text
                text = path.read_text(encoding="utf-8", errors="replace")
                max_length = 12000
                if len(text) > max_length:
                    text = text[:max_length] + "...\n[Content truncated]"
                blocks.append(TextBlock(type="text", text=f"File: {params.filename}\n\n{text}"))

            return ToolOutput(
                blocks=blocks,
                metadata={"filename": params.filename, "type": suffix},
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Failed to load attachment: {e}")],
                metadata={"filename": params.filename, "error": str(e)},
                reward=0.0,
                finished=False,
            )

    # ── Report submission + grading ──

    @terminal
    @tool
    async def submit_report(self, params: ReportInput) -> ToolOutput:
        """Grade the assistant's final message as a research report."""
        result = await self.grader.grade_report(
            task_prompt=self.validated.query,
            report=params.report,
            attachment_contents=self.attachment_contents if self.attachment_contents else None,
        )

        total_score = result["total_score"]

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
                "key_facts": result.get("key_facts"),
            },
            blocks=[TextBlock(type="text", text=display)],
            reward=total_score,
            finished=True,
        )
