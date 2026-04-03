"""
Comprehensive tests for MiroEval environment.

Categories:
1. Data Loading (no LLM)
2. JSON Extraction (no LLM)
3. Aggregation (no LLM)
4. Default Fallbacks (no LLM)
5. Environment Class (no LLM)
6. Integration Tests (require LLM + Tavily, marked @pytest.mark.integration)
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from grading import FIXED_DIMENSIONS, PointwiseGrader
from miroeval import MiroEval, ReportInput, WebSearchInput, FetchUrlInput, ViewAttachmentInput, tasks
from utils import extract_json_from_analysis_output, extract_json_from_response, load_tasks


# ══════════════════════════════════════════════════════════════════════════════
# Category 1: Data Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLoading:
    def test_task_count(self):
        """100 total tasks (70 text + 30 multimodal)."""
        loaded_tasks, _ = load_tasks()
        assert len(loaded_tasks) == 100

    def test_text_task_count(self):
        """70 text tasks have no files."""
        loaded_tasks, _ = load_tasks()
        text_tasks = [t for t in loaded_tasks if not t["files"]]
        assert len(text_tasks) == 70

    def test_multimodal_task_count(self):
        """30 multimodal tasks have files."""
        loaded_tasks, _ = load_tasks()
        mm_tasks = [t for t in loaded_tasks if t["files"]]
        assert len(mm_tasks) == 30

    def test_task_schema(self):
        """Each task has required fields."""
        loaded_tasks, _ = load_tasks()
        for task in loaded_tasks:
            assert "id" in task
            assert "query" in task
            assert "domain" in task
            assert "language" in task
            assert "files" in task
            assert len(task["query"]) > 0

    def test_multimodal_files_schema(self):
        """Multimodal tasks have properly structured files."""
        loaded_tasks, _ = load_tasks()
        mm_tasks = [t for t in loaded_tasks if t["files"]]
        for task in mm_tasks:
            for f in task["files"]:
                assert "filename" in f
                assert "type" in f
                assert "dir" in f

    def test_splits(self):
        assert MiroEval.list_splits() == ["test"]

    def test_invalid_split(self):
        with pytest.raises(ValueError, match="Unknown split"):
            MiroEval.list_tasks("invalid")

    def test_task_ids_unique(self):
        loaded_tasks, _ = load_tasks()
        ids = [t["id"] for t in loaded_tasks]
        assert len(ids) == len(set(ids))

    def test_domains_valid(self):
        valid_domains = {
            "tech", "finance", "medical", "engineering", "business",
            "humanities", "science", "lifestyle", "cybersecurity",
            "education", "energy", "geopolitics", "health", "legal",
            "policy", "trade", "other",
        }
        loaded_tasks, _ = load_tasks()
        for task in loaded_tasks:
            assert task["domain"] in valid_domains, f"Unknown domain: {task['domain']}"

    def test_languages(self):
        loaded_tasks, _ = load_tasks()
        for task in loaded_tasks:
            assert task["language"] in ("en", "zh"), f"Unknown language: {task['language']}"

    def test_full_tasks_keyed_by_id(self):
        _, full = load_tasks()
        assert isinstance(full, dict)
        assert len(full) == 100

    def test_list_tasks_matches_load(self):
        env_tasks = MiroEval.list_tasks("test")
        loaded_tasks, _ = load_tasks()
        assert len(env_tasks) == len(loaded_tasks)

    def test_multimodal_ids_range(self):
        """Multimodal tasks have IDs 71-100."""
        loaded_tasks, _ = load_tasks()
        mm_tasks = [t for t in loaded_tasks if t["files"]]
        mm_ids = sorted([int(t["id"]) for t in mm_tasks])
        assert mm_ids[0] == 71
        assert mm_ids[-1] == 100


# ══════════════════════════════════════════════════════════════════════════════
# Category 2: JSON Extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestJsonExtraction:
    def test_extract_json_output_tags(self):
        text = '<analysis>reasoning</analysis>\n<json_output>\n[{"a": 1}]\n</json_output>'
        result = extract_json_from_analysis_output(text)
        assert result is not None
        assert json.loads(result) == [{"a": 1}]

    def test_extract_json_fenced_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = extract_json_from_response(text)
        assert result is not None
        assert json.loads(result) == {"key": "value"}

    def test_extract_json_raw_array(self):
        text = 'The result is [{"x": 1}, {"x": 2}]'
        result = extract_json_from_response(text)
        assert result is not None
        assert len(json.loads(result)) == 2

    def test_extract_json_raw_object(self):
        text = 'Here: {"coverage": 0.3, "insight": 0.7}'
        result = extract_json_from_response(text)
        assert result is not None
        assert json.loads(result)["coverage"] == 0.3

    def test_extract_json_none_on_invalid(self):
        assert extract_json_from_response("no json here at all") is None

    def test_extract_json_none_on_empty(self):
        assert extract_json_from_response("") is None
        assert extract_json_from_response(None) is None  # type: ignore

    def test_extract_analysis_output_prefers_tags(self):
        text = (
            '<analysis>some analysis</analysis>\n'
            '<json_output>{"from_tags": true}</json_output>\n'
            '```json\n{"from_fence": true}\n```'
        )
        result = extract_json_from_analysis_output(text)
        assert json.loads(result) == {"from_tags": True}

    def test_extract_generic_fence(self):
        text = "```\n[1, 2, 3]\n```"
        result = extract_json_from_response(text)
        assert result is not None
        assert json.loads(result) == [1, 2, 3]

    def test_extract_analysis_fallback(self):
        text = '```json\n{"fallback": true}\n```'
        result = extract_json_from_analysis_output(text)
        assert result is not None
        assert json.loads(result) == {"fallback": True}


# ══════════════════════════════════════════════════════════════════════════════
# Category 3: Aggregation
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregation:
    def _make_grader(self):
        return object.__new__(PointwiseGrader)

    def test_hierarchical_scores_basic(self):
        grader = self._make_grader()
        scores = {
            "coverage": [
                {"criterion": "c1", "analysis": "...", "report_score_0_to_10": 8.0},
                {"criterion": "c2", "analysis": "...", "report_score_0_to_10": 6.0},
            ],
            "insight": [
                {"criterion": "c1", "analysis": "...", "report_score_0_to_10": 7.0},
            ],
        }
        all_criteria = {
            "coverage": [{"criterion": "c1", "weight": 0.6}, {"criterion": "c2", "weight": 0.4}],
            "insight": [{"criterion": "c1", "weight": 1.0}],
        }
        dim_weights = {"coverage": 0.5, "insight": 0.5}
        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)
        assert abs(result["total_weighted_score"] - 7.1) < 0.01

    def test_hierarchical_scores_failed_dimension(self):
        grader = self._make_grader()
        scores = {"coverage": [{"criterion": "c1", "analysis": ".", "report_score_0_to_10": 8.0}]}
        all_criteria = {
            "coverage": [{"criterion": "c1", "weight": 1.0}],
            "insight": [{"criterion": "c1", "weight": 1.0}],
        }
        dim_weights = {"coverage": 0.6, "insight": 0.4}
        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)
        assert abs(result["total_weighted_score"] - 8.0) < 0.01

    def test_hierarchical_scores_all_fail(self):
        grader = self._make_grader()
        result = grader.calculate_hierarchical_scores(
            scores={},
            all_criteria={"coverage": [{"criterion": "c1", "weight": 1.0}]},
            dimension_weights={"coverage": 1.0},
        )
        assert result["total_weighted_score"] == 0.0

    def test_single_dimension_single_criterion(self):
        grader = self._make_grader()
        scores = {"clarity": [{"criterion": "readable", "analysis": "ok", "report_score_0_to_10": 5.5}]}
        all_criteria = {"clarity": [{"criterion": "readable", "weight": 1.0}]}
        result = grader.calculate_hierarchical_scores(scores, all_criteria, {"clarity": 1.0})
        assert abs(result["total_weighted_score"] - 5.5) < 0.01

    def test_empty_dim_scores_treated_as_failed(self):
        grader = self._make_grader()
        result = grader.calculate_hierarchical_scores(
            scores={"coverage": []},
            all_criteria={"coverage": [{"criterion": "c1", "weight": 1.0}]},
            dimension_weights={"coverage": 1.0},
        )
        assert result["coverage_score"] is None
        assert result["total_weighted_score"] == 0.0

    def test_reward_range(self):
        grader = self._make_grader()
        scores = {"coverage": [{"criterion": "c1", "analysis": ".", "report_score_0_to_10": 10.0}]}
        all_criteria = {"coverage": [{"criterion": "c1", "weight": 1.0}]}
        result = grader.calculate_hierarchical_scores(scores, all_criteria, {"coverage": 1.0})
        assert 0.0 <= result["total_weighted_score"] <= 10.0

    def test_multiple_dimensions_weighted(self):
        grader = self._make_grader()
        scores = {
            "coverage": [{"criterion": "c", "analysis": ".", "report_score_0_to_10": 7.0}],
            "insight": [{"criterion": "c", "analysis": ".", "report_score_0_to_10": 8.0}],
            "instruction_following": [{"criterion": "c", "analysis": ".", "report_score_0_to_10": 9.0}],
            "clarity": [{"criterion": "c", "analysis": ".", "report_score_0_to_10": 6.0}],
        }
        all_criteria = {k: [{"criterion": "c", "weight": 1.0}] for k in scores}
        dim_weights = {"coverage": 0.3, "insight": 0.3, "instruction_following": 0.2, "clarity": 0.2}
        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)
        assert abs(result["total_weighted_score"] - 7.5) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Category 4: Default Fallbacks
# ══════════════════════════════════════════════════════════════════════════════

class TestDefaultFallbacks:
    def _make_grader(self):
        return object.__new__(PointwiseGrader)

    def test_default_weights_count(self):
        grader = self._make_grader()
        dims = [{"meta_dimension_name": "Market Timing"}]
        assert len(grader._get_default_weights(dims)) == 5

    def test_default_weights_sum(self):
        grader = self._make_grader()
        dims = [{"meta_dimension_name": "X"}, {"meta_dimension_name": "Y"}]
        assert abs(sum(grader._get_default_weights(dims).values()) - 1.0) < 0.001

    def test_default_weights_no_additional(self):
        grader = self._make_grader()
        weights = grader._get_default_weights([])
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_default_criteria_count(self):
        grader = self._make_grader()
        assert len(grader._get_default_criteria("coverage", "Breadth and depth")) == 3

    def test_default_criteria_weights_sum(self):
        grader = self._make_grader()
        criteria = grader._get_default_criteria("insight", "Quality of analysis")
        assert abs(sum(c["weight"] for c in criteria) - 1.0) < 0.001

    def test_default_criteria_fields(self):
        grader = self._make_grader()
        for c in grader._get_default_criteria("clarity"):
            assert "criterion" in c and "explanation" in c and "weight" in c

    def test_snake_case_normalization(self):
        grader = self._make_grader()
        assert "market_timing" in grader._get_default_weights([{"meta_dimension_name": "Market Timing"}])
        assert "data_quality" in grader._get_default_weights([{"meta_dimension_name": "Data-Quality"}])


# ══════════════════════════════════════════════════════════════════════════════
# Category 5: Environment Class
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvironmentClass:
    def _text_task(self):
        return [t for t in tasks if not t["files"]][0]

    def _mm_task(self):
        return [t for t in tasks if t["files"]][0]

    def test_environment_missing_openai_secret(self):
        with pytest.raises(ValueError, match="openai_api_key"):
            MiroEval(task_spec=self._text_task(), secrets={"tavily_api_key": "tvly-xxx"})

    def test_environment_missing_tavily_secret(self):
        with pytest.raises(ValueError, match="tavily_api_key"):
            MiroEval(task_spec=self._text_task(), secrets={"openai_api_key": "sk-xxx"})

    def test_environment_init_text_task(self):
        env = MiroEval(
            task_spec=self._text_task(),
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        assert not env._is_multimodal
        assert env.attachment_contents == []

    def test_environment_init_multimodal_task(self):
        env = MiroEval(
            task_spec=self._mm_task(),
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        assert env._is_multimodal
        # attachment_contents may be empty if files not on disk yet

    @pytest.mark.asyncio
    async def test_prompt_contains_query(self):
        env = MiroEval(
            task_spec=self._text_task(),
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        blocks = await env.get_prompt()
        assert len(blocks) >= 1
        assert self._text_task()["query"][:50] in blocks[0].text

    @pytest.mark.asyncio
    async def test_prompt_multimodal_lists_attachments(self):
        env = MiroEval(
            task_spec=self._mm_task(),
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        blocks = await env.get_prompt()
        text = blocks[0].text
        # Should list the attachment filenames
        for f in self._mm_task()["files"]:
            assert f["filename"] in text

    @pytest.mark.asyncio
    async def test_prompt_bilingual_chinese(self):
        zh_task = next((t for t in tasks if t["language"] == "zh" and not t["files"]), None)
        assert zh_task is not None
        env = MiroEval(task_spec=zh_task, secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"})
        blocks = await env.get_prompt()
        assert "可用工具" in blocks[0].text

    @pytest.mark.asyncio
    async def test_prompt_bilingual_english(self):
        en_task = next((t for t in tasks if t["language"] == "en" and not t["files"]), None)
        assert en_task is not None
        env = MiroEval(task_spec=en_task, secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"})
        blocks = await env.get_prompt()
        assert "Available Tools" in blocks[0].text

    @pytest.mark.asyncio
    async def test_view_attachment_invalid_filename(self):
        env = MiroEval(
            task_spec=self._mm_task(),
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        result = await env.view_attachment(ViewAttachmentInput(filename="nonexistent.jpg"))
        assert result.finished is False
        assert "not found" in result.blocks[0].text.lower() or "not found" in str(result.metadata)

    def test_task_spec_validation(self):
        with pytest.raises(Exception):
            MiroEval(task_spec={"id": "1"}, secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"})


# ══════════════════════════════════════════════════════════════════════════════
# Category 6: Integration Tests (require real API keys)
# ══════════════════════════════════════════════════════════════════════════════

def _get_secrets():
    from dotenv import load_dotenv
    load_dotenv("../.env")
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not openai_key or not tavily_key:
        pytest.skip("OPENAI_API_KEY and TAVILY_API_KEY required for integration tests")
    return {"openai_api_key": openai_key, "tavily_api_key": tavily_key}


_en_tasks = [t for t in tasks if t["language"] == "en" and not t["files"]]
_integration_task = _en_tasks[0] if _en_tasks else tasks[0]


@pytest.mark.integration
class TestIntegration:
    @pytest.mark.asyncio
    async def test_web_search_returns_results(self):
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)
        result = await env.web_search(WebSearchInput(query="quantum computing advances 2025"))
        assert result.finished is False
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_fetch_url_returns_content(self):
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)
        result = await env.fetch_url(FetchUrlInput(url="https://en.wikipedia.org/wiki/Python_(programming_language)"))
        assert result.finished is False
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_good_vs_bad_report(self):
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)

        import openai as oai
        client = oai.AsyncClient(api_key=secrets["openai_api_key"])
        response = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": f"Write a brief 500-word research report on: {_integration_task['query'][:500]}"}],
        )
        good_report = response.choices[0].message.content or ""

        good_result = await env.submit_report(ReportInput(report=good_report))
        bad_result = await env.submit_report(ReportInput(report="This is garbage. Nothing relevant here."))

        assert good_result.reward > bad_result.reward
        assert good_result.finished is True

    @pytest.mark.asyncio
    async def test_full_pipeline_structure(self):
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)
        result = await env.submit_report(
            ReportInput(report="A moderate quality report discussing the topic with some analysis and depth.")
        )
        assert result.finished is True
        assert "total_score" in result.metadata
        assert "dimension_scores" in result.metadata
        assert "dimension_weights" in result.metadata

    @pytest.mark.asyncio
    async def test_reward_in_range(self):
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)
        result = await env.submit_report(ReportInput(report="A research report with basic coverage."))
        assert 0.0 <= result.reward <= 10.0

    @pytest.mark.asyncio
    async def test_empty_report_low_score(self):
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)
        result = await env.submit_report(ReportInput(report=""))
        assert result.reward < 3.0
        assert result.finished is True
