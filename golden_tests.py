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

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(__file__))

from grading import FIXED_DIMENSIONS, PointwiseGrader
from miroeval import MiroEval, ReportInput, WebSearchInput, FetchUrlInput, tasks
from utils import extract_json_from_analysis_output, extract_json_from_response, load_tasks


# ══════════════════════════════════════════════════════════════════════════════
# Category 1: Data Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestDataLoading:
    def test_task_count(self):
        """Exactly 70 text tasks."""
        loaded_tasks, _ = load_tasks()
        assert len(loaded_tasks) == 70

    def test_task_schema(self):
        """Each task has required fields."""
        loaded_tasks, _ = load_tasks()
        for task in loaded_tasks:
            assert "id" in task
            assert "query" in task
            assert "domain" in task
            assert "language" in task
            assert len(task["query"]) > 0

    def test_splits(self):
        assert MiroEval.list_splits() == ["text"]

    def test_invalid_split(self):
        with pytest.raises(ValueError, match="Unknown split"):
            MiroEval.list_tasks("invalid")

    def test_task_ids_unique(self):
        """No duplicate IDs."""
        loaded_tasks, _ = load_tasks()
        ids = [t["id"] for t in loaded_tasks]
        assert len(ids) == len(set(ids))

    def test_domains_valid(self):
        """All domains are in a known set."""
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
        """All languages are 'en' or 'zh'."""
        loaded_tasks, _ = load_tasks()
        for task in loaded_tasks:
            assert task["language"] in ("en", "zh"), f"Unknown language: {task['language']}"

    def test_full_tasks_keyed_by_id(self):
        """full_tasks dict is keyed by string id."""
        _, full = load_tasks()
        assert isinstance(full, dict)
        for key in full:
            assert isinstance(key, str)

    def test_list_tasks_matches_load(self):
        """MiroEval.list_tasks returns same data as load_tasks."""
        env_tasks = MiroEval.list_tasks("text")
        loaded_tasks, _ = load_tasks()
        assert len(env_tasks) == len(loaded_tasks)
        assert env_tasks[0]["id"] == loaded_tasks[0]["id"]


# ══════════════════════════════════════════════════════════════════════════════
# Category 2: JSON Extraction
# ══════════════════════════════════════════════════════════════════════════════

class TestJsonExtraction:
    def test_extract_json_output_tags(self):
        text = '<analysis>reasoning</analysis>\n<json_output>\n[{"a": 1}]\n</json_output>'
        result = extract_json_from_analysis_output(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == [{"a": 1}]

    def test_extract_json_fenced_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = extract_json_from_response(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_extract_json_raw_array(self):
        text = 'The result is [{"x": 1}, {"x": 2}]'
        result = extract_json_from_response(text)
        assert result is not None
        parsed = json.loads(result)
        assert len(parsed) == 2

    def test_extract_json_raw_object(self):
        text = 'Here: {"coverage": 0.3, "insight": 0.7}'
        result = extract_json_from_response(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["coverage"] == 0.3

    def test_extract_json_none_on_invalid(self):
        assert extract_json_from_response("no json here at all") is None

    def test_extract_json_none_on_empty(self):
        assert extract_json_from_response("") is None
        assert extract_json_from_response(None) is None  # type: ignore

    def test_extract_analysis_output_prefers_tags(self):
        """<json_output> tags should be preferred over fenced blocks."""
        text = (
            '<analysis>some analysis</analysis>\n'
            '<json_output>{"from_tags": true}</json_output>\n'
            '```json\n{"from_fence": true}\n```'
        )
        result = extract_json_from_analysis_output(text)
        parsed = json.loads(result)
        assert parsed == {"from_tags": True}

    def test_extract_generic_fence(self):
        text = "```\n[1, 2, 3]\n```"
        result = extract_json_from_response(text)
        assert result is not None
        assert json.loads(result) == [1, 2, 3]

    def test_extract_analysis_fallback(self):
        """When no tags, falls back to general extraction."""
        text = '```json\n{"fallback": true}\n```'
        result = extract_json_from_analysis_output(text)
        assert result is not None
        assert json.loads(result) == {"fallback": True}


# ══════════════════════════════════════════════════════════════════════════════
# Category 3: Aggregation
# ══════════════════════════════════════════════════════════════════════════════

class TestAggregation:
    def _make_grader(self):
        """Create a PointwiseGrader without a real client (for aggregation tests only)."""
        grader = object.__new__(PointwiseGrader)
        return grader

    def test_hierarchical_scores_basic(self):
        """Verify weighted calculation with known inputs."""
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
            "coverage": [
                {"criterion": "c1", "weight": 0.6},
                {"criterion": "c2", "weight": 0.4},
            ],
            "insight": [
                {"criterion": "c1", "weight": 1.0},
            ],
        }
        dim_weights = {"coverage": 0.5, "insight": 0.5}

        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)

        # coverage: 8.0*0.6 + 6.0*0.4 = 7.2
        # insight: 7.0*1.0 = 7.0
        # total: 7.2*0.5 + 7.0*0.5 = 7.1
        assert abs(result["total_weighted_score"] - 7.1) < 0.01
        assert abs(result["coverage_score"] - 7.2) < 0.01
        assert abs(result["insight_score"] - 7.0) < 0.01

    def test_hierarchical_scores_failed_dimension(self):
        """Failed dimensions should be excluded with weight redistribution."""
        grader = self._make_grader()

        scores = {
            "coverage": [
                {"criterion": "c1", "analysis": "...", "report_score_0_to_10": 8.0},
            ],
            # "insight" missing (failed)
        }
        all_criteria = {
            "coverage": [{"criterion": "c1", "weight": 1.0}],
            "insight": [{"criterion": "c1", "weight": 1.0}],
        }
        dim_weights = {"coverage": 0.6, "insight": 0.4}

        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)

        # Only coverage scored. Rescaled weight = 0.6/0.6 = 1.0
        # total = 8.0 * 1.0 = 8.0
        assert abs(result["total_weighted_score"] - 8.0) < 0.01

    def test_hierarchical_scores_all_fail(self):
        """If all dimensions fail, total = 0.0."""
        grader = self._make_grader()
        result = grader.calculate_hierarchical_scores(
            scores={},
            all_criteria={"coverage": [{"criterion": "c1", "weight": 1.0}]},
            dimension_weights={"coverage": 1.0},
        )
        assert result["total_weighted_score"] == 0.0

    def test_single_dimension_single_criterion(self):
        """Simplest possible case."""
        grader = self._make_grader()
        scores = {
            "clarity": [
                {"criterion": "readable", "analysis": "ok", "report_score_0_to_10": 5.5},
            ],
        }
        all_criteria = {"clarity": [{"criterion": "readable", "weight": 1.0}]}
        dim_weights = {"clarity": 1.0}

        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)
        assert abs(result["total_weighted_score"] - 5.5) < 0.01
        assert abs(result["clarity_score"] - 5.5) < 0.01

    def test_empty_dim_scores_treated_as_failed(self):
        """A dimension with empty scores list is treated as failed."""
        grader = self._make_grader()
        scores = {"coverage": []}
        all_criteria = {"coverage": [{"criterion": "c1", "weight": 1.0}]}
        dim_weights = {"coverage": 1.0}

        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)
        assert result["coverage_score"] is None
        assert result["total_weighted_score"] == 0.0

    def test_reward_range(self):
        """total_weighted_score should be in 0-10."""
        grader = self._make_grader()

        scores = {
            "coverage": [
                {"criterion": "c1", "analysis": "...", "report_score_0_to_10": 10.0},
            ],
        }
        all_criteria = {"coverage": [{"criterion": "c1", "weight": 1.0}]}
        dim_weights = {"coverage": 1.0}

        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)
        assert 0.0 <= result["total_weighted_score"] <= 10.0

    def test_multiple_dimensions_weighted(self):
        """Test with all 4 fixed dimensions and realistic weights."""
        grader = self._make_grader()

        scores = {
            "coverage": [{"criterion": "c", "analysis": ".", "report_score_0_to_10": 7.0}],
            "insight": [{"criterion": "c", "analysis": ".", "report_score_0_to_10": 8.0}],
            "instruction_following": [{"criterion": "c", "analysis": ".", "report_score_0_to_10": 9.0}],
            "clarity": [{"criterion": "c", "analysis": ".", "report_score_0_to_10": 6.0}],
        }
        all_criteria = {k: [{"criterion": "c", "weight": 1.0}] for k in scores}
        dim_weights = {
            "coverage": 0.3,
            "insight": 0.3,
            "instruction_following": 0.2,
            "clarity": 0.2,
        }

        result = grader.calculate_hierarchical_scores(scores, all_criteria, dim_weights)
        # 7*0.3 + 8*0.3 + 9*0.2 + 6*0.2 = 2.1 + 2.4 + 1.8 + 1.2 = 7.5
        assert abs(result["total_weighted_score"] - 7.5) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Category 4: Default Fallbacks
# ══════════════════════════════════════════════════════════════════════════════

class TestDefaultFallbacks:
    def _make_grader(self):
        grader = object.__new__(PointwiseGrader)
        return grader

    def test_default_weights_count(self):
        """4 fixed + N additional."""
        grader = self._make_grader()
        dims = [{"meta_dimension_name": "Market Timing"}]
        weights = grader._get_default_weights(dims)
        assert len(weights) == 5

    def test_default_weights_sum(self):
        """Sums to 1.0."""
        grader = self._make_grader()
        dims = [{"meta_dimension_name": "X"}, {"meta_dimension_name": "Y"}]
        weights = grader._get_default_weights(dims)
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_default_weights_no_additional(self):
        """With no additional dims, 4 equal weights."""
        grader = self._make_grader()
        weights = grader._get_default_weights([])
        assert len(weights) == 4
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_default_criteria_count(self):
        """3 default criteria."""
        grader = self._make_grader()
        criteria = grader._get_default_criteria("coverage", "Breadth and depth")
        assert len(criteria) == 3

    def test_default_criteria_weights_sum(self):
        """Weights sum to 1.0."""
        grader = self._make_grader()
        criteria = grader._get_default_criteria("insight", "Quality of analysis")
        total = sum(c["weight"] for c in criteria)
        assert abs(total - 1.0) < 0.001

    def test_default_criteria_fields(self):
        """Each criterion has criterion, explanation, weight."""
        grader = self._make_grader()
        criteria = grader._get_default_criteria("clarity")
        for c in criteria:
            assert "criterion" in c
            assert "explanation" in c
            assert "weight" in c

    def test_snake_case_normalization(self):
        """Dimension names normalize correctly."""
        grader = self._make_grader()
        dims = [{"meta_dimension_name": "Market Timing"}]
        weights = grader._get_default_weights(dims)
        assert "market_timing" in weights

        dims2 = [{"meta_dimension_name": "Data-Quality"}]
        weights2 = grader._get_default_weights(dims2)
        assert "data_quality" in weights2


# ══════════════════════════════════════════════════════════════════════════════
# Category 5: Environment Class (no LLM needed for most)
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvironmentClass:
    def _sample_task(self):
        return tasks[0]

    def test_environment_missing_openai_secret(self):
        with pytest.raises(ValueError, match="openai_api_key"):
            MiroEval(task_spec=self._sample_task(), secrets={"tavily_api_key": "tvly-xxx"})

    def test_environment_missing_tavily_secret(self):
        with pytest.raises(ValueError, match="tavily_api_key"):
            MiroEval(task_spec=self._sample_task(), secrets={"openai_api_key": "sk-xxx"})

    def test_environment_missing_all_secrets(self):
        with pytest.raises(ValueError):
            MiroEval(task_spec=self._sample_task(), secrets={})

    def test_environment_init_valid(self):
        """Environment initializes with valid secrets."""
        env = MiroEval(
            task_spec=self._sample_task(),
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        assert env.validated.id == self._sample_task()["id"]

    @pytest.mark.asyncio
    async def test_prompt_contains_query(self):
        """get_prompt returns TextBlock containing the query."""
        env = MiroEval(
            task_spec=self._sample_task(),
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        blocks = await env.get_prompt()
        assert len(blocks) == 1
        assert self._sample_task()["query"][:50] in blocks[0].text

    @pytest.mark.asyncio
    async def test_prompt_bilingual_chinese(self):
        """Chinese query gets Chinese instructions."""
        zh_task = None
        for t in tasks:
            if t["language"] == "zh":
                zh_task = t
                break
        assert zh_task is not None, "No Chinese task found"

        env = MiroEval(
            task_spec=zh_task,
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        blocks = await env.get_prompt()
        assert "可用工具" in blocks[0].text

    @pytest.mark.asyncio
    async def test_prompt_bilingual_english(self):
        """English query gets English instructions."""
        en_task = None
        for t in tasks:
            if t["language"] == "en":
                en_task = t
                break
        assert en_task is not None, "No English task found"

        env = MiroEval(
            task_spec=en_task,
            secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
        )
        blocks = await env.get_prompt()
        assert "Available Tools" in blocks[0].text

    def test_task_spec_validation(self):
        """Invalid task spec raises validation error."""
        with pytest.raises(Exception):
            MiroEval(
                task_spec={"id": "1"},  # missing required fields
                secrets={"openai_api_key": "sk-test", "tavily_api_key": "tvly-test"},
            )


# ══════════════════════════════════════════════════════════════════════════════
# Category 6: Integration Tests (require real API keys)
# ══════════════════════════════════════════════════════════════════════════════

def _get_secrets():
    """Load real API keys for integration tests."""
    from dotenv import load_dotenv
    load_dotenv("../.env")
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not openai_key or not tavily_key:
        pytest.skip("OPENAI_API_KEY and TAVILY_API_KEY required for integration tests")
    return {"openai_api_key": openai_key, "tavily_api_key": tavily_key}


# Pick a short English task for integration tests
_en_tasks = [t for t in tasks if t["language"] == "en"]
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
        assert len(result.blocks) > 0
        assert "search results" in result.blocks[0].text.lower() or "no search" in result.blocks[0].text.lower()

    @pytest.mark.asyncio
    async def test_fetch_url_returns_content(self):
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)
        result = await env.fetch_url(FetchUrlInput(url="https://en.wikipedia.org/wiki/Python_(programming_language)"))
        assert result.finished is False
        assert result.reward == 0.0
        assert len(result.blocks) > 0

    @pytest.mark.asyncio
    async def test_good_vs_bad_report(self):
        """A substantive report should score higher than gibberish."""
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)

        import openai as oai
        client = oai.AsyncClient(api_key=secrets["openai_api_key"])
        response = await client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"Write a brief 500-word research report on: {_integration_task['query'][:500]}",
            }],
        )
        good_report = response.choices[0].message.content or ""

        good_result = await env.submit_report(ReportInput(report=good_report))
        bad_result = await env.submit_report(ReportInput(report="This is garbage. Nothing relevant here. Random words."))

        assert good_result.reward > bad_result.reward
        assert good_result.finished is True
        assert bad_result.finished is True

    @pytest.mark.asyncio
    async def test_full_pipeline_structure(self):
        """Verify grading pipeline returns all expected fields."""
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)

        result = await env.submit_report(
            ReportInput(report="A moderate quality report discussing the topic with some analysis and depth.")
        )

        assert result.finished is True
        assert result.metadata is not None
        assert "total_score" in result.metadata
        assert "dimension_scores" in result.metadata
        assert "dimension_weights" in result.metadata
        assert "additional_dimensions" in result.metadata

    @pytest.mark.asyncio
    async def test_reward_in_range(self):
        """Reward should be in [0, 10]."""
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)

        result = await env.submit_report(
            ReportInput(report="A research report with basic coverage of the topic.")
        )

        assert 0.0 <= result.reward <= 10.0
        assert 0.0 <= result.metadata["total_score"] <= 10.0

    @pytest.mark.asyncio
    async def test_empty_report_low_score(self):
        """Empty report should score very low."""
        secrets = _get_secrets()
        env = MiroEval(task_spec=_integration_task, secrets=secrets)

        result = await env.submit_report(ReportInput(report=""))
        assert result.reward < 3.0
        assert result.finished is True
