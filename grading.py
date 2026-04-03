"""
MiroEval Point-wise Quality Grading Pipeline.

Implements the 5-stage hierarchical evaluation from the original MiroEval:
https://github.com/MiroMindAI/MiroEval/blob/main/point_quality/deepresearcharena/evaluator/pointwise_core.py

Uses GPT-5.1 as the judge model (faithful to original configuration).
Supports both text-only and multimodal (attachment-aware) grading.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import openai

from prompts import (
    KEY_FACTS_EXTRACTION_PROMPT,
    DIMENSION_GENERATION_PROMPT,
    DIMENSION_GENERATION_WITH_ATTACHMENT_PROMPT,
    WEIGHT_GENERATION_PROMPT,
    CRITERIA_GENERATION_PROMPT,
    CRITERIA_GENERATION_WITH_KEY_FACTS_PROMPT,
    SCORING_PROMPT,
    SCORING_WITH_KEY_FACTS_PROMPT,
)
from utils import extract_json_from_response, extract_json_from_analysis_output

logger = logging.getLogger(__name__)

# Fixed dimensions (always included) — matches original MiroEval
FIXED_DIMENSIONS = {
    "coverage": "Breadth, depth, and relevance of coverage",
    "insight": "Depth, originality, logic, and value of analysis",
    "instruction_following": "Accuracy in meeting all requirements",
    "clarity": "Readability, fluency, structure, and ease of understanding",
}


class PointwiseGrader:
    """
    Implements the MiroEval 5-stage point-wise quality grading pipeline.

    Stage 0 (multimodal only): Extract key facts from attachments
    Stage 1: Generate 1-3 query-specific dimensions
    Stage 2: Assign normalized weights (query-specific <= 20%)
    Stage 3: Generate criteria per dimension with weights
    Stage 4: Score report on each criterion (0-10)
    Aggregation: Hierarchical weighted sum -> total score 0-10
    """

    def __init__(self, client: openai.AsyncClient, model: str = "gpt-5.1"):
        self.client = client
        self.model = model

    async def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call LLM with retry logic and exponential backoff for API errors."""
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=8192,
                    temperature=0.1,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"LLM call failed (attempt {attempt+1}): {e}, retrying in {wait}s")
                    await asyncio.sleep(wait)
        raise last_error  # type: ignore

    async def _call_llm_with_image(self, prompt: str, image_base64: str, mime_type: str, max_retries: int = 3) -> str:
        """Call LLM with an image (vision API). Used for image-based key facts extraction."""
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_base64}",
                                },
                            },
                        ],
                    }],
                    max_completion_tokens=4096,
                    temperature=0.1,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Vision LLM call failed (attempt {attempt+1}): {e}, retrying in {wait}s")
                    await asyncio.sleep(wait)
        raise last_error  # type: ignore

    # ── Stage 0: Extract key facts from attachments (multimodal only) ──

    async def extract_key_facts(
        self, task_prompt: str, attachment_contents: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Extract key facts from attachment contents. Each attachment is either:
        - {"text": "..."} for text/PDF content
        - {"image_base64": "...", "mime_type": "..."} for images

        Faithful to original: each attachment processed independently, results merged.
        Returns list of {"fact": "...", "importance": "..."}.
        """
        all_key_facts: List[Dict[str, str]] = []

        for i, content in enumerate(attachment_contents):
            logger.info(f"Extracting key facts from attachment {i+1}/{len(attachment_contents)}")
            try:
                if "text" in content:
                    # Text-based attachment (PDF text, text files)
                    formatted_prompt = KEY_FACTS_EXTRACTION_PROMPT.format(
                        task_prompt=task_prompt,
                        attachment_content=content["text"],
                    )
                    response = await self._call_llm(formatted_prompt)
                elif "image_base64" in content:
                    # Image attachment — use vision API
                    formatted_prompt = KEY_FACTS_EXTRACTION_PROMPT.format(
                        task_prompt=task_prompt,
                        attachment_content="[See attached image]",
                    )
                    response = await self._call_llm_with_image(
                        formatted_prompt,
                        content["image_base64"],
                        content.get("mime_type", "image/jpeg"),
                    )
                else:
                    continue

                json_str = extract_json_from_response(response)
                if json_str:
                    facts = json.loads(json_str)
                    if isinstance(facts, list):
                        all_key_facts.extend(facts)
                        logger.info(f"Extracted {len(facts)} key facts from attachment {i+1}")
            except Exception as e:
                logger.error(f"Failed to extract key facts from attachment {i+1}: {e}")

        logger.info(f"Total key facts extracted: {len(all_key_facts)}")
        return all_key_facts

    # ── Stage 1: Generate query-specific dimensions ──

    async def generate_query_dimensions(
        self, task_prompt: str, key_facts: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Generate 1-3 query-specific evaluation dimensions.
        Uses attachment-aware prompt when key_facts are provided.
        """
        if key_facts:
            key_facts_json = json.dumps(key_facts, ensure_ascii=False, indent=2)
            formatted_prompt = DIMENSION_GENERATION_WITH_ATTACHMENT_PROMPT.format(
                task_prompt=task_prompt,
                key_facts_json=key_facts_json,
            )
        else:
            formatted_prompt = DIMENSION_GENERATION_PROMPT.format(task_prompt=task_prompt)

        try:
            response = await self._call_llm(formatted_prompt)
            json_str = extract_json_from_response(response)
            if json_str:
                dimensions = json.loads(json_str)
                if isinstance(dimensions, list) and len(dimensions) > 0:
                    return dimensions[:3]
        except Exception as e:
            logger.error(f"Dimension generation failed: {e}")

        logger.warning("Dimension generation failed, using empty additional dimensions")
        return []

    # ── Stage 2: Generate hierarchical weights ──

    async def generate_hierarchical_weights(
        self, task_prompt: str, additional_dimensions: List[Dict[str, str]]
    ) -> Dict[str, float]:
        """
        Assign weights to all dimensions summing to 1.0.
        Query-specific dimensions capped at 20% total (via prompt instruction).
        Falls back to equal weights on failure.
        """
        additional_dimensions_json = json.dumps(additional_dimensions, ensure_ascii=False, indent=2)
        formatted_prompt = WEIGHT_GENERATION_PROMPT.format(
            task_prompt=task_prompt,
            additional_dimensions_json=additional_dimensions_json,
        )

        try:
            response = await self._call_llm(formatted_prompt)
            weights_json = extract_json_from_analysis_output(response)
            if weights_json:
                weights = json.loads(weights_json)
                total_weight = sum(weights.values())
                if total_weight > 0:
                    weights = {k: v / total_weight for k, v in weights.items()}
                normalized = {}
                for k, v in weights.items():
                    key = k.lower().replace(" ", "_").replace("-", "_")
                    normalized[key] = v
                return normalized
        except Exception as e:
            logger.error(f"Weight generation failed: {e}")

        return self._get_default_weights(additional_dimensions)

    def _get_default_weights(self, additional_dimensions: List[Dict]) -> Dict[str, float]:
        """Equal weights fallback."""
        num_dims = 4 + len(additional_dimensions)
        w = 1.0 / num_dims
        weights = {k: w for k in FIXED_DIMENSIONS}
        for dim in additional_dimensions:
            key = dim.get("meta_dimension_name", "").lower().replace(" ", "_").replace("-", "_")
            if key:
                weights[key] = w
        return weights

    # ── Stage 3: Generate criteria per dimension ──

    async def generate_dimension_criteria(
        self,
        task_prompt: str,
        dimension_name: str,
        all_dims_with_definition: Dict[str, str],
        key_facts: Optional[List[Dict[str, str]]] = None,
        is_dynamic_dim: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate evaluation criteria for a single dimension.
        Uses key-facts-aware prompt for dynamic dimensions with attachments.
        """
        meta_dims_str = "\n".join(
            f"- **{dim}**: {defn}" for dim, defn in all_dims_with_definition.items()
        )

        if key_facts and is_dynamic_dim:
            key_facts_json = json.dumps(key_facts, ensure_ascii=False, indent=2)
            formatted_prompt = CRITERIA_GENERATION_WITH_KEY_FACTS_PROMPT.format(
                task_prompt=task_prompt,
                num_dimensions=len(all_dims_with_definition),
                meta_dimensions=meta_dims_str,
                dimension_name=dimension_name,
                key_facts_json=key_facts_json,
            )
        else:
            formatted_prompt = CRITERIA_GENERATION_PROMPT.format(
                task_prompt=task_prompt,
                num_dimensions=len(all_dims_with_definition),
                meta_dimensions=meta_dims_str,
                dimension_name=dimension_name,
            )

        for attempt in range(2):
            try:
                response = await self._call_llm(formatted_prompt)
                json_str = extract_json_from_analysis_output(response)
                if json_str:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        total_w = sum(item.get("weight", 0) for item in parsed)
                        if total_w > 0:
                            for item in parsed:
                                item["weight"] = item.get("weight", 0) / total_w
                        return parsed
            except Exception as e:
                logger.warning(f"Criteria generation attempt {attempt+1} failed for '{dimension_name}': {e}")

        definition = all_dims_with_definition.get(dimension_name, f"Quality of {dimension_name}")
        return self._get_default_criteria(dimension_name, definition)

    def _get_default_criteria(self, dimension_name: str, definition: str = "") -> List[Dict[str, Any]]:
        """Get default criteria if generation failed. Matches original fallback."""
        if not definition:
            definition = f"Quality of {dimension_name}"
        return [
            {
                "criterion": f"Core quality of {dimension_name}",
                "explanation": f"How well the report addresses the primary aspects of: {definition}",
                "weight": 0.5,
            },
            {
                "criterion": f"Depth and specificity of {dimension_name}",
                "explanation": f"Whether the report provides detailed, specific analysis rather than superficial coverage for: {definition}",
                "weight": 0.3,
            },
            {
                "criterion": f"Relevance and task-alignment of {dimension_name}",
                "explanation": f"Whether the report's treatment of this dimension is well-aligned with the task requirements: {definition}",
                "weight": 0.2,
            },
        ]

    # ── Stage 4: Score each dimension ──

    async def score_single_dimension(
        self,
        task_prompt: str,
        report: str,
        dim_name: str,
        criteria_list: List[Dict[str, Any]],
        key_facts: Optional[List[Dict[str, str]]] = None,
        is_dynamic_dim: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Score a report on a single dimension.
        Uses key-facts-aware scoring for dynamic dimensions with attachments.
        """
        criteria_for_prompt = [
            {"criterion": c["criterion"], "explanation": c["explanation"]}
            for c in criteria_list
        ]
        criteria_json = json.dumps(criteria_for_prompt, ensure_ascii=False, indent=2)

        if key_facts and is_dynamic_dim:
            key_facts_json = json.dumps(key_facts, ensure_ascii=False, indent=2)
            formatted_prompt = SCORING_WITH_KEY_FACTS_PROMPT.format(
                task_prompt=task_prompt,
                report=report,
                criteria_of_one_dimension_json=criteria_json,
                key_facts_json=key_facts_json,
            )
        else:
            formatted_prompt = SCORING_PROMPT.format(
                task_prompt=task_prompt,
                report=report,
                criteria_of_one_dimension_json=criteria_json,
            )

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = await self._call_llm(formatted_prompt)
                json_str = extract_json_from_analysis_output(response)
                if not json_str:
                    raise ValueError("No JSON found in scoring response")
                scored = json.loads(json_str)

                resp_map = {item["criterion"]: item for item in scored}
                dimension_scores = []
                for c in criteria_list:
                    name = c["criterion"]
                    item = resp_map[name]
                    dimension_scores.append({
                        "criterion": name,
                        "analysis": item["analysis"],
                        "report_score_0_to_10": float(item["report_score_0_to_10"]),
                    })
                return dim_name, dimension_scores

            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for '{dim_name}': {e}")

        raise Exception(f"Scoring failed for '{dim_name}' after {max_retries} attempts: {last_error}")

    # ── Aggregation ──

    def calculate_hierarchical_scores(
        self,
        scores: Dict[str, List[Dict]],
        all_criteria: Dict[str, List[Dict]],
        dimension_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Calculate final hierarchical weighted scores.
        Faithful to original MiroEval calculate_hierarchical_scores.
        """
        final_scores: Dict[str, Any] = {}
        dim_score_map: Dict[str, float] = {}
        failed_dims: list[str] = []

        for dim_name, criteria_list in all_criteria.items():
            if dim_name not in scores:
                failed_dims.append(dim_name)
                continue

            dim_scores = scores[dim_name]
            if not isinstance(dim_scores, list) or len(dim_scores) == 0:
                failed_dims.append(dim_name)
                final_scores[f"{dim_name}_score"] = None
                continue

            weighted_dim_score = 0.0
            total_criterion_weight = 0.0

            for i, criterion_data in enumerate(criteria_list):
                if i < len(dim_scores):
                    score_item = dim_scores[i]
                    if (
                        isinstance(score_item, dict)
                        and criterion_data["criterion"] == score_item["criterion"]
                        and "report_score_0_to_10" in score_item
                    ):
                        score_value = score_item["report_score_0_to_10"]
                        criterion_weight = criterion_data["weight"]
                        weighted_dim_score += float(score_value) * float(criterion_weight)
                        total_criterion_weight += float(criterion_weight)

            if total_criterion_weight > 0:
                final_dim_score = weighted_dim_score / total_criterion_weight
            else:
                failed_dims.append(dim_name)
                final_scores[f"{dim_name}_score"] = None
                continue

            final_scores[f"{dim_name}_score"] = final_dim_score
            dim_score_map[dim_name] = final_dim_score

        if failed_dims:
            logger.warning(f"Dimensions with failed scoring (excluded): {failed_dims}")

        successful_weight_sum = sum(
            dimension_weights.get(d, 0) for d in dim_score_map
        )

        total_weighted_score = 0.0
        if successful_weight_sum > 0:
            for dim_name, dim_score in dim_score_map.items():
                rescaled_weight = dimension_weights.get(dim_name, 0) / successful_weight_sum
                total_weighted_score += dim_score * rescaled_weight

        final_scores["total_weighted_score"] = float(total_weighted_score)
        return final_scores

    # ── Full Pipeline ──

    async def grade_report(
        self,
        task_prompt: str,
        report: str,
        attachment_contents: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full grading pipeline.

        Args:
            task_prompt: The research query
            report: The agent's submitted report
            attachment_contents: Optional list of attachment content dicts for multimodal tasks.
                Each is {"text": "..."} or {"image_base64": "...", "mime_type": "..."}.
        """
        # Stage 0: Extract key facts (multimodal only)
        key_facts: Optional[List[Dict[str, str]]] = None
        if attachment_contents:
            key_facts = await self.extract_key_facts(task_prompt, attachment_contents)
            if not key_facts:
                key_facts = None  # Treat as text-only if extraction failed

        # Stage 1: Generate query-specific dimensions
        additional_dimensions = await self.generate_query_dimensions(task_prompt, key_facts)

        # Build complete dimension map; track which are dynamic
        all_dims: Dict[str, str] = dict(FIXED_DIMENSIONS)
        dynamic_dim_names: set[str] = set()
        for item in additional_dimensions:
            key = item["meta_dimension_name"].lower().replace(" ", "_").replace("-", "_")
            all_dims[key] = item["definition"]
            dynamic_dim_names.add(key)

        # Stage 2: Generate weights
        dimension_weights = await self.generate_hierarchical_weights(
            task_prompt, additional_dimensions
        )

        # Stage 3: Generate criteria for ALL dimensions concurrently
        criteria_tasks = [
            self.generate_dimension_criteria(
                task_prompt, dim_name, all_dims,
                key_facts=key_facts,
                is_dynamic_dim=(dim_name in dynamic_dim_names),
            )
            for dim_name in all_dims
        ]
        criteria_results = await asyncio.gather(*criteria_tasks)
        all_criteria = dict(zip(all_dims.keys(), criteria_results))

        # Stage 4: Score ALL dimensions concurrently
        scoring_tasks = [
            self.score_single_dimension(
                task_prompt, report, dim_name, criteria,
                key_facts=key_facts,
                is_dynamic_dim=(dim_name in dynamic_dim_names),
            )
            for dim_name, criteria in all_criteria.items()
        ]
        scoring_results = await asyncio.gather(*scoring_tasks, return_exceptions=True)

        scores: Dict[str, List[Dict]] = {}
        for result in scoring_results:
            if isinstance(result, Exception):
                logger.error(f"Dimension scoring failed: {result}")
            else:
                dim_name, dim_scores = result
                scores[dim_name] = dim_scores

        # Aggregation
        final = self.calculate_hierarchical_scores(scores, all_criteria, dimension_weights)
        total_score = final.get("total_weighted_score", 0.0)

        dim_scores_out = {}
        for key, val in final.items():
            if key.endswith("_score") and key != "total_weighted_score":
                dim_scores_out[key] = val

        return {
            "total_score": total_score,
            "dimension_scores": dim_scores_out,
            "dimension_weights": dimension_weights,
            "all_criteria": {
                k: [{"criterion": c["criterion"], "weight": c["weight"]} for c in v]
                for k, v in all_criteria.items()
            },
            "raw_scores": scores,
            "additional_dimensions": additional_dimensions,
            "key_facts": key_facts,
        }
