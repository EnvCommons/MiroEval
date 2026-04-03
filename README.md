# MiroEval

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/MiroEval)

## Description

**MiroEval** is an ORS environment for evaluating deep research agents on their ability to conduct web-based research and produce comprehensive reports. It is based on the [MiroEval benchmark](https://arxiv.org/abs/2603.28407) by MiroMindAI, which evaluates multimodal deep research systems across synthesis quality, factual correctness, and process quality dimensions.

This environment implements the **point-wise quality evaluation** pipeline from the original benchmark. Agents receive a research query — optionally with image, PDF, or document attachments — use web search and URL fetching tools to gather information, and submit a comprehensive research report. The report is graded using a 5-stage hierarchical LLM pipeline (GPT-5.1) that dynamically generates query-specific evaluation dimensions, criteria, and scores. For multimodal tasks, grading includes attachment-grounded key facts extraction and grounding-aware scoring.

## Capabilities

- Conducting web-based research across 12 domains (science, finance, medical, engineering, etc.)
- Multi-turn information gathering via web search and URL extraction
- Analyzing multimodal attachments (images, PDFs, documents)
- Synthesizing research findings into comprehensive reports
- Bilingual research (English and Chinese queries)

## Compute Requirements

MiroEval does not require a sandbox. Compute is primarily LLM API calls for grading (~12-16 GPT-5.1 calls per text report, additional calls for multimodal key facts extraction).

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Tasks

There is one split: **test** (100 tasks).

- **70 text-only tasks**: Research queries across 12 domains drawn from real user patterns.
- **30 multimodal tasks**: Research queries with image, PDF, or document attachments (1-10 files per task). Categories include single image analysis, document synthesis, and multi-document comparison.

Approximately 75 queries are in Chinese and 25 in English.

## Reward Structure

Reports are graded using the original MiroEval 5-stage hierarchical point-wise quality evaluation pipeline:

1. **Key Facts Extraction** (multimodal only): Extract 5-15 verifiable facts from attachments
2. **Dimension Generation**: 1-3 query-specific evaluation dimensions dynamically generated beyond the 4 fixed dimensions (Coverage, Insight, Instruction Following, Clarity). For multimodal tasks, these are "Grounding & Task-specific Expertise" composite dimensions.
3. **Weight Assignment**: Normalized weights assigned to all dimensions (query-specific ≤ 20% total)
4. **Criteria Generation**: 1-10 specific evaluation criteria generated per dimension (fact-anchored for multimodal grounding dimensions)
5. **Scoring**: Each criterion scored 0-10 by GPT-5.1 (with key-facts cross-referencing for multimodal)
6. **Aggregation**: Hierarchical weighted sum produces final score

$$\text{Reward} = \sum_{d} w_d \cdot \left(\sum_{c \in d} w_c \cdot s_c\right)$$

Rewards range from 0.0 to 10.0. We use GPT-5.1 as the grading model to match the original MiroEval evaluation methodology.

## Data

Queries are sourced from the [MiroEval benchmark](https://github.com/MiroMindAI/MiroEval). Text queries from `mirobench_text.json` (70 tasks) and multimodal queries from `mirobench_multimodal.json` (30 tasks) with attachment files. Data files are stored on the OpenReward platform.

## Tools

Agents are given four tools:

- `web_search`: Search the web for information. Returns titles, URLs, and snippets. Uses Tavily API with advanced search depth.
- `fetch_url`: Fetch full text content from a URL. Content truncated at 12,000 characters.
- `view_attachment`: View an attachment file associated with the task. Images are returned as visual content. PDFs are rendered page-by-page as images. Text files are returned as text.
- `submit_report`: Submit the final research report for evaluation. Triggers the grading pipeline. Returns the quality score and dimension breakdown. Ends the task.

## Time Horizon

MiroEval is a multi-turn environment. The agent conducts research over multiple turns using web search, URL fetching, and attachment viewing, then submits a report.

## Other Environment Requirements

MiroEval requires two API keys passed as secrets:

- `openai_api_key`: For GPT-5.1 grading of submitted reports
- `tavily_api_key`: For web search and URL content extraction via the Tavily API

## Safety

Agents in MiroEval are asked to research topics and produce reports. The environment does not present direct safety risks. Web access is mediated through the Tavily API, which provides content extraction without full browser execution.

## Citations

```bibtex
@article{ye2026miroeval,
  title={MiroEval: Benchmarking Multimodal Deep Research Agents in Process and Outcome},
  author={Ye, Fangda and Hu, Yuxin and Zhu, Pengxiang and Li, Yibo and Jin, Ziqi and Xiao, Yao and Wang, Yibo and Wang, Lei and Zhang, Zhen and Wang, Lu and Deng, Yue and Wang, Bin and Zhang, Yifan and Su, Liangcai and Wang, Xinyu and Zhao, He and Wei, Chen and Ren, Qiang and Hooi, Bryan and Bo, An and Yan, Shuicheng and Bing, Lidong},
  journal={arXiv preprint arXiv:2603.28407},
  year={2026}
}
```
