# MiroEval

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/MiroEval)

## Description

**MiroEval** is an environment for evaluating deep research agents on their ability to conduct web-based research and produce comprehensive reports. It is based on the [MiroEval benchmark](https://arxiv.org/abs/2603.28407) by MiroMindAI, which evaluates multimodal deep research systems across synthesis quality, factual correctness, and process quality dimensions.

This environment implements the **point-wise quality evaluation** pipeline from the original benchmark. Agents receive a research query, use web search and URL fetching tools to gather information, then submit a comprehensive research report. The report is graded using a 5-stage hierarchical LLM pipeline (GPT-5.1) that dynamically generates query-specific evaluation dimensions, criteria, and scores.

## Capabilities

- Conducting web-based research across 12 domains (science, finance, medical, engineering, etc.)
- Multi-turn information gathering via web search and URL extraction
- Synthesizing research findings into comprehensive reports
- Bilingual research (English and Chinese queries)

## Compute Requirements

MiroEval does not require a sandbox. Compute is primarily LLM API calls for grading (~12-16 GPT-5.1 calls per report evaluation).

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Tasks

There is one split: **text** (70 tasks). Each task presents a research query drawn from real user patterns across 12 domains. Queries span diverse topics including quantitative finance, DevOps, drug discovery, urban planning, materials science, and more. Approximately 50 queries are in English and 20 in Chinese.

## Reward Structure

Reports are graded using the original MiroEval 5-stage hierarchical point-wise quality evaluation pipeline:

1. **Dimension Generation**: 1-3 query-specific evaluation dimensions are dynamically generated beyond the 4 fixed dimensions (Coverage, Insight, Instruction Following, Clarity)
2. **Weight Assignment**: Normalized weights assigned to all dimensions (query-specific ≤ 20% total)
3. **Criteria Generation**: 1-10 specific evaluation criteria generated per dimension
4. **Scoring**: Each criterion scored 0-10 by GPT-5.1
5. **Aggregation**: Hierarchical weighted sum produces final score

$$\text{Reward} = \sum_{d} w_d \cdot \left(\sum_{c \in d} w_c \cdot s_c\right)$$

where $w_d$ are dimension weights, $w_c$ are criterion weights within dimension $d$, and $s_c$ are criterion scores (0-10).

Rewards range from 0.0 to 10.0. We use GPT-5.1 as the grading model to match the original MiroEval evaluation methodology.

## Data

Queries are sourced from the [MiroEval benchmark](https://github.com/MiroMindAI/MiroEval) (70 text queries from `mirobench_text.json`). Data files are stored on the OpenReward platform.

## Tools

Agents are given three tools:

- `web_search`: Search the web for information. Returns titles, URLs, and snippets. Uses Tavily API with advanced search depth. Can be called multiple times.
- `fetch_url`: Fetch full text content from a specific URL. Useful for reading detailed information from search results. Content truncated at 12,000 characters.
- `submit_report`: Submit the final research report for evaluation. Triggers the 5-stage grading pipeline. Returns the quality score and dimension breakdown. This tool ends the task.

## Time Horizon

MiroEval is a multi-turn environment. The agent conducts research over multiple turns using web search and URL fetching, then submits a report. A typical task involves 5-20 tool calls across 3-6 turns.

## Other Environment Requirements

MiroEval requires two API keys passed as secrets:

- `openai_api_key`: For GPT-5.1 grading of submitted reports
- `tavily_api_key`: For web search and URL content extraction via the Tavily API

## Safety

Agents in MiroEval are asked to research topics and produce reports. The environment does not present direct safety risks, as agents interact with the web through controlled search and fetch tools. Research topics span academic and professional domains. Web access is mediated through the Tavily API, which provides content extraction without full browser execution.

## Citations

```bibtex
@article{ye2026miroeval,
  title={MiroEval: Benchmarking Multimodal Deep Research Agents in Process and Outcome},
  author={Ye, Fangda and Hu, Yuxin and Zhu, Pengxiang and Li, Yibo and Jin, Ziqi and Xiao, Yao and Wang, Yibo and Wang, Lei and Zhang, Zhen and Wang, Lu and Deng, Yue and Wang, Bin and Zhang, Yifan and Su, Liangcai and Wang, Xinyu and Zhao, He and Wei, Chen and Ren, Qiang and Hooi, Bryan and Bo, An and Yan, Shuicheng and Bing, Lidong},
  journal={arXiv preprint arXiv:2603.28407},
  year={2026}
}
```
