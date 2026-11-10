# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""LiveBench's scorer, grading coding answers in the sandbox and AMPS_Hard without an OpenAI tiebreaker.

Adapted from ``inspect_evals/livebench/scorer.py`` (https://github.com/UKGovernmentBEIS/inspect_evals, MIT, Copyright
(c) 2024 UK AI Security Institute; see ``LICENSE-inspect_evals`` in this folder), whose dispatch follows LiveBench's
``gen_ground_truth_judgment.py``. Changed: coding answers run in the sandbox (``lcb_sandbox``), AMPS_Hard uses the
local ``amps_hard`` copy, an errored sample without an answer scores 0, and the sample metadata is not copied into
the score.
"""

import re

from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_evals.livebench.utils import ensure_nltk_resource, grouped_mean


@scorer(metrics=[grouped_mean("category", "livebench")])
def decide_scorer() -> Scorer:  # noqa: C901  kept as upstream
    """Grade each sample with the LiveBench function for its task."""
    from inspect_evals.livebench.instruction_following import (
        instruction_following_process_results,
    )
    from livebench.process_results.data_analysis.cta.utils import (  # type: ignore
        cta_process_results,
    )
    from livebench.process_results.data_analysis.tablejoin.utils import (  # type: ignore
        joinmap_process_results,
    )
    from livebench.process_results.data_analysis.tablereformat.utils import (  # type: ignore
        table_process_results,
    )
    from livebench.process_results.math.math_competitions.utils import (  # type: ignore
        aime_process_results,
        mathcontest_process_results,
    )
    from livebench.process_results.math.olympiad.utils import (  # type: ignore
        proof_rearrangement_process_results,
    )
    from livebench.process_results.reasoning.house_traversal.utils import (  # type: ignore
        house_traversal_process_results,
    )
    from livebench.process_results.reasoning.spatial.utils import (  # type: ignore
        spatial_process_results,
    )
    from livebench.process_results.reasoning.web_of_lies_v2.utils import (  # type: ignore
        web_of_lies_process_results,
    )
    from livebench.process_results.reasoning.zebra_puzzle.utils import (  # type: ignore
        get_zebra_puzzle_evaluator,
    )
    from livebench.process_results.writing.connections.utils import (  # type: ignore
        get_connections_puzzle_evaluator,
    )
    from livebench.process_results.writing.plot_unscrambling.utils import (  # type: ignore
        plot_unscrambling_process_results,
    )
    from livebench.process_results.writing.typos.utils import (  # type: ignore
        typos_process_results,
    )

    # Local copy: upstream's OpenAI tiebreaker can't run on offline nodes.
    from .amps_hard import amps_hard_process_results
    from .lcb_sandbox import lcb_generation_sandboxed

    scorer_map = {
        "web_of_lies_v2": web_of_lies_process_results,  # Reasoning
        "house_traversal": house_traversal_process_results,  # Reasoning
        "spatial": spatial_process_results,  # Reasoning
        "cta": cta_process_results,  # Data Analysis
        "typos": typos_process_results,  # Language
        "plot_unscrambling": plot_unscrambling_process_results,  # Language
    }

    async def score(state: TaskState, target: Target) -> Score:  # noqa: C901  kept as upstream
        ground_truth = target.text
        # The model's (single) answer; upstream reads it as messages[2], after the system and user messages
        answers = [m for m in state.messages if m.role == "assistant"]
        if not answers or not answers[0].text.strip():  # e.g. the sample errored
            return Score(value=0, answer="", explanation="no answer")
        llm_answer = answers[0].text
        llm_answer = re.sub(r"<think>.*?<\/think>", "", llm_answer, flags=re.DOTALL)
        # An unterminated <think> means generation stopped inside the reasoning
        # block, so what follows is truncated reasoning and not an answer.
        llm_answer = re.sub(r"<think>.*\Z", "", llm_answer, flags=re.DOTALL)
        prompt_text = state.input_text
        # TODO: check math scores (seem a bit off)

        # Logic taken from https://github.com/LiveBench/LiveBench/blob/main/livebench/gen_ground_truth_judgment.py
        task_or_subtask = state.metadata["task"]
        splits = task_or_subtask.split("_")
        category = state.metadata["category"]

        if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):  # Math
            value = mathcontest_process_results(ground_truth, llm_answer, prompt_text)
        elif splits[0] == "aime":  # Math
            value = aime_process_results(ground_truth, llm_answer)

        elif splits[0] in ["imo", "usamo"]:  # Math
            value = proof_rearrangement_process_results(ground_truth, llm_answer)

        elif task_or_subtask == "zebra_puzzle":  # Reasoning
            zebra_evaluator = get_zebra_puzzle_evaluator(state.metadata["livebench_release_date"].strftime("%Y-%m-%d"))
            value = zebra_evaluator(ground_truth, llm_answer)

        elif task_or_subtask == "tablejoin":  # Data Analysis
            value = joinmap_process_results(prompt_text, ground_truth, llm_answer)

        elif task_or_subtask == "tablereformat":  # Data Analysis
            value = table_process_results(prompt_text, ground_truth, llm_answer)

        elif "amps_hard" in task_or_subtask:  # Math
            # This could be amps_hard_geometric_mean, amps_hard_derivatives, or others
            value = amps_hard_process_results(ground_truth, llm_answer)

        elif task_or_subtask == "connections":  # Language
            connections_evaluator = get_connections_puzzle_evaluator(
                state.metadata["livebench_release_date"].strftime("%Y-%m-%d")
            )
            value = connections_evaluator(ground_truth, llm_answer)

        elif task_or_subtask in ["coding_completion", "LCB_generation"]:  # Coding
            question_obj = state.metadata["coding"]
            question_obj["question_id"] = state.sample_id
            # Runs the model's code, so in the sandbox (samples need lcb_sandbox.sandbox_coding_samples)
            value = await lcb_generation_sandboxed(question_obj, llm_answer)

        elif category == "instruction_following":
            ensure_nltk_resource()  # Required before calling instruction following
            # Instruction following is handled differently see `instruction_following.py`
            question_obj = state.metadata["instruction_following"]
            question_obj["question_id"] = state.sample_id
            value = instruction_following_process_results(
                question_obj, prompt_text, llm_answer
            )  # Instruction following

        else:  # Simpler mappings - see scorer_map
            scorer_function = scorer_map[task_or_subtask]  # let it fail if not found
            value = scorer_function(ground_truth, llm_answer)

        return Score(value=value, answer=llm_answer)

    return score
