"""LiveBench AMPS_Hard scoring without the OpenAI tiebreaker.

Copy of ``livebench.process_results.math.AMPS_Hard.utils.amps_hard_process_results``
with one change: when sympy does not confirm equivalence, upstream asks gpt-5-mini
(``is_equiv_llm``), and raises RuntimeError when OPENAI_API_KEY is unset. Compute
nodes have neither the key nor network access, so upstream errors on every AMPS_Hard
sample that is not an exact sympy match -- i.e. on every wrong answer. Here an
unconfirmed answer simply scores 0.

Copied from https://github.com/LiveBench/LiveBench (see ``LICENSE`` in this folder, which carries LiveBench's
license file as published), with the change above.
"""

import warnings


def amps_hard_process_results(ground_truth: str, llm_answer: str) -> int:  # noqa: C901  kept as upstream
    """1 if the answer's last boxed (or $-delimited) expression is equivalent to the ground truth, else 0."""
    from livebench.process_results.math.AMPS_Hard.utils import (  # type: ignore
        is_equiv,
        normalize_final_answer,
    )
    from livebench.process_results.util import (  # type: ignore
        last_boxed_only_string,
        remove_boxed,
    )

    parsed_answer = None

    if isinstance(ground_truth, list):
        ground_truth = ground_truth[-1]
    for old, new in (
        ("+C", ""),
        ("+ C", ""),
        ("+ c", ""),
        ("+c", ""),
        ("\\\\fbox{", "\\\\boxed{"),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\left", ""),
        ("\\right", ""),
        ("\\bigl", ""),
        ("\\bigr", ""),
        ("\\Bigl", ""),
        ("\\Bigr", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\n", ""),
        ("\\cdot", "*"),
    ):
        llm_answer = llm_answer.replace(old, new)
    for old, new in (("\\left", ""), ("\\right", ""), (" ^", "^"), ("\\ ", "*")):
        ground_truth = ground_truth.replace(old, new)

    last_boxed = last_boxed_only_string(llm_answer)
    if last_boxed:
        parsed_answer = normalize_final_answer(remove_boxed(last_boxed))

    if parsed_answer is None:
        # try to extract from the last block of $ $
        last_line = llm_answer.split("\n")[-1]
        if last_line.count("$") >= 2:
            close_pos = last_line.rfind("$")
            if last_line[close_pos - 1] == "$":
                # make sure this works with $$ $$ blocks too
                close_pos -= 1
            open_pos = last_line.rfind("$", 0, close_pos)
            math = last_line[open_pos + 1 : close_pos]
            if "=" in math:
                math = math.split("=")[-1].strip()
            elif "\\quad \\text{or} \\quad" in math:
                math = math.split("\\quad \\text{or} \\quad")[-1].strip()
            parsed_answer = normalize_final_answer(math)

    if parsed_answer is not None:
        try:
            return int(bool(is_equiv(ground_truth, parsed_answer)))
        except TimeoutError:
            warnings.warn("Timeout when comparing ground truth and parsed answer", stacklevel=2)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Error when comparing ground truth and parsed answer: {e}", stacklevel=2)
        return 0

    if len(llm_answer) > 0 and llm_answer[-1] == ".":
        llm_answer = llm_answer[:-1]
    return int(ground_truth == llm_answer[-len(ground_truth) :])
