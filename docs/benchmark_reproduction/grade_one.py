"""Call the pinned upstream task scorer, exclusively inside the grading container."""

import contextlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

from common import LIVEBENCH_COMMIT


def official_score(question, answer):
    if not Path("/.dockerenv").exists():
        raise RuntimeError("run this grader through score.py; host execution is forbidden")
    source = json.loads(importlib.metadata.distribution("livebench").read_text("direct_url.json"))
    if source["vcs_info"]["commit_id"] != LIVEBENCH_COMMIT:
        raise ValueError("LiveBench installation is not the pinned commit")
    task = question["task"]
    target = question.get("ground_truth")
    if question["category"] == "instruction_following":
        from livebench.if_runner.instruction_following_eval.evaluation_main import (
            read_prompt_list,
            test_instruction_following_strict,
        )
        from livebench.process_results.instruction_following.utils import score_results
        prompt = read_prompt_list([dict(question)])[0]
        result = test_instruction_following_strict(prompt, {prompt.prompt: answer})
        return score_results(result.follow_all_instructions, result.follow_instruction_list)
    if task in ("coding_completion", "LCB_generation"):
        from livebench.process_results.coding.utils import LCB_generation_process_results
        return LCB_generation_process_results(question, answer)
    if task == "cta":
        from livebench.process_results.data_analysis.cta.utils import cta_process_results
        return cta_process_results(target, answer)
    if task == "tablejoin":
        from livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
        return joinmap_process_results(question["turns"][0], target, answer)
    if task == "typos":
        from livebench.process_results.writing.typos.utils import typos_process_results
        return typos_process_results(target, answer)
    if task == "plot_unscrambling":
        from livebench.process_results.writing.plot_unscrambling.utils import (
            plot_unscrambling_process_results,
        )
        return plot_unscrambling_process_results(target, answer)
    if task == "AMPS_Hard":
        from livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results
        return amps_hard_process_results(target, answer)
    if task == "math_comp":
        from livebench.process_results.math.math_competitions.utils import (
            aime_process_results,
            mathcontest_process_results,
        )
        if question["subtask"].startswith("aime"):
            return aime_process_results(target, answer)
        if "amc" in question["subtask"] or question["subtask"].startswith("smc"):
            return mathcontest_process_results(target, answer, question["turns"][0])
    if task == "spatial":
        from livebench.process_results.reasoning.spatial.utils import spatial_process_results
        return spatial_process_results(target, answer)
    if task == "web_of_lies_v2":
        from livebench.process_results.reasoning.web_of_lies_v2.utils import (
            web_of_lies_process_results,
        )
        return web_of_lies_process_results(target, answer)
    raise ValueError(f"unsupported frozen task: {task}")


if __name__ == "__main__":
    if not Path("/.dockerenv").exists():
        raise SystemExit("The grading worker may only run inside Docker.")
    os.chdir("/tmp")
    request = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        score = float(official_score(request["question"], request["answer"]))
    print(json.dumps({"score": score, "livebench_commit": LIVEBENCH_COMMIT}, allow_nan=False))
