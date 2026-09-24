"""Derive the public question set and keep small, durable JSON records."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

SCHEMA = "moespresso-livebench-short-v1"
RELEASE = "2024-11-25"
SELECTION_SHA256 = "94b769c87673b7bcec1b851ee921a82684d8a135a44a83bfde4872c3cbef8049"
LIVEBENCH_COMMIT = "1263ee472f4b9ac3833c0d2f6ad50dd3747fd1df"
PACKAGE_ID = "pkg:e469be1b1a19e966d4f546c8eaa7b5ee980ad19f0e41883d3c25678a4728fef5"
MANIFEST_SHA256 = "4f3faa73c23cfd0e0624f703514d9d92bc5d69aed68f0c43a405da0e9a6f2182"

SOURCE_REVISIONS = {
    "coding": "a958549fdd8aa57be0a3fafe7b205ffc160ed5f4",
    "data_analysis": "31b9661ff678df9958e2f7fa228427f4c858c1a1",
    "instruction_following": "0868379c4b5cf62aeacaf8be4f08fced815c81bb",
    "language": "3ada32a2e53d5e04e57fa503384cb85ce9116c40",
    "math": "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb",
    "reasoning": "6fc6498a5dfba553f69f4413feabade1f1a2d384",
}
SOURCE_SHA256 = {
    "coding": "5f02d01fb21672f5d84169f940adab46ff3ca09b9159fd42fdd289bc9be23502",
    "data_analysis": "fb86a7a02fa9eabf785d9e8af85955990cf6d228cc9d3f805a54f863bc8c4c52",
    "instruction_following": "a9bb97bbaf8788142c310bcb33d50e2f6f5df8cbd8b8c3db677816b06f0f4f25",
    "language": "76ba142afd242ca02d6baa8bb737608d2b416674f311d8f9d798b4e3908a499c",
    "math": "3d365cad1f9b8d7c5416d63866653d6854b270d9c93b51323d405ed5fd51df54",
    "reasoning": "4204bb94c812690ef8ba5f4c1f10b5b1082ca0b7bc532166834f798aa56e2a3c",
}
TASK_QUOTAS = {
    "coding": {"coding_completion": 4, "LCB_generation": 4},
    "data_analysis": {"cta": 4, "tablejoin": 4},
    "instruction_following": {"paraphrase": 4, "simplify": 4},
    "language": {"typos": 4, "plot_unscrambling": 4},
    "math": {"math_comp": 4, "AMPS_Hard": 4},
    "reasoning": {"spatial": 4, "web_of_lies_v2": 4},
}
TASK_CAPS = {
    "coding_completion": 512,
    "LCB_generation": 640,
    "cta": 512,
    "tablejoin": 512,
    "paraphrase": 768,
    "simplify": 768,
    "typos": 512,
    "plot_unscrambling": 896,
    "math_comp": 768,
    "AMPS_Hard": 576,
    "spatial": 512,
    "web_of_lies_v2": 1024,
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, sort_keys=True, default=str, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def active_on_release(row):
    introduced = str(row["livebench_release_date"])[:10]
    removed = str(row.get("livebench_removal_date") or "")[:10]
    return introduced <= RELEASE and (not removed or removed > RELEASE)


def selection_rank(category, task, question_id):
    value = f"{SCHEMA}\0{category}\0{task}\0{question_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def check_questions(questions, expected=None):
    count = sum(sum(tasks.values()) for tasks in TASK_QUOTAS.values())
    if len(questions) != count or len({row["question_id"] for row in questions}) != count:
        raise ValueError(f"expected {count} distinct questions")
    counts = Counter((row["category"], row["task"]) for row in questions)
    quotas = Counter({(category, task): count for category, tasks in TASK_QUOTAS.items()
                      for task, count in tasks.items()})
    if counts != quotas:
        raise ValueError("question quotas differ")
    if expected is not None and [digest(row) for row in questions] != [
        digest(row) for row in expected
    ]:
        raise ValueError("questions differ from the salted selection")
    return questions


def select_questions(rows_by_category):
    questions = []
    records = []
    for category, tasks in TASK_QUOTAS.items():
        for task, count in tasks.items():
            candidates = [row for row in rows_by_category[category]
                          if row["task"] == task and active_on_release(row)]
            if len(candidates) < count:
                raise ValueError(f"too few active {category}/{task} questions")
            candidates.sort(key=lambda row: (
                selection_rank(category, task, row["question_id"]), row["question_id"]
            ))
            for row in candidates[:count]:
                questions.append(row)
                records.append({"category": category, "task": task,
                                "question_id": row["question_id"],
                                "question_sha256": digest(row)})
    check_questions(questions)
    contract = {
        "schema": SCHEMA,
        "livebench_source_commit": LIVEBENCH_COMMIT,
        "release": RELEASE,
        "source_revisions": SOURCE_REVISIONS,
        "source_sha256": SOURCE_SHA256,
        "selection_rule": (
            "lowest salted SHA256 question ID ranks within fixed category/task quotas"
        ),
        "task_quotas": TASK_QUOTAS,
        "task_max_output_tokens": TASK_CAPS,
        "questions": records,
    }
    return questions, digest(contract)


def load_questions(offline=False):
    from huggingface_hub import hf_hub_download
    from pyarrow import parquet

    rows = {}
    for category, revision in SOURCE_REVISIONS.items():
        path = Path(hf_hub_download(
            f"livebench/{category}", "data/test-00000-of-00001.parquet",
            repo_type="dataset", revision=revision, local_files_only=offline,
        ))
        with path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
        if checksum != SOURCE_SHA256[category]:
            raise ValueError(f"public dataset hash differs: {category}")
        rows[category] = parquet.read_table(path).to_pylist()
    questions, selection_sha256 = select_questions(rows)
    if selection_sha256 != SELECTION_SHA256:
        raise ValueError("salted selection differs from the frozen benchmark")
    return questions


def answer_fields(response, ceiling):
    if response.get("error") or len(response.get("choices", [])) != 1:
        raise ValueError("response has an error or does not have exactly one choice")
    choice = response["choices"][0]
    message = choice["message"]
    text = message.get("content")
    text = "" if text is None else text
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    finish = choice.get("finish_reason")
    if not isinstance(text, str) or (reasoning is not None and not isinstance(reasoning, str)):
        raise ValueError("response content must be text")
    if (finish not in ("stop", "length") or message.get("tool_calls")
            or message.get("function_call")):
        raise ValueError("response is incomplete or invoked tools")
    tokens = (response.get("usage") or {}).get("completion_tokens")
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        raise ValueError("invalid completion token usage")
    final = text
    if "<think>" in text:
        final = text.split("</think>", 1)[1].strip() if "</think>" in text else ""
    return {"text": text, "final_text": final, "reasoning_content": reasoning,
            "finish_reason": finish, "cap_hit": finish == "length" or
            (tokens is not None and tokens >= ceiling)}


def checked_item(path, question, protocol):
    item = read_json(path)
    if (item.get("protocol_sha256") != digest(protocol)
            or item.get("question_sha256") != digest(question)
            or item.get("question_id") != question["question_id"]):
        raise ValueError(f"incompatible saved item: {path.name}")
    fields = answer_fields(item["response"], protocol["request"][protocol["token_field"]])
    if item.get("answer") != fields:
        raise ValueError(f"saved answer differs from its response: {path.name}")
    return item
