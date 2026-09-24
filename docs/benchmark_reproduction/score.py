"""Score saved answers with one isolated official LiveBench grader per answer."""

import argparse
import json
import subprocess
import uuid
from pathlib import Path

from common import (
    LIVEBENCH_COMMIT,
    SELECTION_SHA256,
    TASK_QUOTAS,
    check_questions,
    checked_item,
    digest,
    load_questions,
    read_json,
    save_json,
)

IMAGE = "livebench-short-grader:1263ee4"


def container_command(image, name):
    return ["docker", "run", "--rm", "--name", name, "-i", "--network", "none",
            "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "128", "--memory", "4g", "--cpus", "2", "--user", "65534:65534",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m", image]


def isolated_score(question, answer, image):
    name = "livebench-short-" + uuid.uuid4().hex
    try:
        completed = subprocess.run(
            container_command(image, name),
            input=json.dumps({"question": question, "answer": answer}, default=str),
            text=True, capture_output=True, check=True, timeout=900,
        )
        result = json.loads(completed.stdout)
        if result["livebench_commit"] != LIVEBENCH_COMMIT:
            raise ValueError("container uses a different LiveBench commit")
        score = result["score"]
        if type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError("official scorer returned an invalid score")
        return score
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"isolated grader failed:\n{error.stderr[-2000:]}") from None
    finally:
        subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=30)


def score_run(root, image=IMAGE):
    protocol = read_json(root / "protocol.json")
    if protocol.get("version") != 1 or protocol.get("selection_sha256") != SELECTION_SHA256:
        raise ValueError("run does not use the frozen protocol")
    expected = load_questions(offline=True)
    questions = check_questions(read_json(root / "questions.json"), expected)
    items = [checked_item(root / f"{q['question_id']}.json", q, protocol) for q in questions]
    results = []
    for question, item in zip(questions, items, strict=True):
        answer = item["answer"]
        score = 0.0 if answer["cap_hit"] else isolated_score(question, answer["final_text"], image)
        results.append({"question_id": question["question_id"], "category": question["category"],
                        "task": question["task"], "score": score, "cap_hit": answer["cap_hit"]})
        print(f"{question['category']}/{question['task']}: {score:.3f}", flush=True)
    categories = {category: sum(r["score"] for r in results if r["category"] == category) / 8
                  for category in TASK_QUOTAS}
    return {"protocol_sha256": digest(protocol), "livebench_commit": LIVEBENCH_COMMIT,
            "grader_image": image, "results": results, "categories": categories,
            "macro": sum(categories.values()) / 6, "scale": "0..1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image", default=IMAGE)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = score_run(args.run, args.image)
    save_json(args.out, report)
    print(json.dumps({"categories": report["categories"], "macro": report["macro"]}, indent=2))


if __name__ == "__main__":
    main()
