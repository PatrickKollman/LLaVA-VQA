"""
Python script for formatting LLaVA inference answers for submission
to VizWiz's VQA evaluation servers.

For more information: https://vizwiz.org/tasks-and-datasets/vqa/
"""

import json

from fire import Fire
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def vizwiz_submission(annotation_file: str, result_file: str, output_file: str, eval_ai_process: bool = False) -> None:
    """Verify LLaVA inference answers and annotations match. Conform to desired format

    :param annotation_file: file path for test annotations
    :param result_file: file path for inference answers
    :param output_file: file path for new, formatted data
    :param eval_ai_process: run Eval AI server pre processing.
        Not necessary because the Eval AI servers already do this.
    """
    # Aggregate answers from result file
    results = []
    error_line = 0
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:  # pylint: disable=bare-except
                error_line += 1
    print(f"Number of Output Answers: {len(results)}. Number of Error Lines: {error_line}")

    # Aggregate annotation filenames
    with open(annotation_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    print(f"Total Annotations: {len(annotations)}")

    # Check results match annotations
    assert len(results) == len(annotations)

    # Convert to Submission Answers
    submission_answers = []
    eval_ai_processor = EvalAIAnswerProcessor()
    for res in results:
        if eval_ai_process:
            submission_answers.append({"image": res["image_filename"], "answer": eval_ai_processor(res["answer"])})
        else:
            submission_answers.append({"image": res["image_filename"], "answer": res["answer"]})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(submission_answers, f)


if __name__ == "__main__":
    Fire(vizwiz_submission)
