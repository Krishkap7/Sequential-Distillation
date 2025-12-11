import os
import argparse
from typing import List

from distil_hidden_sequential import (
    build_config as build_hidden_config,
    run_hidden_distillation,
)


def sanitize_model_id(model_id: str) -> str:
    """
    Turn a HF model ID like 'Qwen/Qwen3-1.7B' into a filesystem-friendly name.
    """
    return model_id.replace("/", "_").replace(":", "_")


def run_parallel_hidden(
    teacher: str,
    students: List[str],
    base_output_dir: str,
):
    os.makedirs(base_output_dir, exist_ok=True)

    for student in students:
        student_name = sanitize_model_id(student)
        output_dir = os.path.join(
            base_output_dir,
            f"{student_name}_from_{sanitize_model_id(teacher)}_hidden",
        )

        cfg = build_hidden_config(
            teacher=teacher,
            student=student,
            output_dir=output_dir,
        )
        run_hidden_distillation(cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hidden-state-based parallel distillation: a single teacher distills into one or "
            "more students (run sequentially in this process, but conceptually parallel)."
        )
    )

    parser.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="HF model ID or local path for the teacher model (e.g. arcee-ai/Arcee-Spark).",
    )
    parser.add_argument(
        "--students",
        type=str,
        nargs="+",
        required=True,
        help="One or more HF model IDs for student models (e.g. Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B).",
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default="./parallel_checkpoints/hidden",
        help="Base directory under which each student checkpoint will be stored.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_parallel_hidden(
        teacher=args.teacher,
        students=args.students,
        base_output_dir=args.base_output_dir,
    )


if __name__ == "__main__":
    main()


