import json
import os
import shutil
from pathlib import Path

import torch


RUN_DIR = Path(os.environ.get("RUN_DIR", "runs/tinystories_gpt"))
EXPORT_DIR = RUN_DIR / "export"
DEFAULT_OUTPUT = EXPORT_DIR / "tinystories_gpt_weights_only.pt"


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def latest_checkpoint(checkpoint_dir):
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def json_ready_merge_rules(merge_rules):
    ready = []
    for rule in merge_rules:
        ready.append(
            {
                "pair": list(rule["pair"]),
                "new_token": rule["new_token"],
                "count": rule["count"],
            }
        )
    return ready


def export_checkpoint(checkpoint_path, output_path):
    checkpoint = load_checkpoint(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export = {
        "step": checkpoint["step"],
        "config": checkpoint["config"],
        "model_state": checkpoint["model_state"],
        "merge_rules": checkpoint["merge_rules"],
        "vocab_tokens": checkpoint["vocab_tokens"],
    }
    torch.save(export, output_path)

    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "export_path": str(output_path),
        "step": checkpoint["step"],
        "config": checkpoint["config"],
        "vocab_size": len(checkpoint["vocab_tokens"]),
        "merge_rule_count": len(checkpoint["merge_rules"]),
        "merge_rules": json_ready_merge_rules(checkpoint["merge_rules"]),
        "vocab_tokens": checkpoint["vocab_tokens"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    loss_log_path = checkpoint_path.parents[1] / "loss_log.csv"
    copied_loss_log = None
    if loss_log_path.exists():
        copied_loss_log = output_path.parent / loss_log_path.name
        shutil.copy2(loss_log_path, copied_loss_log)

    readme_path = output_path.parent / "README.txt"
    readme_path.write_text(
        "\n".join(
            [
                "TinyStories GPT export",
                "",
                f"weights: {output_path.name}",
                f"metadata: {metadata_path.name}",
                f"loss log: {copied_loss_log.name if copied_loss_log else 'not found'}",
                "",
                "This export contains model weights, model config, BPE merge rules, and vocab tokens.",
                "It does not contain optimizer state, so it is intended for local sampling/eval rather than resuming training.",
                "",
            ]
        )
    )

    return {
        "output_path": output_path,
        "metadata_path": metadata_path,
        "loss_log_path": copied_loss_log,
        "readme_path": readme_path,
    }


def main():
    checkpoint_path = latest_checkpoint(RUN_DIR / "checkpoints")
    paths = export_checkpoint(checkpoint_path, DEFAULT_OUTPUT)

    print(f"source checkpoint: {checkpoint_path}")
    print(f"weights-only export: {paths['output_path']}")
    print(f"metadata json: {paths['metadata_path']}")
    if paths["loss_log_path"]:
        print(f"loss log copied: {paths['loss_log_path']}")
    print(f"readme: {paths['readme_path']}")


if __name__ == "__main__":
    main()
