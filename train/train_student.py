"""Weekly QLoRA training — designed to run as a Colab cell (T4, free tier).

Usage (in Colab):
    !pip install unsloth
    !python train_student.py --solver quantum --week 3 --batch logs/week03_batch_quantum.jsonl

One run per solver lineage per week. Continues from that lineage's previous
adapter. ALL hyperparameters come from config/protocol.yaml — frozen; the CLI
only chooses lineage, week, and data file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True, choices=["greedy", "classical", "quantum"])
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--batch", required=True, help="jsonl of {prompt, response} pairs")
    ap.add_argument("--config", default="config/protocol.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    lcfg, tcfg = cfg["lora"], cfg["training"]

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    prev_adapter = Path(f"adapters/{args.solver}/week{args.week - 1:02d}")
    base = str(prev_adapter) if prev_adapter.exists() else cfg["student"]["base_model"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        load_in_4bit=lcfg["load_in_4bit"],
        max_seq_length=2048,
    )
    if not prev_adapter.exists():
        model = FastLanguageModel.get_peft_model(
            model,
            r=lcfg["r"],
            lora_alpha=lcfg["alpha"],
            lora_dropout=lcfg["dropout"],
            target_modules=lcfg["target_modules"],
            random_state=cfg["student"]["seed"],
        )

    rows = [json.loads(l) for l in Path(args.batch).read_text().splitlines() if l.strip()]
    ds = Dataset.from_list(
        [
            {
                "text": tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": r["prompt"]},
                        {"role": "assistant", "content": r["response"]},
                    ],
                    tokenize=False,
                )
            }
            for r in rows
        ]
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=tcfg["batch_size"],
            gradient_accumulation_steps=tcfg["grad_accum"],
            num_train_epochs=tcfg["epochs"],
            learning_rate=tcfg["lr"],
            warmup_ratio=tcfg["warmup_ratio"],
            lr_scheduler_type=tcfg["scheduler"],
            seed=cfg["student"]["seed"],
            output_dir=f"adapters/{args.solver}/week{args.week:02d}",
            logging_steps=5,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(f"adapters/{args.solver}/week{args.week:02d}")
    print(f"saved adapters/{args.solver}/week{args.week:02d} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
