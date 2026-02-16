import argparse
import json
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

SYSTEM_PROMPT = """You are an information extraction system for Toulmin argument components.

Definitions:
- Claim: a disputable position, judgment, or persuasive statement that the text is trying to promote.
- Ground: any explicitly stated reason, evidence, fact, example, or justification that supports the claim.
- Warrant: an explicitly stated assumption, principle, or reasoning that explains why the ground supports the claim.

Instructions:
- Extract only information that is explicitly stated in the text.
- Do not infer, assume, or add unstated reasoning.
- Components may be absent.
- If a component is not explicitly present, output "Not present".
- Warrants are rare; do not create one unless the reasoning link is clearly written.
- If the text is purely factual or descriptive and does not promote a disputable position, the claim is "Not present".

Return STRICT JSON only: {"claim": "...", "ground": "...", "warrant": "..."}"""


def make_example(text: str, label: str):
    # TRL SFTTrainer can apply chat templates if you provide "messages".
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text:\n{text}\n\nExtract claim/ground/warrant."},
            {"role": "assistant", "content": label},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="meta-llama/Llama-3.1-8B-Instruct")

    # CSV inputs
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--eval_csv", default=None)

    # Column names
    ap.add_argument("--text_col", default="ad_text")
    ap.add_argument("--label_col", default="label")  # e.g., "gpt_toulmin_json" if that's your column

    ap.add_argument("--out", default="./llama3.1-claim-sft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)

    # Optional: limit rows for quick tests
    ap.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bf16 if possible, else fp16, else fp32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=None,
    )

    # Align model and generation configs with tokenizer special tokens.
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.bos_token_id is not None:
            model.generation_config.bos_token_id = tokenizer.bos_token_id

    # Load CSV
    train_raw = load_dataset("csv", data_files=args.train_csv, split="train")

    # Optional limit
    if args.limit is not None:
        train_raw = train_raw.select(range(min(args.limit, len(train_raw))))

    # Validate columns exist
    for col in [args.text_col, args.label_col]:
        if col not in train_raw.column_names:
            raise ValueError(
                f"Column '{col}' not found in train CSV. Found columns: {train_raw.column_names}"
            )

    train_ds = train_raw.map(
        lambda r: make_example(r[args.text_col], r[args.label_col]),
        remove_columns=train_raw.column_names,
    )

    eval_ds = None
    if args.eval_csv:
        eval_raw = load_dataset("csv", data_files=args.eval_csv, split="train")
        for col in [args.text_col, args.label_col]:
            if col not in eval_raw.column_names:
                raise ValueError(
                    f"Column '{col}' not found in eval CSV. Found columns: {eval_raw.column_names}"
                )
        eval_ds = eval_raw.map(
            lambda r: make_example(r[args.text_col], r[args.label_col]),
            remove_columns=eval_raw.column_names,
        )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_args = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        logging_steps=10,
        save_steps=200,
        eval_steps=200 if eval_ds is not None else None,
        eval_strategy="steps" if eval_ds is not None else "no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
