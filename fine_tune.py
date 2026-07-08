"""
Fine-Tuning Script
Fine-tunes a small transformer (distilbert) for skill entity recognition (NER)
on job posting data.

Usage:
    python fine_tune.py --generate-data     # Step 1: Generate training data via Groq
    python fine_tune.py --train             # Step 2: Fine-tune the model
    python fine_tune.py --test "some text"  # Step 3: Test the model

The fine-tuned model is saved to ./models/skill-ner/ and can optionally be
loaded by the main pipeline to boost extraction accuracy.
"""

import argparse
import json
import os
import sys
import requests

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "ner_train.json")
MODEL_DIR = "models/skill-ner"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# BIO tags
LABEL_LIST = ["O", "B-SKILL", "I-SKILL"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def generate_training_data(num_samples: int = 200):
    """Use Groq to generate labeled NER training sentences from job descriptions."""

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("Set GROQ_API_KEY first.")
        sys.exit(1)

    print(f"Generating {num_samples} NER training samples via Groq...")

    prompt = f"""Generate {num_samples} short sentences that might appear in tech job descriptions.
Each sentence should contain 1-3 technical skills, tools, or frameworks.

For EACH sentence, provide BIO NER labels for every token.
B-SKILL = beginning of a skill entity
I-SKILL = inside a skill entity (for multi-word skills like "Apache Spark")
O = not a skill

Return as a JSON array:
[
    {{
        "tokens": ["Experience", "with", "Apache", "Spark", "and", "Python", "required"],
        "labels": ["O", "O", "B-SKILL", "I-SKILL", "O", "B-SKILL", "O"]
    }},
    {{
        "tokens": ["Must", "know", "Docker", "and", "Kubernetes"],
        "labels": ["O", "O", "B-SKILL", "O", "B-SKILL"]
    }}
]

Generate diverse sentences covering: programming languages, ML frameworks,
cloud platforms, databases, DevOps tools, data tools, and AI/LLM tools.
Ensure tokens and labels arrays are always the same length.
Return ONLY the JSON array."""

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You generate NER training data. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.8,
                "max_tokens": 8000,
            },
            timeout=120,
        )

        if resp.status_code != 200:
            print(f"Groq error: {resp.status_code}")
            sys.exit(1)

        text = resp.json()["choices"][0]["message"]["content"].strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        samples = json.loads(text)

        # Validate: tokens and labels must have same length
        valid = []
        for s in samples:
            if len(s["tokens"]) == len(s["labels"]):
                # Normalize labels
                s["labels"] = [l if l in LABEL_LIST else "O" for l in s["labels"]]
                valid.append(s)

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(TRAIN_FILE, "w") as f:
            json.dump(valid, f, indent=2)

        print(f"Saved {len(valid)} valid training samples to {TRAIN_FILE}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def train():
    """Fine-tune distilbert-base-uncased on the generated NER data."""

    if not os.path.exists(TRAIN_FILE):
        print(f"Training data not found. Run: python fine_tune.py --generate-data")
        sys.exit(1)

    print("Loading training data...")
    with open(TRAIN_FILE) as f:
        samples = json.load(f)

    print(f"Training on {len(samples)} samples...")

    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification,
        )
        from datasets import Dataset
        import numpy as np
    except ImportError:
        print("Install: pip install transformers datasets torch")
        sys.exit(1)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128,
        )

        all_labels = []
        for i, labels in enumerate(examples["label_ids"]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = []
            prev_word = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)
                elif wid != prev_word:
                    aligned.append(labels[wid] if wid < len(labels) else 0)
                else:
                    # For sub-word tokens, use I-SKILL if the word is a skill
                    lbl = labels[wid] if wid < len(labels) else 0
                    aligned.append(lbl if lbl == LABEL2ID["I-SKILL"] else (LABEL2ID["I-SKILL"] if lbl == LABEL2ID["B-SKILL"] else 0))
                prev_word = wid
            all_labels.append(aligned)

        tokenized["labels"] = all_labels
        return tokenized

    # Build dataset
    token_lists = [s["tokens"] for s in samples]
    label_id_lists = [[LABEL2ID.get(l, 0) for l in s["labels"]] for s in samples]

    ds = Dataset.from_dict({"tokens": token_lists, "label_ids": label_id_lists})

    # 80/20 split
    split = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"].map(tokenize_and_align, batched=True, remove_columns=["tokens", "label_ids"])
    eval_ds = split["test"].map(tokenize_and_align, batched=True, remove_columns=["tokens", "label_ids"])

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        learning_rate=3e-5,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"\nModel saved to {MODEL_DIR}/")
    print("You can now load it in the pipeline for faster skill extraction.")


def test(text: str):
    """Test the fine-tuned model on a sample text."""

    if not os.path.exists(MODEL_DIR):
        print(f"No model found at {MODEL_DIR}. Run training first.")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    except ImportError:
        print("Install: pip install transformers torch")
        sys.exit(1)

    print(f"Loading model from {MODEL_DIR}...")
    ner = pipeline(
        "ner",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        aggregation_strategy="simple",
    )

    results = ner(text)

    print(f"\nInput: {text}")
    print(f"\nExtracted skills:")
    for ent in results:
        if ent["entity_group"] == "SKILL":
            print(f"  {ent['word']}  (confidence: {ent['score']:.3f})")

    if not results:
        print("  (no skills detected)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NER model for skill extraction")
    parser.add_argument("--generate-data", action="store_true", help="Generate training data via Groq")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", type=str, help="Test with a sample sentence")
    parser.add_argument("--samples", type=int, default=200, help="Number of training samples to generate")

    args = parser.parse_args()

    if args.generate_data:
        generate_training_data(args.samples)
    elif args.train:
        train()
    elif args.test:
        test(args.test)
    else:
        parser.print_help()
