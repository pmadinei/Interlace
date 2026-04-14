"""Prepare domain-specific training datasets for ablation studies.

Downloads and processes individual benchmarks (TextVQA, ChartQA, ScienceQA)
into JSON annotation files for domain-specific fine-tuning experiments.
"""

import os
import json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset


DATASETS = {
    "TextVQA": {
        "repo": "facebook/textvqa",
        "splits": ["train", "validation", "test"],
    },
    "ChartQA": {
        "repo": "HuggingFaceM4/ChartQA",
        "splits": ["train", "val", "test"],
    },
    "ScienceQA": {
        "repo": "derek-thomas/ScienceQA",
        "splits": ["train", "validation", "test"],
    },
}


def pil_to_img_path(img_pil, data_dir):
    img_pil = img_pil.convert("RGB")
    img_file = f"img_{int(time.time() * 1000000)}.jpg"
    img_path = os.path.join(data_dir, img_file)
    img_pil.save(img_path)
    return img_path


def dataset_specific_processing(dataset_name, data_dir, rec):
    if dataset_name == "TextVQA":
        image_path = pil_to_img_path(rec.get("image"), data_dir)
        question = rec.get("question") or rec.get("question_text")
        answers = list(set(rec.get("answers")))
    elif dataset_name == "ChartQA":
        image_path = pil_to_img_path(rec.get("image"), data_dir)
        question = rec.get("query")
        answers = list(set(rec.get("label")))
    elif dataset_name == "ScienceQA":
        image_path = pil_to_img_path(rec.get("image"), data_dir)
        question = rec.get("question")
        choices = rec.get("choices")
        answer_idx = rec.get("answer")
        answers = [f"{chr(65 + answer_idx)}) {choices[answer_idx]}"]
        choices_str = [f"{chr(65 + i)}) {choice}" for i, choice in enumerate(choices)]
        question = question + "\n" + "\n".join(choices_str)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return image_path, question, answers


def main():
    parser = argparse.ArgumentParser(
        description="Prepare domain-specific datasets for ablation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASETS.keys()),
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving images and annotations",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Max samples to process (default: all)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_info = DATASETS[dataset_name]
    images_folder = os.path.join(args.output_dir, "images")
    os.makedirs(images_folder, exist_ok=True)

    train_split = dataset_info["splits"][0]
    train_save_path = os.path.join(args.output_dir, f"{dataset_name}_train.json")

    train_dataset = load_dataset(dataset_info["repo"], split=train_split)

    n_train = len(train_dataset)
    if args.n_samples is not None:
        n_train = min(n_train, args.n_samples)

    new_dataset = []
    for i in tqdm(range(n_train), desc=f"Processing {dataset_name} train split"):
        rec = train_dataset[i]
        if rec.get("image") is None:
            continue

        image_path, question, labels = dataset_specific_processing(
            dataset_name, images_folder, rec
        )
        for label in labels:
            new_dataset.append(
                {
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": f"{question}\n<image>"},
                        {"from": "gpt", "value": f"{label}"},
                    ],
                }
            )

    with open(train_save_path, "w") as f:
        json.dump(new_dataset, f, indent=4)
    print(f"Saved {len(new_dataset)} samples to {train_save_path}")


if __name__ == "__main__":
    main()
