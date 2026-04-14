"""Compute hidden state cosine similarities across layers.

For each pair of layers separated by `pack_size`, computes the average
cosine similarity of their hidden state representations. These similarity
scores are used by the INTERLACE layer selection algorithm.
"""

import torch
import os
import json
import argparse
import random
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch.nn.functional as F


def load_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    return model, processor


def get_hidden_states(args):
    model, processor = load_model(args.model_path)
    model.eval()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset file {args.dataset_path} not found.")

    with open(args.dataset_path, "r") as f:
        data = json.load(f)

    sample_size = int(len(data) * args.sample_portion)
    random.seed(args.random_seed)
    data = random.sample(data, sample_size)

    total_layers = len(model.model.language_model.layers)
    all_similarities = {i: [] for i in range(total_layers - args.pack_size)}

    progress_bar = tqdm(total=len(data), desc="Computing hidden state similarities")

    for item in data:
        if "image" not in item:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item["conversations"][0]["value"]},
                    ],
                }
            ]
        elif isinstance(item["image"], str):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {
                            "type": "text",
                            "text": item["conversations"][0]["value"].replace(
                                "<image>\n", ""
                            ),
                        },
                    ],
                }
            ]
        elif isinstance(item["image"], list):
            image_contents = [
                {"type": "image", "image": img_path} for img_path in item["image"]
            ]
            image_contents.append(
                {
                    "type": "text",
                    "text": item["conversations"][0]["value"].replace(
                        "<image>\n", ""
                    ),
                }
            )
            messages = [{"role": "user", "content": image_contents}]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for i in range(total_layers - args.pack_size):
            input_hs = hidden_states[i].squeeze(0)
            output_hs = hidden_states[i + args.pack_size].squeeze(0)
            cosine_sim = F.cosine_similarity(input_hs, output_hs, dim=1)
            all_similarities[i].append(cosine_sim.mean().item())

        progress_bar.update(1)
    progress_bar.close()

    model_name = args.model_path.split("/")[-1]
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]
    output_filename = (
        f"{model_name}_{dataset_name}_pack{args.pack_size}"
        f"_hidden_state_similarities.json"
    )
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_similarities, f, indent=4)
    print(f"Saved similarity scores to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute hidden state similarities for INTERLACE layer selection"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the JSON dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hidden_states",
        help="Directory to save similarity scores",
    )
    parser.add_argument(
        "--sample_portion",
        type=float,
        default=0.1,
        help="Fraction of dataset to sample",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--pack_size",
        type=int,
        default=3,
        help="Number of layers in each group for similarity computation",
    )
    args = parser.parse_args()
    get_hidden_states(args)
