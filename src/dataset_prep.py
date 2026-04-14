"""Prepare training data from the FineVision dataset.

Downloads and processes a subset of HuggingFaceM4/FineVision into a
JSON annotation file suitable for INTERLACE fine-tuning.
"""

import os
import json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names


def pil_to_img_path(img_pil, data_dir):
    img_pil = img_pil.convert("RGB")
    img_file = f"img_{int(time.time() * 1000000)}.jpg"
    img_path = os.path.join(data_dir, img_file)
    img_pil.save(img_path)
    return img_path


def dataset_processing(rec, data_dir):
    images_paths = []
    images = rec.get("images")
    if len(images) > 3:
        return None
    for img in images:
        image_path = pil_to_img_path(img, data_dir)
        images_paths.append(image_path)
    flag_first_user_msg = len(images_paths) > 0

    initial_img_text = "".join(["<image>\n" for _ in range(len(images_paths))])
    images_paths = images_paths[0] if len(images_paths) == 1 else images_paths
    texts = rec.get("texts")
    max_msgs = 5
    conversations = []
    flag_finish = False

    for conv_item in texts:
        for key, value in conv_item.items():
            if key == "user":
                if flag_first_user_msg:
                    value = initial_img_text + value.replace("<image>", "")
                    flag_first_user_msg = False
                conversations.append({"from": "human", "value": value})
            elif key == "assistant":
                conversations.append({"from": "gpt", "value": value})
                if len(conversations) >= max_msgs:
                    flag_finish = True
                    break
            elif key == "system":
                conversations.append({"from": "system", "value": value})
        if flag_finish:
            break

    if len(images_paths) > 0:
        new_item = {"image": images_paths, "conversations": conversations}
    else:
        new_item = {"conversations": conversations}

    return new_item


def jsonl_to_json(jsonl_path, json_path):
    """Convert JSONL file to JSON array format."""
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Converted {len(data)} records to {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FineVision dataset")
    parser.add_argument(
        "--sample_portion",
        type=float,
        default=0.01,
        help="Fraction of each subset to sample (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root directory for saving images and annotations",
    )
    parser.add_argument(
        "--sample_counts_path",
        type=str,
        default="dataset_sample_counts.json",
        help="Path to JSON with per-subset sample counts",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    portion_str = str(args.sample_portion).split(".")[-1]
    images_folder = os.path.join(args.output_dir, f"FineVision_{portion_str}")
    os.makedirs(images_folder, exist_ok=True)
    train_save_path = os.path.join(
        args.output_dir, f"FineVision_{portion_str}.jsonl"
    )

    with open(args.sample_counts_path, "r") as f:
        sample_counts = json.load(f)

    finevision_subsets = get_dataset_config_names("HuggingFaceM4/FineVision")
    print(f"FineVision subsets: {finevision_subsets}")

    file_mode = "a" if os.path.exists(train_save_path) else "w"
    n_processed = 0

    for subset in finevision_subsets:
        try:
            dataset = load_dataset(
                "HuggingFaceM4/FineVision", subset, streaming=True
            )
            n_samples = max(
                1, int(sample_counts.get(subset, 0) * args.sample_portion)
            )
            dataset_stream = dataset["train"].take(n_samples)
            progress_bar = tqdm(total=n_samples, desc=f"Processing {subset}")

            with open(train_save_path, file_mode) as f:
                for item in dataset_stream:
                    new_ds_item = dataset_processing(item, images_folder)
                    if new_ds_item is not None:
                        f.write(json.dumps(new_ds_item) + "\n")
                        f.flush()
                    progress_bar.update(1)
                    del item, new_ds_item

            progress_bar.close()
            file_mode = "a"
        except Exception as e:
            print(f"Error processing {subset}: {e}")

        n_processed += 1
        if subset == "yesbut":
            break
        print(f"Processed {n_processed} / {len(finevision_subsets)} subsets.")

    json_path = os.path.join(
        args.output_dir, f"FineVision_{portion_str}.json"
    )
    jsonl_to_json(train_save_path, json_path)
    print(f"Dataset saved to {json_path}")


if __name__ == "__main__":
    main()
