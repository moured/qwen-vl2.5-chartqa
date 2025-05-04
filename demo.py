#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────────
#  Minimal VQA runner for Qwen‑VL 2.0 / 2.5  (7B, 14B, 72B …)
#  Keeps library‑exact behaviour: pixel limits, VQA prompt, token cap,
#  auto‑split device‑map, 72‑B layer placement, 2.5 / 2.0 processor.
# ────────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse, json, os, pathlib, sys, time
from typing import Any, Dict, List
from functools import lru_cache   # ← add this
from tqdm.auto import tqdm   # add at top of file

import torch
from PIL import Image

# -------------------------------------------------------------------#
#  Constants                                                          #
# -------------------------------------------------------------------#
MODEL_ID   = "Qwen/Qwen2.5-VL-7B-Instruct" 
MIN_PIXELS = 1280 * 28 * 28            # 1 003 520
MAX_PIXELS = 16384 * 28 * 28           # 12 843 776

# -------------------------------------------------------------------#
#  Cheap fall‑back helpers for the library utilities (`smp`)          #
# -------------------------------------------------------------------#
def get_rank_and_world_size() -> Tuple[int, int]:
    """Assume single‑GPU unless run under torch.distributed."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1

def get_gpu_memory():
    mems = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        mems.append(props.total_memory // 2**20)         # MiB
    return mems

def auto_split_flag() -> bool:
    return os.environ.get("AUTO_SPLIT", "0") == "1"

# -------------------------------------------------------------------#
#  72‑B device‑map splitter (copied from the official repo)           #
# -------------------------------------------------------------------#
def _split_model_72b() -> Dict[str, int]:
    device_map: Dict[str, int] = {}
    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = max(1, total_gpus // world_size)

    num_layers = 80 + 8                       # 80 transformer + 8 visual
    layers_per_gpu = math.ceil(num_layers / num_gpus)
    dist = [layers_per_gpu] * num_gpus
    dist[0]  -= 6
    dist[-1] -= 2

    layer = 0
    for g, n in enumerate(dist):
        for _ in range(n):
            device_map[f"model.layers.{layer}"] = rank + g * world_size
            layer += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map.update({
        "visual":              rank,
        "model.embed_tokens":  rank,
        "model.norm":          last_gpu,
        "model.rotary_emb":    last_gpu,
        "lm_head":             last_gpu,
    })
    return device_map

# -------------------------------------------------------------------#
#  Library‑compatible model & processor loader                        #
# -------------------------------------------------------------------#
@lru_cache(maxsize=1)
def load_qwen(model_path: str = MODEL_ID):
    """Return (processor, model) with the same logic as Qwen2VLChat."""
    if any(x in model_path.lower() for x in ("2.5", "2_5", "qwen25")):
        from transformers import (
            Qwen2_5_VLForConditionalGeneration as QwenModel,
            AutoProcessor as QwenProcessor,
        )
    else:
        from transformers import (
            Qwen2VLForConditionalGeneration  as QwenModel,
            Qwen2VLProcessor                 as QwenProcessor,
        )
    processor = QwenProcessor.from_pretrained(model_path, trust_remote_code=True)

    gpu_mems     = get_gpu_memory()
    auto_split   = auto_split_flag()
    device_map: str | Dict[str, int]

    if "72b" in model_path.lower():               # big model
        device_map = _split_model_72b()
    elif auto_split and get_rank_and_world_size()[1] == 1:
        device_map = "auto"
    else:
        device_map = "cpu"                        # will .cuda() later if possible

    model = QwenModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    if device_map == "cpu" and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    torch.cuda.empty_cache()
    return processor, model

# -------------------------------------------------------------------#
#  Image preparation                                                  #
# -------------------------------------------------------------------#
def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    p = w * h
    if MIN_PIXELS <= p <= MAX_PIXELS:
        return img
    tgt_p  = max(min(p, MAX_PIXELS), MIN_PIXELS)
    scale  = (tgt_p / p) ** 0.5
    new_wh = (int(w * scale), int(h * scale))
    return img.resize(new_wh, Image.BICUBIC)

# -------------------------------------------------------------------#
#  VQA Inference – fixed (matches Qwen2VLChat.generate_inner)         #
# -------------------------------------------------------------------#
def vqa(image_path: str,
        question: str,
        dataset: str | None = None,
        model_path: str = MODEL_ID) -> str:

    from qwen_vl_utils import process_vision_info   # make sure pip install qwen-vl-utils

    # 1. load model / processor
    processor, model = load_qwen(model_path)

    # 2. build the VQA prompt
    user_prompt = f"{question}\nPlease try to answer the question with short words or phrases if possible."

    # 3. build chat messages with *multimodal* content
    img = _resize(Image.open(image_path).convert("RGB"))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": user_prompt},
        ],
    }]

    # 4. convert messages → text with <|image|> placeholder
    text = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True,
    )

    # 5. collect the vision inputs exactly like the library
    images, videos = process_vision_info([messages])

    # 6. tokenise + feature‑extract
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # 7. decide max tokens (100 for ChartQA)
    gen_kwargs = {"max_new_tokens": 100}

    # 8. generate  – run the model
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, **gen_kwargs)

    # 9. slice away the prompt tokens (keep only newly generated ones)
    new_tokens = gen_ids[0][len(inputs.input_ids[0]):]

    # 10. decode → clean answer
    answer = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return answer

# ─────────────────────────────────────────────────────────── #
#  Relaxed‑accuracy helper  (safe when gold answer == 0)      #
# ─────────────────────────────────────────────────────────── #
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """
    5 % numeric tolerance.  Exact match for non‑numeric.
    Implementation identical to pix2struct (avoids /0).
    """
    def _to_float(text: str):
        try:
            return float(text.rstrip('%')) / 100.0 if text.endswith('%') else float(text)
        except ValueError:
            return None

    prediction, target = str(prediction).strip(), str(target).strip()
    p_float, t_float = _to_float(prediction), _to_float(target)

    # NB: the "and t_float" check is what prevents ZeroDivisionError
    if p_float is not None and t_float:
        rel_change = abs(p_float - t_float) / abs(t_float)
        return rel_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


# ------------------------------------------------------------------ #
#  Batch driver                                                      #
# ------------------------------------------------------------------ #
def run_split(entries, img_root, split_name, model_id):
    """Run VQA over one split, show live progress & predictions."""
    results = []
    for ex in tqdm(entries, desc=f"Infer {split_name}", ncols=80):
        img_path = os.path.join(img_root, ex["imgname"])
        pred     = vqa(img_path, ex["query"], model_path=model_id)

        # live print
        print(f"[{split_name}] Q: {ex['query']}  →  {pred}")

        results.append({
            "imgname":    ex["imgname"],
            "query":      ex["query"],
            "prediction": pred,
            "answer":     ex["label"],
            "split":      split_name,
        })
    return results

# ─────────────────────────────────────────────────────────── #
#  Accuracy for a single split                                #
# ─────────────────────────────────────────────────────────── #
def compute_accuracy(recs: List[Dict[str, Any]]) -> float:
    if not recs:
        return 0.0
    hits = sum(relaxed_correctness(r["answer"], r["prediction"]) for r in recs)
    return hits / len(recs)


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Batch VQA for ChartQA splits")
    ap.add_argument("--test_human",     required=True)
    ap.add_argument("--test_augmented", required=True)
    ap.add_argument("--img_root",       required=True)
    ap.add_argument("--out",            required=True,
                    help="Output predictions JSON (full path or filename)")
    ap.add_argument("--model", default=MODEL_ID,
                    help="HF model id (default Qwen2‑VL‑7B‑Instruct)")
    args = ap.parse_args()

    t0 = time.time()

    # read the two input JSONs
    with open(args.test_human, "r") as f:
        human_entries = json.load(f)
    with open(args.test_augmented, "r") as f:
        aug_entries = json.load(f)

    # inference
    preds_h = run_split(human_entries, args.img_root, "test_human", args.model)
    preds_a = run_split(aug_entries,  args.img_root, "test_augmented", args.model)
    all_preds = preds_h + preds_a

    # save predictions
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_preds, f, indent=2)

    # evaluation
    acc_h = compute_accuracy(preds_h)
    acc_a = compute_accuracy(preds_a)
    total = len(preds_h) + len(preds_a)
    acc_o = (
        (acc_h * len(preds_h) + acc_a * len(preds_a)) / total
        if total else 0.0
    )

    # save evaluation
    eval_json = {
        "test_human":     round(acc_h * 100, 2),
        "test_augmented": round(acc_a * 100, 2),
        "overall":        round(acc_o * 100, 2),
    }
    eval_path = os.path.join(os.path.dirname(args.out), "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_json, f, indent=2)

    # print summary
    print("\n────────  Finished inference  ────────")
    for k, v in eval_json.items():
        print(f"{k:>15}: {v:.2f}%")
    print(f"Predictions saved to : {args.out}")
    print(f"Evaluation  saved to : {eval_path}")
    print(f"Elapsed time         : {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()