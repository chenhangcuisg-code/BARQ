import argparse
import os

import torch
from transformers import AutoModelForCausalLM

from datautils import _load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 OT-GPTQ 量化后的 Llama3.1 模型做推理"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="OT-GPTQ 导出的 HF checkpoint 目录（例如 ot_runs/20260309_120000）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="输入提示词",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="生成的最大新 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="nucleus sampling 的 top_p",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="使用贪心解码（忽略 temperature / top_p）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"加载量化模型自: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # 复用 datautils 中的 tokenizer 加载逻辑，兼容 Llama3.1 本地权重
    tokenizer = _load_tokenizer(model_dir)

    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=" * 80)
    print(text)
    print("=" * 80)


if __name__ == "__main__":
    main()

