import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from decomposer_v4.eval.ResearchQA.researchqa_loader import export_researchqa_split


SPLIT_ALIASES = {
    "full": "full.json",
    "full.json": "full.json",
    "test": "test.json",
    "test.json": "test.json",
    "valid": "valid.json",
    "valid.json": "valid.json",
    "test_mini": "test_mini",
    "test_mini.json": "test_mini",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ResearchQA dataset from Hugging Face.")
    parser.add_argument(
        "--split",
        type=str,
        default="test_mini",
        choices=sorted(SPLIT_ALIASES.keys()),
        help="Which split to download. `test_mini` is exported from the Hugging Face dataset builder.",
    )
    parser.add_argument("--output-path", type=str, default="", help="Optional explicit output path.")
    args = parser.parse_args()
    requested_split = SPLIT_ALIASES[args.split]

    if requested_split == "test_mini":
        output_path = Path(args.output_path).expanduser() if args.output_path else Path("test_mini.json")
        exported = export_researchqa_split("test_mini", output_path)
        print(f"Downloaded test_mini to: {exported.resolve()}")
    else:
        file_path = hf_hub_download(
            repo_id="realliyifei/ResearchQA",
            filename=requested_split,
            repo_type="dataset",
        )
        if args.output_path:
            output_path = Path(args.output_path).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, output_path)
            print(f"Downloaded {requested_split} to: {output_path.resolve()}")
        else:
            print(f"Downloaded {requested_split} to: {file_path}")
