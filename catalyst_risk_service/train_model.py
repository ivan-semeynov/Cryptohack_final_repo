from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_utils import load_source, save_model_package, train_model_from_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Train catalyst risk production model.")
    parser.add_argument("--data", required=True, help="Path to unified_3h.csv")
    parser.add_argument("--output-dir", required=True, help="Directory where model and artifacts will be stored")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_source(args.data)
    package, artifacts = train_model_from_dataframe(df)
    saved = save_model_package(package, output_dir)

    artifacts["prepared_dataset"].to_csv(output_dir / "prepared_dataset.csv", index=False, encoding="utf-8-sig")
    artifacts["quality_report"].to_csv(output_dir / "quality_report.csv", index=False, encoding="utf-8-sig")
    artifacts["train_df"].to_csv(output_dir / "train_split.csv", index=False, encoding="utf-8-sig")
    artifacts["val_df"].to_csv(output_dir / "validation_split.csv", index=False, encoding="utf-8-sig")
    artifacts["test_df"].to_csv(output_dir / "test_split.csv", index=False, encoding="utf-8-sig")

    summary = {
        "saved": {k: str(v) for k, v in saved.items()},
        "metrics": package.metrics,
        "top_features": package.feature_importance[:10],
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
