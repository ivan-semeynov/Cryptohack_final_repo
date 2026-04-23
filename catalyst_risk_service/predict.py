from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from model_utils import load_model_package, predict_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict catalyst risk.")
    parser.add_argument("--model-dir", required=True, help="Directory with model.joblib and metadata.json")
    parser.add_argument("--input", required=True, help="CSV or JSON file with feature rows")
    parser.add_argument("--output", required=False, help="Optional path to save predictions")
    args = parser.parse_args()

    package = load_model_package(args.model_dir)
    input_path = Path(args.input)

    if input_path.suffix.lower() == ".json":
        records = json.loads(input_path.read_text(encoding="utf-8"))
    else:
        records = pd.read_csv(input_path).to_dict(orient="records")

    predictions = predict_records(records, package)
    payload = {"predictions": predictions, "threshold": package.threshold}

    if args.output:
        output_path = Path(args.output)
        if output_path.suffix.lower() == ".csv":
            pd.DataFrame(predictions).to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
