#!/usr/bin/env python3
import argparse
from pathlib import Path

from eval import read_json, validate_struct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute parse/schema/rule rates for a folder of JSON files."
    )
    parser.add_argument(
        "--json-dir",
        default="/mnt/ssd1/dataset_ft_VLM/Dataset_v2_json_final_prompt",
        help="Folder containing JSON files to validate.",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    json_files = sorted(p for p in json_dir.iterdir() if p.is_file() and p.suffix == ".json")
    if not json_files:
        raise ValueError(f"No .json files found in {json_dir}")

    parse_ok = 0
    schema_ok = 0
    rule_ok = 0

    for p in json_files:
        obj = read_json(p)
        vrep = validate_struct(obj)
        if vrep.parse_ok:
            parse_ok += 1
        if vrep.schema_ok:
            schema_ok += 1
        if vrep.rule_ok:
            rule_ok += 1

    total = len(json_files)
    parse_rate = parse_ok / total
    schema_rate = schema_ok / total
    rule_rate = rule_ok / total

    print(f"n_json: {total}")
    print(f"parse_rate: {parse_rate:.6f}")
    print(f"schema_rate: {schema_rate:.6f}")
    print(f"rule_rate: {rule_rate:.6f}")


if __name__ == "__main__":
    main()
