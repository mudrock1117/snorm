#!/usr/bin/env python
"""Select reproducible, non-contiguous Fog/Night diagnostic samples."""
import argparse
import json
from pathlib import Path

from detection.config import cfg
from detection.data.build import build_datasets
from diagnostics.dataset_index import (
    canonical_sample_id,
    stratified_sample_indices,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output-ids", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--count-per-domain", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    datasets = build_datasets(
        cfg.DATASETS.TESTS,
        transforms=cfg.INPUT.TRANSFORMS_TEST,
        is_train=False,
    )

    selected = []
    lines = []
    for dataset in datasets:
        domain = dataset.dataset_name
        lower = domain.lower()
        if "fog" not in lower and "night" not in lower:
            print("skip non Fog/Night dataset:", domain)
            continue
        indices = stratified_sample_indices(
            len(dataset), args.count_per_domain, args.seed, domain
        )
        for index in indices:
            sample_id = canonical_sample_id(dataset, index, domain)
            lines.append("%s:%s" % (domain, sample_id))
            selected.append(
                {
                    "dataset": domain,
                    "dataset_index": int(index),
                    "sample_id": sample_id,
                    "selection": "seeded stratified index sampling",
                }
            )
        print("selected %d samples from %s" % (len(indices), domain))

    output_ids = Path(args.output_ids)
    output_ids.parent.mkdir(parents=True, exist_ok=True)
    output_ids.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "seed": int(args.seed),
        "count_per_domain": int(args.count_per_domain),
        "note": (
            "Not VOC evaluation. Indices are sampled across equal dataset strata; "
            "actual target count/scale distributions are reported after inference."
        ),
        "samples": selected,
    }
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("IDs:", output_ids)
    print("manifest:", output_manifest)


if __name__ == "__main__":
    main()
