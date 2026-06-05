from __future__ import annotations

import argparse
import json
import re
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from tokenizer_evaluation.datasets.nsynth import pitch_stratified_sample
from tokenizer_evaluation.embeddings import load_embedding_bundle
from tokenizer_evaluation.metrics import compute_knn_purity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate timbre-label kNN purity for saved embedding bundles."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing model subdirectories with embeddings.npz files.",
    )
    source.add_argument(
        "--embeddings",
        nargs="+",
        type=Path,
        help="One or more embeddings.npz files to evaluate.",
    )
    parser.add_argument(
        "--pattern",
        default="*/embeddings.npz",
        help="Glob pattern used under --input-dir. Default: */embeddings.npz",
    )
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--label-col", default="instrument_family_str")
    parser.add_argument("--pitch-label-col", default="pitch")
    parser.add_argument("--source-label-col", default="instrument_source_str")
    parser.add_argument(
        "--sample-family-col",
        default="instrument_family_str",
        help=(
            "Metadata column used for within-pitch family balancing when "
            "--pitch-stratified is enabled."
        ),
    )
    parser.add_argument("--pitch-stratified", action="store_true")
    parser.add_argument("--pitch-bin-size", default=1, type=int)
    parser.add_argument("--max-per-family", default=None, type=int)
    parser.add_argument("--max-per-pitch", default=None, type=int)
    parser.add_argument("--keep-incomplete-pitch-strata", action="store_true")
    parser.add_argument("--pitch-source-k", default=10, type=int)
    parser.add_argument("--skip-pitch-source", action="store_true")
    parser.add_argument(
        "--k",
        "--ks",
        dest="ks",
        nargs="+",
        default=[5, 10, 30, 50],
        type=int,
        help="One or more k values for the main purity table.",
    )
    parser.add_argument(
        "--normalization",
        default="standardize",
        choices=["standardize", "l2"],
        help="Feature normalization before cosine kNN.",
    )
    parser.add_argument(
        "--pca-components",
        default=None,
        type=int,
        help="Optional PCA dimension for the main table. Omit or set 0 for raw features.",
    )
    parser.add_argument(
        "--clip-pooling",
        default="mean",
        choices=["mean", "mean_std", "flatten"],
        help="Pooling for sequence-shaped embeddings. Already pooled 2D embeddings are unchanged.",
    )
    parser.add_argument(
        "--clip-pool-axis",
        default=1,
        type=int,
        help="Sequence/time axis for clip pooling when embeddings are not 2D.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--robustness-k", default=10, type=int)
    parser.add_argument(
        "--robustness-pca-components",
        nargs="*",
        default=[64, 128, 256],
        type=int,
        help="PCA dimensions for robustness table. Pass the flag with no values for Raw only.",
    )
    parser.add_argument("--skip-robustness", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    embedding_paths = _discover_embedding_paths(args)
    output_dir = args.output_dir or _default_output_dir(args, embedding_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_results = []
    long_records: list[dict[str, Any]] = []
    for embedding_path in embedding_paths:
        embeddings, metadata = _load_embedding_bundle_for_eval(embedding_path, args)
        bundle = _infer_bundle_info(embedding_path)
        result = compute_knn_purity(
            embeddings,
            metadata,
            label_col=args.label_col,
            ks=args.ks,
            normalization=args.normalization,
            pca_components=args.pca_components,
            clip_pooling=args.clip_pooling,
            clip_pool_axis=args.clip_pool_axis,
            seed=args.seed,
        )
        main_results.append({"bundle": bundle, "path": embedding_path, "result": result})
        long_records.extend(_result_records("main", bundle, embedding_path, result))

        if not args.skip_pitch_source:
            auxiliary = _compute_pitch_source_purity(
                embeddings=embeddings,
                metadata=metadata,
                args=args,
            )
            main_results[-1]["auxiliary"] = auxiliary
            for name, aux_result in auxiliary.items():
                long_records.extend(
                    _result_records(
                        f"{name}_main",
                        bundle,
                        embedding_path,
                        aux_result,
                    )
                )

    robustness_results = []
    if not args.skip_robustness:
        pca_values = _dedupe_pca_values([None, *args.robustness_pca_components])
        for embedding_path in embedding_paths:
            embeddings, metadata = _load_embedding_bundle_for_eval(embedding_path, args)
            bundle = _infer_bundle_info(embedding_path)
            variants = {}
            for pca_components in pca_values:
                result = compute_knn_purity(
                    embeddings,
                    metadata,
                    label_col=args.label_col,
                    ks=[args.robustness_k],
                    normalization=args.normalization,
                    pca_components=pca_components,
                    clip_pooling=args.clip_pooling,
                    clip_pool_axis=args.clip_pool_axis,
                    seed=args.seed,
                )
                variant_key = _pca_label(pca_components)
                variants[variant_key] = result
                long_records.extend(
                    _result_records(
                        "robustness",
                        bundle,
                        embedding_path,
                        result,
                        pca_label=variant_key,
                    )
                )
            robustness_results.append(
                {"bundle": bundle, "path": embedding_path, "variants": variants}
            )

    main_table = _build_main_table(main_results, args.ks, args.pitch_source_k)
    robustness_table = _build_robustness_table(robustness_results, args.robustness_k)
    long_df = pd.DataFrame(long_records).sort_values(
        ["section", "sort_order", "pca_label", "k"],
        ignore_index=True,
    )

    main_csv = output_dir / "knn_purity_summary.csv"
    robustness_csv = output_dir / "knn_purity_robustness.csv"
    long_csv = output_dir / "knn_purity_long.csv"
    json_path = output_dir / "knn_purity.json"
    html_path = output_dir / "knn_purity_report.html"

    main_table.to_csv(main_csv, index=False)
    robustness_table.to_csv(robustness_csv, index=False)
    long_df.to_csv(long_csv, index=False)
    json_path.write_text(
        json.dumps(
            _json_payload(args, main_results, robustness_results),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_html_report(
        html_path,
        main_results=main_results,
        robustness_results=robustness_results,
        ks=args.ks,
        robustness_k=args.robustness_k,
        label_col=args.label_col,
        pitch_label_col=args.pitch_label_col,
        source_label_col=args.source_label_col,
        pitch_source_k=args.pitch_source_k,
        normalization=args.normalization,
        main_pca=args.pca_components,
    )

    print(_markdown_main_table(main_results, args.ks, args.pitch_source_k))
    if robustness_results:
        print()
        print(_markdown_robustness_table(robustness_results, args.robustness_k))
    print(f"\nSaved kNN purity outputs under {output_dir}")


def _load_embedding_bundle_for_eval(
    embedding_path: Path,
    args: argparse.Namespace,
) -> tuple[Any, pd.DataFrame]:
    embeddings, metadata = load_embedding_bundle(embedding_path)
    if not args.pitch_stratified:
        return embeddings, metadata

    sampled_metadata = pitch_stratified_sample(
        metadata,
        label_col=args.sample_family_col,
        pitch_col=args.pitch_label_col,
        max_per_label=args.max_per_family,
        max_per_pitch=args.max_per_pitch,
        pitch_bin_size=args.pitch_bin_size,
        require_all_labels_per_pitch=not args.keep_incomplete_pitch_strata,
        seed=args.seed,
        reset_index=False,
    )
    row_indices = sampled_metadata.index.to_numpy()
    return embeddings[row_indices], sampled_metadata.reset_index(drop=True)


def _discover_embedding_paths(args: argparse.Namespace) -> list[Path]:
    if args.embeddings:
        paths = [path.resolve() for path in args.embeddings]
    else:
        paths = sorted(path.resolve() for path in args.input_dir.glob(args.pattern))
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_label = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing embedding files: {missing_label}")
    if not paths:
        raise FileNotFoundError("No embeddings.npz files matched the requested input.")
    return sorted(paths, key=lambda path: _infer_bundle_info(path)["sort_order"])


def _default_output_dir(args: argparse.Namespace, embedding_paths: list[Path]) -> Path:
    if args.input_dir:
        return args.input_dir
    common_parent = embedding_paths[0].parent if len(embedding_paths) == 1 else Path("outputs")
    return common_parent / "knn_purity"


def _infer_bundle_info(embedding_path: Path) -> dict[str, Any]:
    name = embedding_path.parent.name
    lower = name.lower()
    model = name
    layer: str | int = "-"

    if lower.startswith("same") or "same_" in lower or "same-" in lower:
        if "same_l" in lower or "same-l" in lower or lower.endswith("_l") or lower.endswith("-l"):
            model = "SAME-L"
        elif "same_s" in lower or "same-s" in lower or lower.endswith("_s") or lower.endswith("-s"):
            model = "SAME-S"
        else:
            model = "SAME"
    elif "mert" in lower:
        model = "MERT"
        match = re.search(r"(?:layer|l)[_-]?(-?\d+)", lower)
        if match:
            layer = int(match.group(1))

    row_label = model if layer == "-" else f"{model} L{layer}"
    return {
        "model": model,
        "layer": layer,
        "row_label": row_label,
        "source_name": name,
        "sort_order": _sort_order(model, layer, name),
    }


def _sort_order(model: str, layer: str | int, fallback: str) -> str:
    model_rank = {
        "SAME-S": "00",
        "SAME-L": "01",
        "SAME": "02",
        "MERT": "10",
    }.get(model, "99")
    if isinstance(layer, int):
        layer_rank = f"{layer + 1000:04d}"
    else:
        layer_rank = "0000"
    return f"{model_rank}-{layer_rank}-{fallback}"


def _dedupe_pca_values(values: list[int | None]) -> list[int | None]:
    seen = set()
    deduped = []
    for value in values:
        normalized = None if value is None or int(value) <= 0 else int(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _compute_pitch_source_purity(
    embeddings,
    metadata: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    results = {}
    for name, label_col in [
        ("pitch", args.pitch_label_col),
        ("source", args.source_label_col),
    ]:
        if label_col not in metadata.columns:
            continue
        results[name] = compute_knn_purity(
            embeddings,
            metadata,
            label_col=label_col,
            ks=[args.pitch_source_k],
            normalization=args.normalization,
            pca_components=args.pca_components,
            clip_pooling=args.clip_pooling,
            clip_pool_axis=args.clip_pool_axis,
            seed=args.seed,
        )
    return results


def _result_records(
    section: str,
    bundle: dict[str, Any],
    embedding_path: Path,
    result: dict[str, Any],
    pca_label: str | None = None,
) -> list[dict[str, Any]]:
    records = []
    for k, values in result["ks"].items():
        records.append(
            {
                "section": section,
                "model": bundle["model"],
                "layer": bundle["layer"],
                "row_label": bundle["row_label"],
                "source_name": bundle["source_name"],
                "sort_order": bundle["sort_order"],
                "embedding_path": str(embedding_path),
                "label_col": result["label_col"],
                "normalization": result["normalization"],
                "pca_label": pca_label or _pca_label(result["pca_components"]),
                "pca_components": result["pca_components"],
                "effective_pca_components": result["effective_pca_components"],
                "clip_pooling": result["clip_pooling"],
                "clip_pool_axis": result["clip_pool_axis"],
                "num_items": result["num_items"],
                "num_classes": result["num_classes"],
                "k": int(k),
                "purity": values["purity"],
                "random_baseline": values["random_baseline"],
                "delta_vs_random": values["delta_vs_random"],
            }
        )
    return records


def _build_main_table(
    main_results: list[dict[str, Any]],
    ks: list[int],
    pitch_source_k: int,
) -> pd.DataFrame:
    rows = []
    for item in sorted(main_results, key=lambda value: value["bundle"]["sort_order"]):
        result = item["result"]
        row: dict[str, Any] = {"Model": item["bundle"]["model"], "Layer": item["bundle"]["layer"]}
        for k in ks:
            values = result["ks"][str(k)]
            row[f"Timbre Purity@{k}"] = values["purity"]
            row[f"Random@{k}"] = values["random_baseline"]
            row[f"Delta@{k}"] = values["delta_vs_random"]
        auxiliary = item.get("auxiliary", {})
        _insert_auxiliary_columns(row, auxiliary, "pitch", "Pitch", pitch_source_k)
        _insert_auxiliary_columns(row, auxiliary, "source", "Source", pitch_source_k)
        rows.append(row)
    return pd.DataFrame(rows)


def _insert_auxiliary_columns(
    row: dict[str, Any],
    auxiliary: dict[str, dict[str, Any]],
    key: str,
    display_name: str,
    default_k: int,
) -> None:
    result = auxiliary.get(key)
    if not result:
        row[f"{display_name} Purity@{default_k}"] = ""
        return
    k = next(iter(result["ks"]))
    values = result["ks"][k]
    row[f"{display_name} Purity@{k}"] = values["purity"]
    row[f"{display_name} Random@{k}"] = values["random_baseline"]
    row[f"{display_name} Delta@{k}"] = values["delta_vs_random"]


def _build_robustness_table(
    robustness_results: list[dict[str, Any]],
    robustness_k: int,
) -> pd.DataFrame:
    rows = []
    for item in sorted(robustness_results, key=lambda value: value["bundle"]["sort_order"]):
        row: dict[str, Any] = {"Model": f"{item['bundle']['row_label']} Purity@{robustness_k}"}
        for pca_label, result in item["variants"].items():
            row[pca_label] = result["ks"][str(robustness_k)]["purity"]
        rows.append(row)
    return pd.DataFrame(rows)


def _json_payload(
    args: argparse.Namespace,
    main_results: list[dict[str, Any]],
    robustness_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "label_col": args.label_col,
        "ks": [int(k) for k in args.ks],
        "normalization": args.normalization,
        "pca_components": args.pca_components,
        "pitch_label_col": args.pitch_label_col,
        "source_label_col": args.source_label_col,
        "pitch_source_k": args.pitch_source_k,
        "clip_pooling": args.clip_pooling,
        "clip_pool_axis": args.clip_pool_axis,
        "main": [
            {
                "bundle": item["bundle"],
                "embedding_path": str(item["path"]),
                "result": item["result"],
                "auxiliary": item.get("auxiliary", {}),
            }
            for item in main_results
        ],
        "robustness": [
            {
                "bundle": item["bundle"],
                "embedding_path": str(item["path"]),
                "variants": item["variants"],
            }
            for item in robustness_results
        ],
    }


def _write_html_report(
    output_path: Path,
    main_results: list[dict[str, Any]],
    robustness_results: list[dict[str, Any]],
    ks: list[int],
    robustness_k: int,
    label_col: str,
    pitch_label_col: str,
    source_label_col: str,
    pitch_source_k: int,
    normalization: str,
    main_pca: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    main_pca_label = _pca_label(main_pca)
    rows = [
        _html_main_row(item, ks)
        for item in sorted(main_results, key=lambda x: x["bundle"]["sort_order"])
    ]
    robustness_rows = [
        _html_robustness_row(item, robustness_k)
        for item in sorted(robustness_results, key=lambda x: x["bundle"]["sort_order"])
    ]
    robustness_headers = []
    if robustness_results:
        robustness_headers = list(robustness_results[0]["variants"].keys())

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>kNN Purity Report</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #20242a;
      --muted: #667085;
      --line: #d6dbe3;
      --head: #eef3f8;
      --accent: #1f7a8c;
      --good: #176b45;
      --paper: #ffffff;
      --page: #f5f7fa;
    }}
    body {{
      margin: 0;
      background: var(--page);
      color: var(--ink);
      font-family: Inter, Segoe UI, Arial, sans-serif;
      line-height: 1.45;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 44px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 28px 0 10px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      margin: 0 0 18px;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      background: var(--head);
      color: #394150;
      font-size: 13px;
      font-weight: 700;
    }}
    td:first-child, th:first-child {{
      text-align: left;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .score {{
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    .delta {{
      display: block;
      margin-top: 2px;
      color: var(--good);
      font-size: 12px;
    }}
    .empty {{
      color: var(--muted);
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      margin: 8px 0 0;
    }}
  </style>
</head>
<body>
<main>
  <h1>kNN Purity Report</h1>
  <p class="meta">Label: {escape(label_col)} | normalization: {escape(normalization)} |
  main PCA: {escape(main_pca_label)} | distance: cosine |
  random baseline: exact label-frequency expectation.<br />
  Pitch label: {escape(pitch_label_col)} | source label: {escape(source_label_col)}.</p>

  <h2>Purity Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        <th>Layer</th>
        {''.join(f'<th>Timbre Purity@{int(k)}</th>' for k in ks)}
        <th>Pitch Purity@{int(pitch_source_k)}</th>
        <th>Source Purity@{int(pitch_source_k)}</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <p class="note">Each score cell shows purity, then delta versus the random baseline.</p>

  <h2>Robustness</h2>
  <table>
    <thead>
      <tr>
        <th>Model</th>
        {''.join(f'<th>{escape(header)}</th>' for header in robustness_headers)}
      </tr>
    </thead>
    <tbody>
      {''.join(robustness_rows)}
    </tbody>
  </table>
</main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _html_main_row(item: dict[str, Any], ks: list[int]) -> str:
    bundle = item["bundle"]
    result = item["result"]
    cells = [
        f"<td>{escape(str(bundle['model']))}</td>",
        f"<td>{escape(str(bundle['layer']))}</td>",
    ]
    for k in ks:
        values = result["ks"][str(k)]
        cells.append(f"<td>{_score_html(values['purity'], values['delta_vs_random'])}</td>")
    cells.append(f"<td>{_aux_score_html(item, 'pitch')}</td>")
    cells.append(f"<td>{_aux_score_html(item, 'source')}</td>")
    return f"<tr>{''.join(cells)}</tr>"


def _aux_score_html(item: dict[str, Any], key: str) -> str:
    result = item.get("auxiliary", {}).get(key)
    if not result:
        return '<span class="empty"></span>'
    k = next(iter(result["ks"]))
    values = result["ks"][k]
    return _score_html(values["purity"], values["delta_vs_random"])


def _html_robustness_row(item: dict[str, Any], robustness_k: int) -> str:
    cells = [f"<td>{escape(str(item['bundle']['row_label']))} Purity@{robustness_k}</td>"]
    for result in item["variants"].values():
        values = result["ks"][str(robustness_k)]
        cells.append(f"<td>{_score_html(values['purity'], values['delta_vs_random'])}</td>")
    return f"<tr>{''.join(cells)}</tr>"


def _score_html(value: float, delta: float) -> str:
    return (
        f'<span class="score">{value:.3f}</span>'
        f'<span class="delta">{delta:+.3f} vs random</span>'
    )


def _markdown_main_table(
    main_results: list[dict[str, Any]],
    ks: list[int],
    pitch_source_k: int,
) -> str:
    headers = [
        "Model",
        "Layer",
        *[f"Timbre Purity@{int(k)}" for k in ks],
        f"Pitch Purity@{int(pitch_source_k)}",
        f"Source Purity@{int(pitch_source_k)}",
    ]
    rows = []
    for item in sorted(main_results, key=lambda value: value["bundle"]["sort_order"]):
        row = [str(item["bundle"]["model"]), str(item["bundle"]["layer"])]
        for k in ks:
            values = item["result"]["ks"][str(k)]
            row.append(f"{values['purity']:.3f} ({values['delta_vs_random']:+.3f})")
        row.append(_aux_markdown_score(item, "pitch"))
        row.append(_aux_markdown_score(item, "source"))
        rows.append(row)
    return _markdown_table(headers, rows)


def _aux_markdown_score(item: dict[str, Any], key: str) -> str:
    result = item.get("auxiliary", {}).get(key)
    if not result:
        return ""
    k = next(iter(result["ks"]))
    values = result["ks"][k]
    return f"{values['purity']:.3f} ({values['delta_vs_random']:+.3f})"


def _markdown_robustness_table(robustness_results: list[dict[str, Any]], robustness_k: int) -> str:
    if not robustness_results:
        return ""
    headers = ["Model", *robustness_results[0]["variants"].keys()]
    rows = []
    for item in sorted(robustness_results, key=lambda value: value["bundle"]["sort_order"]):
        row = [f"{item['bundle']['row_label']} Purity@{robustness_k}"]
        for result in item["variants"].values():
            values = result["ks"][str(robustness_k)]
            row.append(f"{values['purity']:.3f} ({values['delta_vs_random']:+.3f})")
        rows.append(row)
    return _markdown_table(headers, rows)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return "\n".join(output)


def _pca_label(pca_components: int | None) -> str:
    if pca_components is None or int(pca_components) <= 0:
        return "Raw"
    return f"PCA-{int(pca_components)}"


if __name__ == "__main__":
    main()
