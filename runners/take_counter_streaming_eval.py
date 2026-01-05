from __future__ import annotations

import json
from dataclasses import asdict, datac
lass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from detection.sam3_transformers_streaming import Sam3TransformersStreamingEngine
from runners.take_counter_streaming import StreamingRunConfig, run_streaming_take_counter_on_video


@dataclass(frozen=True)
class EvalItem:
    video_path: str
    true_positive: bool
    take_count: Optional[int]
    predicted_positive: Optional[bool]
    summary_path: Optional[str]
    debug_video_path: Optional[str]
    error: Optional[str]


@dataclass(frozen=True)
class EvalSummary:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    num_videos: int
    num_errors: int


def _iter_videos(dir_path: Path) -> List[Path]:
    exts = {".mp4", ".MP4"}
    videos = [p for p in dir_path.rglob("*") if p.is_file() and p.suffix in exts]
    return sorted(videos)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def evaluate_streaming_take_counter(
    *,
    repo_root: Path,
    neg_dir: Path,
    pos_dir: Path,
    out_dir: Path,
    cfg: StreamingRunConfig,
    positive_threshold: int = 2,
    stop_on_error: bool = False,
) -> Tuple[EvalSummary, List[EvalItem]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    items: List[EvalItem] = []

    def _run_group(videos: Iterable[Path], true_positive: bool, group_name: str) -> None:
        nonlocal items
        group_out = out_dir / group_name
        group_out.mkdir(parents=True, exist_ok=True)
        for video_path in videos:
            try:
                result = run_streaming_take_counter_on_video(
                    video_path=str(video_path),
                    out_dir=group_out,
                    cfg=cfg,
                    engine=engine,
                )
                predicted_positive = bool(result.take_count >= positive_threshold)
                items.append(
                    EvalItem(
                        video_path=str(video_path),
                        true_positive=true_positive,
                        take_count=int(result.take_count),
                        predicted_positive=predicted_positive,
                        summary_path=result.summary_path,
                        debug_video_path=result.debug_video_path,
                        error=None,
                    )
                )
                print(
                    f"[{group_name}] {video_path.name}: take_count={result.take_count} pred={predicted_positive}"
                )
            except Exception as e:
                items.append(
                    EvalItem(
                        video_path=str(video_path),
                        true_positive=true_positive,
                        take_count=None,
                        predicted_positive=None,
                        summary_path=None,
                        debug_video_path=None,
                        error=str(e),
                    )
                )
                print(f"[{group_name}] {video_path.name}: ERROR: {e}")
                if stop_on_error:
                    raise

    if not neg_dir.exists():
        raise FileNotFoundError(f"Negative directory not found: {neg_dir}")
    if not pos_dir.exists():
        raise FileNotFoundError(f"Positive directory not found: {pos_dir}")

    neg_videos = _iter_videos(neg_dir)
    pos_videos = _iter_videos(pos_dir)
    print(f"[dataset] neg={len(neg_videos)} videos @ {neg_dir}")
    print(f"[dataset] pos={len(pos_videos)} videos @ {pos_dir}")

    if len(neg_videos) == 0 and len(pos_videos) == 0:
        raise RuntimeError(
            "No videos found in either dataset directory. "
            "Check that the folders contain .mp4/.MP4 files on this machine."
        )

    import torch

    engine = Sam3TransformersStreamingEngine(
        model_name="facebook/sam3",
        dtype=torch.bfloat16,
        model_image_size=cfg.model_image_size,
    )

    _run_group(neg_videos, true_positive=False, group_name="neg_1_or_0")
    _run_group(pos_videos, true_positive=True, group_name="pos_2_or_more")

    tp = fp = fn = tn = 0
    num_errors = 0
    for it in items:
        if it.error is not None:
            num_errors += 1
            continue
        assert it.predicted_positive is not None and it.true_positive is not None
        if it.true_positive and it.predicted_positive:
            tp += 1
        elif (not it.true_positive) and it.predicted_positive:
            fp += 1
        elif it.true_positive and (not it.predicted_positive):
            fn += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    summary = EvalSummary(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        num_videos=len(items),
        num_errors=num_errors,
    )

    summary_path = out_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

    results_path = out_dir / "eval_items.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")

    print(f"[metrics] precision={summary.precision:.3f} recall={summary.recall:.3f} f1={summary.f1:.3f}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {results_path}")

    return summary, items


def run_default_eval() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    neg_dir = repo_root / "data" / "가져감(1개)_화각변환"
    pos_dir = repo_root / "data" / "가져감(2개이상)_화각변환"
    out_dir = repo_root / "logs" / "streaming_eval_take_counter"

    cfg = StreamingRunConfig(
        prompts=["objects in refrigerator"],
        foods_prompt="objects in refrigerator",
        target_fps=15.0,
        resize_width=1280,
        resize_height=720,
        mask_input_outside_roi=True,
        processing_device="cpu",
        video_storage_device="cpu",
        model_image_size=None,
        alpha=0.45,
    )

    evaluate_streaming_take_counter(
        repo_root=repo_root,
        neg_dir=neg_dir,
        pos_dir=pos_dir,
        out_dir=out_dir,
        cfg=cfg,
        positive_threshold=2,
        stop_on_error=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_default_eval())
