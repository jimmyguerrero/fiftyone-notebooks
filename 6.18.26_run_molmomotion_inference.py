#!/usr/bin/env python3
"""
run_molmomotion_inference.py
============================

Produce the precomputed prediction files that
`MolmoMotion_PointMotionBench_FiftyOne_REAL.ipynb` loads.

For every PointMotionBench clip (filtered to the splits you ask for), this script:
  1. reads the clip's annotation (caption / action description, query points, GT 3D track),
  2. locates the reconstructed video,
  3. runs MolmoMotion to forecast the future 3D trajectory of the query points,
  4. writes  {OUT}/{clip_id}.npz  with arrays:
         ar : (T, N_QUERY, 3)   # MolmoMotion-AR prediction   (present when --variant ar/both)
         fm : (T, N_QUERY, 3)   # MolmoMotion-FM prediction   (present when --variant fm/both)
     plus bookkeeping arrays:  clip_id, split, query_init (N_QUERY,3), caption.

The notebook only requires `ar` (and optionally `fm`). Everything else is for traceability.

--------------------------------------------------------------------------------------------
WHAT IS VERIFIED vs. WHAT YOU MUST CONFIRM
--------------------------------------------------------------------------------------------
The file I/O, annotation parsing, clip lookup, query-point handling, horizon alignment, and the
exact .npz schema are implemented and unit-tested (see `--self-test`, which runs with a stub model
and needs no GPU or weights).

The ONE thing this script cannot guess is MolmoMotion's exact forward/generate signature, which
lives in the model repo. That call is isolated in `MolmoMotionRunner.forecast()` and marked
clearly. Read it against https://github.com/allenai/molmo-motion (backbone: allenai/molmo2) and
adjust the marked block to match the released API. Nothing else should need to change.

Weights / code:
  - checkpoints: allenai/MolmoMotion-4B-H3-F30  (3-frame history, paper's "3f")
                 allenai/MolmoMotion-4B-H1-F32  (1-frame history, "1f")
  - collection : https://huggingface.co/collections/allenai/molmomotion
  - code       : https://github.com/allenai/molmo-motion
  - backbone   : https://github.com/allenai/molmo2
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- annotation IO
# PointMotionBench real layout (confirmed against the released repo):
#   davis/davis_captions.json   : { clip: {"description": str} }
#   davis/tracks/{clip}_3d.npz  : points_3d -> 0-d obj -> { object_name: (F, N, 3) }
#   davis/tracks/{clip}_2d.npz  : tracks -> { object_name: (N, F, 2) }, dim -> [H, W]
# clip_id == DAVIS sequence name, matching the reconstructed {clip}.mp4.
N_QUERY_DEFAULT = 8


def _unwrap(npz, key):
    if key not in npz.files:
        return None
    a = npz[key]
    return a.item() if a.shape == () else a


def _first_object(d):
    if not isinstance(d, dict) or not d:
        return None, None
    name = next(iter(d))
    return name, np.asarray(d[name], dtype=np.float32)


def _subsample(n_total, k):
    if k is None or k >= n_total:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, k).round().astype(int)


def load_clips(pmb_root: Path, video_dirs: dict, splits: set, n_query: int = N_QUERY_DEFAULT):
    """Load PointMotionBench clips. Currently implements DAVIS (the released layout);
    add HOT3D/WorldTrack readers here as needed. Returns list of dicts with
    clip_id, split, video_path, caption, gt_3d (F,k,3), query_init (k,3)."""
    clips, skipped = [], 0
    if "DAVIS" in splits:
        davis_dir = Path(pmb_root) / "davis"
        tracks_dir = davis_dir / "tracks"
        videos_dir = Path(video_dirs.get("DAVIS", davis_dir / "videos" / "input_480p"))
        caps_path = davis_dir / "davis_captions.json"
        captions = json.loads(caps_path.read_text()) if caps_path.exists() else {}
        if not tracks_dir.exists():
            raise FileNotFoundError(f"DAVIS tracks not found at {tracks_dir}. Did the HF download finish?")
        for f3 in sorted(tracks_dir.glob("*_3d.npz")):
            clip = f3.name[:-len("_3d.npz")]
            vpath = videos_dir / f"{clip}.mp4"
            if not vpath.exists():
                skipped += 1
                continue
            obj3d = _unwrap(np.load(f3, allow_pickle=True), "points_3d")
            obj_name, p3 = _first_object(obj3d)            # (F, N, 3)
            if p3 is None:
                skipped += 1
                continue
            idx = _subsample(p3.shape[1], n_query)
            gt3d = p3[:, idx, :].astype(np.float32)        # (F, k, 3)
            cap = captions.get(clip, {})
            caption = cap.get("description", "") if isinstance(cap, dict) else str(cap)
            clips.append(dict(clip_id=clip, split="DAVIS", video_path=str(vpath),
                              caption=caption, gt_3d=gt3d, query_init=gt3d[0]))
    for other in ("HOT3D", "WorldTrack"):
        if other in splits:
            print(f"  ! {other} loader not implemented in this script yet — skipping.")
    return clips, skipped


# ----------------------------------------------------------------------------- model runner
class MolmoMotionRunner:
    """Wraps MolmoMotion. `stub=True` produces deterministic fake trajectories for --self-test."""

    def __init__(self, weights: str, variant: str, device: str = "cuda",
                 dtype: str = "bfloat16", stub: bool = False):
        self.variant = variant
        self.stub = stub
        if stub:
            self.model = self.processor = None
            return
        # ----- real load (HF-transformers path, matching Ai2 model cards) -----------------
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            weights, trust_remote_code=True, torch_dtype="auto", device_map=device)
        self.model = AutoModelForCausalLM.from_pretrained(
            weights, trust_remote_code=True, torch_dtype="auto", device_map=device)
        self.dtype = getattr(torch, dtype)

    # --------------------------------------------------------------------------------------
    # >>> THE ONE REPO-SPECIFIC BLOCK — confirm against github.com/allenai/molmo-motion <<<
    # --------------------------------------------------------------------------------------
    def forecast(self, video_path: str, caption: str, query_init: np.ndarray,
                 horizon: int) -> np.ndarray:
        """
        Return predicted future 3D trajectory of shape (horizon, N_QUERY, 3).

        Inputs:
          video_path : reconstructed clip (mp4, or npz with embedded frames for WorldTrack)
          caption    : the action description / instruction
          query_init : (N_QUERY, 3) initial 3D positions of the query points
          horizon    : number of future steps T to predict

        The released MolmoMotion repo exposes the actual call. Typical shape (pseudocode):

            frame0 = read_first_frame(video_path)            # RGB observation
            inputs = self.processor.process(
                images=[frame0], text=caption,
                query_points_3d=query_init, variant=self.variant)
            inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
            with self.torch.autocast(self.model.device.type, dtype=self.dtype):
                out = self.model.forecast_trajectory(inputs, horizon=horizon)   # repo-specific
            return out["points_3d"].squeeze(0).float().cpu().numpy()            # (horizon, N, 3)

        Replace the body below with the repo's real call. Until then this raises, so you can't
        accidentally ship random numbers as if they were model output.
        """
        if self.stub:
            # Deterministic synthetic motion — ONLY for --self-test. Not a real prediction.
            n = query_init.shape[0]
            t = np.linspace(0, 1, horizon)[:, None, None]
            drift = np.array([0.3, 0.05, 0.0], dtype=np.float32)
            jitter = (0.02 if self.variant == "ar" else 0.05)
            rs = np.random.RandomState(abs(hash(caption)) % (2**32))
            return (query_init[None] + t * drift
                    + rs.normal(0, jitter, (horizon, n, 3))).astype(np.float32)
        raise NotImplementedError(
            "Wire MolmoMotion's real forecast call here — see the docstring and "
            "https://github.com/allenai/molmo-motion")
    # --------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------- main driver
def infer_all(clips, runner_ar, runner_fm, out_dir: Path, horizon: int | None,
              n_query: int, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    for c in clips:
        out_path = out_dir / f"{c['clip_id']}.npz"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        q_init = c["query_init"]
        if q_init is None:
            print(f"  ! {c['clip_id']}: no query points in annotation — skipping")
            skipped += 1
            continue
        q_init = np.asarray(q_init, dtype=np.float32)[:n_query]

        # horizon: match the GT track length when available, else use --horizon
        T = horizon or (c["gt_3d"].shape[0] if c["gt_3d"] is not None else None)
        if T is None:
            print(f"  ! {c['clip_id']}: no horizon (no GT and no --horizon) — skipping")
            skipped += 1
            continue

        payload = dict(clip_id=c["clip_id"], split=c["split"],
                       caption=c["caption"], query_init=q_init)
        if runner_ar is not None:
            payload["ar"] = runner_ar.forecast(c["video_path"], c["caption"], q_init, T)
        if runner_fm is not None:
            payload["fm"] = runner_fm.forecast(c["video_path"], c["caption"], q_init, T)
        np.savez(out_path, **payload)
        written += 1
        if written % 10 == 0:
            print(f"  ...{written} written")
    return written, skipped


def build_video_dirs(args):
    return {
        "DAVIS":      args.davis_videos,
        "HOT3D":      args.hot3d_videos,
        "WorldTrack": args.worldtrack_videos,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    DATA_ROOT = Path(os.environ.get("POINTMOTION_DATA_ROOT", "~/pointmotion_data")).expanduser()
    p.add_argument("--pmb-root", type=Path, default=DATA_ROOT / "PointMotionBench")
    p.add_argument("--davis-videos", type=Path,
                   default=DATA_ROOT / "PointMotionBench" / "davis" / "videos" / "input_480p")
    p.add_argument("--hot3d-videos", type=Path,
                   default=DATA_ROOT / "PointMotionBench" / "hot3d" / "rgbs")
    p.add_argument("--worldtrack-videos", type=Path,
                   default=DATA_ROOT / "PointMotionBench" / "worldtrack")
    p.add_argument("--out", type=Path, default=DATA_ROOT / "molmomotion_predictions")
    p.add_argument("--weights", default="allenai/MolmoMotion-4B-H3-F30",
                   help="HF id or local path to MolmoMotion weights (see the HF collection)")
    p.add_argument("--weights-fm", default=None,
                   help="separate weights for the FM variant, if --variant both")
    p.add_argument("--variant", choices=["ar", "fm", "both"], default="ar")
    p.add_argument("--splits", nargs="+", default=["DAVIS"],
                   choices=["DAVIS", "HOT3D", "WorldTrack"])
    p.add_argument("--horizon", type=int, default=None,
                   help="future steps to predict; default = match each clip's GT length")
    p.add_argument("--n-query", type=int, default=N_QUERY_DEFAULT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="run the full pipeline on a tiny synthetic dataset with a STUB model "
                        "(no GPU, no weights) and verify the .npz schema")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    splits = set(args.splits)
    video_dirs = build_video_dirs(args)
    print(f"Loading clips from {args.pmb_root} (splits={sorted(splits)}) ...")
    clips, skipped = load_clips(args.pmb_root, video_dirs, splits)
    print(f"  {len(clips)} clips with matching videos ({skipped} annotations had no video)")
    if not clips:
        print("Nothing to do. Check that videos are reconstructed and clip ids match.")
        return 1

    runner_ar = runner_fm = None
    if args.variant in ("ar", "both"):
        print(f"Loading MolmoMotion-AR weights: {args.weights}")
        runner_ar = MolmoMotionRunner(args.weights, "ar", args.device, args.dtype)
    if args.variant in ("fm", "both"):
        w = args.weights_fm or args.weights
        print(f"Loading MolmoMotion-FM weights: {w}")
        runner_fm = MolmoMotionRunner(w, "fm", args.device, args.dtype)

    written, skipped = infer_all(clips, runner_ar, runner_fm, args.out,
                                 args.horizon, args.n_query, args.overwrite)
    print(f"Done. Wrote {written} prediction files to {args.out} (skipped {skipped}).")
    print(f"Next: set PREDICTIONS_DIR = {args.out} in the notebook and re-run preflight.")
    return 0


# ----------------------------------------------------------------------------- self test
def _self_test():
    import tempfile, numpy as np
    tmp = Path(tempfile.mkdtemp())
    davis = tmp / "PMB" / "davis"
    tracks = davis / "tracks"; tracks.mkdir(parents=True)
    vids = davis / "videos" / "input_480p"; vids.mkdir(parents=True)
    clips = {"bear": "brown_bear", "blackswan": "black_swan"}
    caps = {c: {"description": f"a {o.replace('_',' ')} moves"} for c, o in clips.items()}
    (davis / "davis_captions.json").write_text(json.dumps(caps))
    rs = np.random.RandomState(0)
    F, N = 97, 82
    for clip, obj in clips.items():
        (vids / f"{clip}.mp4").write_bytes(b"x")
        p3 = (rs.standard_normal((F, N, 3)) * 0.2).astype(np.float32)
        np.savez(tracks / f"{clip}_3d.npz", points_3d=np.array({obj: p3}, dtype=object))
        p2 = (rs.random((N, F, 2)) * np.array([854, 480])).astype(np.float32)
        np.savez(tracks / f"{clip}_2d.npz",
                 tracks=np.array({obj: p2}, dtype=object), dim=np.array([480, 854]))

    video_dirs = {"DAVIS": vids, "HOT3D": tmp / "h", "WorldTrack": tmp / "w"}
    clips_l, skipped = load_clips(tmp / "PMB", video_dirs, {"DAVIS"}, n_query=8)
    assert len(clips_l) == 2, f"expected 2 clips, got {len(clips_l)}"
    c0 = clips_l[0]
    assert c0["gt_3d"].shape == (F, 8, 3), c0["gt_3d"].shape
    assert c0["query_init"].shape == (8, 3)
    assert c0["caption"].startswith("a ")

    runner = MolmoMotionRunner("stub", "ar", stub=True)
    out = tmp / "preds"
    written, _ = infer_all(clips_l, runner, None, out, horizon=None, n_query=8, overwrite=True)
    assert written == 2, written
    d = np.load(out / f"{c0['clip_id']}.npz", allow_pickle=True)
    assert d["ar"].shape == (F, 8, 3), d["ar"].shape
    assert "fm" not in d.files
    w2, s2 = infer_all(clips_l, runner, None, out, horizon=None, n_query=8, overwrite=False)
    assert w2 == 0 and s2 == 2
    print(f"SELF-TEST PASSED — real DAVIS schema, 2 clips, GT (F,8,3)=({F},8,3), "
          f"npz ar ok, idempotent re-run ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
