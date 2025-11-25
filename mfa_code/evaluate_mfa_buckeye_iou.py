#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
from textgrid import TextGrid
from collections import defaultdict, Counter

# ===================
# CONFIG
# ===================
# Adjust these paths to point to your Buckeye gold annotations and MFA TextGrids
GOLD_DIR = "/Users/goku/Work/Fall_25/IDL/Project/Buckeye"        # .phones (Buckeye) or .TextGrid per utt
HYP_DIR  = "/Users/goku/Work/Fall_25/IDL/Project/Buckeye-Alignments"  # MFA-produced TextGrids
SR = None  # Buckeye .phones already use seconds in timestamps
TIER_NAME = "phones"
TOLERANCES_MS = [30, 50, 70]


# ===================
# LOAD FUNCTIONS
# ===================

def load_buckeye_phones(path):
    """Load Buckeye .phones file where each data line contains a timestamp and a label.

    The file has header lines then rows like:
      41.069375  122 g
    where the first column is time in seconds (start of label). We convert these to
    intervals by pairing consecutive timestamps: interval i = (t_i, t_{i+1}, label_i).
    The last timestamp has no explicit end and will be skipped.
    """
    times = []
    labels = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # data lines typically start with a digit (time in seconds)
            if not re.match(r"^\s*[0-9]+\.?[0-9]*", line):
                continue
            parts = line.split()
            # first token is time (seconds), last token is label (there may be an extra numeric field)
            t_str = parts[0]
            label_raw = parts[-1]
            # clean label: remove trailing semicolons, stars and surrounding braces
            label = label_raw.strip().strip(';*').strip('{}')
            try:
                t = float(t_str) * 1000.0  # convert seconds -> ms
            except Exception:
                continue
            times.append(t)
            labels.append(label)

    intervals = []
    for i in range(len(times) - 1):
        s = times[i]
        e = times[i+1]
        l = labels[i]
        # skip empty labels
        if l is None or l == "":
            continue
        intervals.append((s, e, l))
    return intervals


def load_textgrid(path, tier_name="phones"):
    # Try using the textgrid library first (if available and working)
    try:
        tg = TextGrid().read(path)
        if tg is not None:
            for tier in tg.tiers:
                if tier.name.lower() == tier_name.lower():
                    return [(i.minTime*1000.0, i.maxTime*1000.0, i.mark.strip()) for i in tier.intervals if i.mark.strip()]
    except Exception:
        tg = None

    # Fallback: simple, robust parser for Praat TextGrid (ASCII)
    intervals = []
    cur_tier = None
    in_target = False
    cur_interval = {}
    # regex helpers
    import re
    name_re = re.compile(r"name\s*=\s*\"(?P<name>.+?)\"")
    xmin_re = re.compile(r"xmin\s*=\s*(?P<x>[-0-9\.eE]+)")
    xmax_re = re.compile(r"xmax\s*=\s*(?P<x>[-0-9\.eE]+)")
    text_re = re.compile(r"text\s*=\s*\"(?P<t>.*)\"")

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = name_re.search(line)
            if m:
                cur_tier = m.group("name").strip()
                in_target = (cur_tier.lower() == tier_name.lower())
                continue

            if not in_target:
                continue

            # inside the target tier, look for intervals
            m = xmin_re.search(line)
            if m:
                cur_interval["xmin"] = float(m.group("x")) * 1000.0
                continue
            m = xmax_re.search(line)
            if m:
                cur_interval["xmax"] = float(m.group("x")) * 1000.0
                continue
            m = text_re.search(line)
            if m:
                txt = m.group("t").strip()
                if txt:
                    cur_interval["text"] = txt
                else:
                    cur_interval["text"] = ""
                # when we have xmin, xmax, text we can append
                if "xmin" in cur_interval and "xmax" in cur_interval and "text" in cur_interval:
                    if cur_interval["text"]:
                        intervals.append((cur_interval["xmin"], cur_interval["xmax"], cur_interval["text"]))
                    cur_interval = {}

    if intervals:
        return intervals

    raise ValueError(f"Tier {tier_name} not found or empty in {path}")


def load_gold(path):
    if path.lower().endswith(".phones"):
        return load_buckeye_phones(path)
    elif path.lower().endswith(".textgrid") or path.lower().endswith(".textgrid"):
        return load_textgrid(path, TIER_NAME)
    else:
        raise ValueError(f"Unsupported gold file format: {path}")


def collapse_adjacent(intervals):
    if not intervals: return intervals
    out = []
    prev_s, prev_e, prev_l = intervals[0]
    for s,e,l in intervals[1:]:
        if l == prev_l and abs(s - prev_e) < 1e-6:
            prev_e = e
        else:
            out.append((prev_s, prev_e, prev_l))
            prev_s, prev_e, prev_l = s,e,l
    out.append((prev_s, prev_e, prev_l))
    return out


def match_segments_by_sequence(gold, hyp):
    pairs = []
    gi = 0; hi = 0
    while gi < len(gold) and hi < len(hyp):
        gs, ge, gl = gold[gi]
        hs, he, hl = hyp[hi]
        if ge <= hs:
            gi += 1
            continue
        if he <= gs:
            hi += 1
            continue
        pairs.append(((gs,ge,gl),(hs,he,hl)))
        if ge <= he:
            gi += 1
        else:
            hi += 1
    return pairs


# ===================
# METRICS
# ===================

def intersection_union(gs, ge, hs, he):
    inter = max(0, min(ge, he) - max(gs, hs))
    union = max(0, max(ge, he) - min(gs, hs))
    return inter, union


def evaluate_file(gold, hyp):
    pairs = match_segments_by_sequence(gold, hyp)

    start_errs = []
    end_errs = []
    ious = []
    conf = Counter()

    for (gs,ge,gl),(hs,he,hl) in pairs:
        start_errs.append(abs(gs - hs))
        end_errs.append(abs(ge - he))
        inter, union = intersection_union(gs,ge,hs,he)
        iou = inter / union if union > 0 else 0
        ious.append(iou)
        conf[(gl,hl)] += inter

    mean_iou = np.mean(ious) if ious else np.nan
    mean_start = np.mean(start_errs) if start_errs else np.nan
    mean_end   = np.mean(end_errs) if end_errs else np.nan

    # Tolerance accuracy
    gold_bounds = np.array([b for seg in gold for b in seg[:2]])
    hyp_bounds  = np.array([b for seg in hyp for b in seg[:2]])
    tol_acc = {}
    if len(gold_bounds) and len(hyp_bounds):
        nearest = np.abs(gold_bounds.reshape(-1,1) - hyp_bounds.reshape(1,-1)).min(axis=1)
        for T in TOLERANCES_MS:
            tol_acc[T] = (nearest <= T).mean()
    else:
        for T in TOLERANCES_MS:
            tol_acc[T] = np.nan

    return {
        "mean_start_err": mean_start,
        "mean_end_err": mean_end,
        "mean_iou": mean_iou,
        **{f"tol_{T}ms": tol_acc[T] for T in TOLERANCES_MS},
        "conf": conf,
        "ious": ious,
        "pairs": pairs
    }


# ===================
# MAIN
# ===================

def find_gold_and_hyp_pairs(gold_dir, hyp_dir):
    # collect gold files (.phones and .TextGrid)
    gold_files = glob.glob(os.path.join(gold_dir, "*.phones")) + glob.glob(os.path.join(gold_dir, "*.TextGrid")) + glob.glob(os.path.join(gold_dir, "*.textgrid"))
    gold_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gold_files}

    # collect hypothesis TextGrids (case insensitive extensions)
    hyp_files = glob.glob(os.path.join(hyp_dir, "*.TextGrid")) + glob.glob(os.path.join(hyp_dir, "*.textgrid"))
    hyp_map = {os.path.splitext(os.path.basename(p))[0]: p for p in hyp_files}

    common = sorted(set(gold_map.keys()) & set(hyp_map.keys()))
    gold_list = [gold_map[k] for k in common]
    hyp_list  = [hyp_map[k]  for k in common]
    return gold_list, hyp_list


def main():
    gold_files, hyp_files = find_gold_and_hyp_pairs(GOLD_DIR, HYP_DIR)
    print(f"Found {len(gold_files)} matching gold/hyp file pairs")

    all_results = []
    per_phone_iou = defaultdict(list)

    for gf, hf in zip(gold_files, hyp_files):
        try:
            gold = collapse_adjacent(load_gold(gf))
        except Exception as e:
            print("⚠️ Failed loading gold:", gf, e)
            continue
        try:
            hyp = collapse_adjacent(load_textgrid(hf, TIER_NAME))
        except Exception as e:
            print("⚠️ Failed loading hyp:", hf, e)
            continue

        res = evaluate_file(gold, hyp)
        all_results.append(res)

        for (gseg,hseg) in res["pairs"]:
            gs,ge,gl = gseg
            hs,he,hl = hseg
            inter, union = intersection_union(gs,ge,hs,he)
            iou = inter / union if union>0 else 0
            per_phone_iou[gl].append(iou)

    df = pd.DataFrame([{
        "mean_start_err": r["mean_start_err"],
        "mean_end_err": r["mean_end_err"],
        "mean_iou": r["mean_iou"],
        **{f"tol_{T}ms": r[f"tol_{T}ms"] for T in TOLERANCES_MS}
    } for r in all_results])

    print("Evaluated files:", len(df))
    if not df.empty:
        print(df[["mean_iou","mean_start_err","mean_end_err"] + [f"tol_{T}ms" for T in TOLERANCES_MS]].mean())

    phone_iou_df = pd.DataFrame([
        {"phone": ph, "mean_iou": np.mean(vals), "median_iou": np.median(vals), "count": len(vals)}
        for ph, vals in per_phone_iou.items() if len(vals)>5
    ]).sort_values("mean_iou")

    os.makedirs("eval_results", exist_ok=True)
    df.to_csv("eval_results/filewise_metrics_buckeye.csv", index=False)
    phone_iou_df.to_csv("eval_results/per_phone_iou_buckeye.csv", index=False)

    print("\nTop 10 phones by IoU:")
    print(phone_iou_df.tail(10))
    print("\nLowest 10 phones by IoU:")
    print(phone_iou_df.head(10))


if __name__ == "__main__":
    main()
