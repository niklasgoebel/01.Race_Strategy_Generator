from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class EffortBlocksConfig:
    # runnable shorter than this does NOT break a climb block
    max_runnable_gap_km: float = 0.50

    # ignore tiny climb blocks
    min_block_gain_m: float = 80.0
    min_block_dist_km: float = 0.8


def calculate_effort_cost(block: Dict[str, Any]) -> float:
    """
    Calculate a single "effort cost" metric for a climb block.
    
    Combines distance, gain, and gradient into a single difficulty score
    that the LLM can use for reasoning about pacing and fatigue.
    
    Formula:
    - Base cost = gain_m / 100 (100m gain = 1 unit)
    - Gradient multiplier = 1 + (avg_gradient / 10) (steeper = harder)
    - Distance fatigue = 1 + (distance_km / 5) (longer = more fatigue)
    
    Returns:
        Float representing relative difficulty (higher = harder)
    """
    base_cost = float(block.get("gain_m", 0)) / 100.0
    gradient_multiplier = 1.0 + (float(block.get("avg_gradient_pct", 0)) / 10.0)
    distance_fatigue = 1.0 + (float(block.get("distance_km", 0)) / 5.0)
    
    return base_cost * gradient_multiplier * distance_fatigue


def build_climb_blocks(seg: pd.DataFrame, cfg: EffortBlocksConfig = EffortBlocksConfig()) -> pd.DataFrame:
    """
    Build higher-level "climb blocks" from the segment table.

    A climb block is a sustained climbing effort that may contain short runnable interruptions.
    Rule:
      - start when we hit a 'climb'
      - allow 'runnable' segments as long as they are <= max_runnable_gap_km
      - stop when we hit a 'descent' OR a runnable > max_runnable_gap_km
    """
    if seg is None or seg.empty:
        return pd.DataFrame(
            columns=[
                "block_id",
                "start_km",
                "end_km",
                "distance_km",
                "gain_m",
                "loss_m",
                "avg_gradient_pct",
                "num_climb_segments",
                "num_runnable_gaps",
                "longest_runnable_gap_km",
                "effort_cost",
                "notes",
            ]
        )

    # defensive copy + ensure ordering
    s = seg.sort_values("start_km").reset_index(drop=True).copy()

    blocks: List[Dict[str, Any]] = []
    in_block = False

    # accumulators
    start_km: float = 0.0
    end_km: float = 0.0
    gain_m: float = 0.0
    loss_m: float = 0.0
    dist_km: float = 0.0
    climb_count: int = 0
    runnable_gaps: int = 0
    longest_gap: float = 0.0

    def close_block():
        nonlocal in_block, start_km, end_km, gain_m, loss_m, dist_km, climb_count, runnable_gaps, longest_gap

        if not in_block:
            return

        # basic metrics
        block_dist = float(end_km - start_km)
        block_gain = float(gain_m)

        if block_gain >= cfg.min_block_gain_m and block_dist >= cfg.min_block_dist_km:
            avg_grad = (block_gain / (block_dist * 1000.0)) * 100.0 if block_dist > 0 else 0.0
            
            block_data = {
                "block_id": len(blocks) + 1,
                "start_km": float(start_km),
                "end_km": float(end_km),
                "distance_km": float(block_dist),
                "gain_m": float(block_gain),
                "loss_m": float(loss_m),
                "avg_gradient_pct": float(avg_grad),
                "num_climb_segments": int(climb_count),
                "num_runnable_gaps": int(runnable_gaps),
                "longest_runnable_gap_km": float(longest_gap),
                "notes": "sustained_climb_with_runnable_gaps" if runnable_gaps > 0 else "sustained_climb",
            }
            
            # Calculate effort cost for this block
            block_data["effort_cost"] = calculate_effort_cost(block_data)
            
            blocks.append(block_data)

        # reset
        in_block = False
        start_km = end_km = 0.0
        gain_m = loss_m = dist_km = 0.0
        climb_count = runnable_gaps = 0
        longest_gap = 0.0

    for _, row in s.iterrows():
        typ = str(row.get("type", "")).strip()
        d = float(row.get("distance_km", 0.0) or 0.0)
        g = float(row.get("gain_m", 0.0) or 0.0)
        l = float(row.get("loss_m", 0.0) or 0.0)
        sk = float(row.get("start_km", 0.0) or 0.0)
        ek = float(row.get("end_km", 0.0) or 0.0)

        if not in_block:
            if typ == "climb":
                in_block = True
                start_km = sk
                end_km = ek
                dist_km += d
                gain_m += g
                loss_m += l
                climb_count += 1
            # ignore anything until a climb starts
            continue

        # if we are in a block:
        if typ == "climb":
            end_km = ek
            dist_km += d
            gain_m += g
            loss_m += l
            climb_count += 1
            continue

        if typ == "runnable":
            # keep runnable in the block only if it's short
            if d <= cfg.max_runnable_gap_km:
                end_km = ek
                dist_km += d
                gain_m += g
                loss_m += l
                runnable_gaps += 1
                longest_gap = max(longest_gap, d)
                continue
            else:
                # long runnable breaks the block
                close_block()
                continue

        # descent breaks the block
        if typ == "descent":
            close_block()
            continue

        # unknown type -> be conservative
        close_block()

    # finalize
    close_block()

    return pd.DataFrame(blocks)


def summarize_climb_blocks(blocks: pd.DataFrame, max_items: int = 10) -> List[str]:
    if blocks is None or blocks.empty:
        return []

    b = blocks.sort_values(["gain_m", "avg_gradient_pct"], ascending=[False, False]).head(max_items)
    out: List[str] = []
    for _, r in b.iterrows():
        out.append(
            f"Climb block #{int(r['block_id'])}: {r['start_km']:.1f}–{r['end_km']:.1f} km "
            f"({r['distance_km']:.1f} km), gain {r['gain_m']:.0f} m, "
            f"avg {r['avg_gradient_pct']:.1f}%, effort cost {r.get('effort_cost', 0):.1f}, "
            f"runnable gaps {int(r['num_runnable_gaps'])} "
            f"(max gap {r['longest_runnable_gap_km']:.2f} km)"
        )
    return out