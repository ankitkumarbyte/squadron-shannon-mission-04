"""
Course-portal simulator test (Week 4 -> 3. Simulation -> "Direct Preference Optimization & Activation Steering Lab").

The values shown in the portal screenshots in this folder were typed in here and recomputed with OUR modules
(DPOLoss from Part 1, SimPOLoss from Part 3). The steering tab only shows a canned example text, so its two
claims are compared with what Part 4 measured on Qwen2.5-0.5B-Instruct.

Run:  python docs/portal_simulator/verify_portal.py      -> prints the comparison, writes portal_results.json
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PS = HERE.parents[0] / "weeks" / "week-04" / "problem_set"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, PS / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


core = _load("dpo_core", "01_DPO_Loss_Core.py")
simpo = _load("simpo_mod", "03_Length_Bias_and_SimPO.py")

# what the portal displayed (screenshots 01-03)
PORTAL_DPO = [
    {"beta": 0.10, "pi_w": -12.4, "pi_l": -18.2, "ref_w": -15.1, "ref_l": -14.8,
     "shown": {"p_win": 0.648, "margin": 0.610, "loss": 0.4340, "grad_scale": 0.0352, "r_w": 0.270, "r_l": -0.340},
     "screenshot": "01_dpo_beta0.10_default.jpg"},
    {"beta": 0.14, "pi_w": -12.4, "pi_l": -18.2, "ref_w": -15.1, "ref_l": -14.8,
     "shown": {"p_win": 0.701, "margin": 0.854, "loss": 0.3547, "grad_scale": 0.0418, "r_w": 0.378, "r_l": -0.476},
     "screenshot": "02_dpo_beta0.14.jpg"},
]
PORTAL_SIMPO = {"gamma": 0.80, "beta": 2.0, "len_w": 64, "len_l": 240, "avg_w": -0.18, "avg_l": -0.32,
                "shown": {"sum_w": -11.5, "sum_l": -76.8, "raw_difference": 65.3, "r_w": -0.360, "r_l": -0.640,
                          "margin_gap": -0.520, "p_win": 0.373, "loss": 0.9866},
                "screenshot": "03_simpo_gamma0.80_beta2.0.jpg"}


def tol(x: float) -> float:
    """Half a unit of the last decimal the portal printed."""
    s = f"{abs(x)}"
    dec = len(s.split(".")[1]) if "." in s else 0
    return 0.5 * 10 ** (-dec) + 1e-12


def main() -> dict:
    out = {"tool": "Course portal · Week 4 · 3. Simulation · 'Direct Preference Optimization & Activation Steering Lab' (dlearn-cai.web.app)",
           "date": "2026-11-01", "tested_by": "Squadron Shannon", "dpo": [], "simpo": {}, "steering": {}}
    all_ok = True
    print("Portal DPO lab vs our DPOLoss (Part 1):")
    for case in PORTAL_DPO:
        t = lambda v: torch.tensor([v], dtype=torch.float64)
        loss, m = core.DPOLoss(case["beta"])(t(case["pi_w"]), t(case["pi_l"]), t(case["ref_w"]), t(case["ref_l"]))
        logit = case["beta"] * ((case["pi_w"] - case["ref_w"]) - (case["pi_l"] - case["ref_l"]))
        ours = {"p_win": float(m["bt_probability"]), "margin": float(m["reward_margin"]), "loss": float(loss),
                "grad_scale": case["beta"] * (1 / (1 + math.exp(logit))), "r_w": float(m["chosen_rewards"]),
                "r_l": float(m["rejected_rewards"])}
        match = {k: abs(ours[k] - v) <= tol(v) for k, v in case["shown"].items()}
        all_ok &= all(match.values())
        print(f"  beta {case['beta']:.2f}: " + "  ".join(f"{k} portal {case['shown'][k]} / ours {ours[k]:.4f} {'OK' if match[k] else 'X'}" for k in ours))
        out["dpo"].append({**case, "ours": ours, "match": match})

    s = PORTAL_SIMPO
    loss, m = simpo.SimPOLoss(s["beta"], s["gamma"])(torch.tensor([s["avg_w"]], dtype=torch.float64), torch.tensor([s["avg_l"]], dtype=torch.float64))
    gap = float(m["reward_gap"]) - s["gamma"]
    ours = {"sum_w": s["avg_w"] * s["len_w"], "sum_l": s["avg_l"] * s["len_l"],
            "raw_difference": s["avg_w"] * s["len_w"] - s["avg_l"] * s["len_l"], "r_w": float(m["reward_w"]),
            "r_l": float(m["reward_l"]), "margin_gap": gap, "p_win": 1 / (1 + math.exp(-gap)), "loss": float(loss)}
    match = {k: abs(ours[k] - v) <= tol(v) for k, v in s["shown"].items()}
    all_ok &= all(match.values())
    print("Portal SimPO lab vs our SimPOLoss (Part 3):")
    print("  " + "  ".join(f"{k} portal {s['shown'][k]} / ours {ours[k]:.4f} {'OK' if match[k] else 'X'}" for k in ours))
    note = ("With gamma = 0.80 the portal's own example gives P = 37 %: the concise answer IS better per token "
            f"(reward gap {float(m['reward_gap']):.2f}), but the gap is smaller than the target margin, so SimPO keeps pushing.")
    print("  " + note)
    out["simpo"] = {**s, "ours": ours, "match": match, "note": note}

    r4 = PS / "results" / "04_steering_audit.json"
    steer = {"portal_claims": {"layer": "Layer 16 of 32 (middle layers 14-22 work best)",
                               "cost": "zero additional floating-point operations during decoding"},
             "screenshot": "04_steering_layer16of32_alpha1.2.jpg"}
    if r4.exists():
        a = json.loads(r4.read_text())
        layer, n = a["chosen_layer"], a["layers"]
        d, params = a["d_model"], 494e6
        steer["ours"] = {"chosen_layer": f"{layer} of {n} ({layer / n:.0%} depth; portal 16 of 32 = 50 %, range 44-69 %)",
                         "extra_flops_per_token": f"{d} additions (one vector add) vs ~{2 * params:.1e} FLOPs per token forward "
                                                  f"= {d / (2 * params):.1e} of the cost: not zero, but negligible"}
        print(f"Portal steering lab: best layer in our sweep {layer}/{n} ({layer / n:.0%} depth) vs the portal's 'middle layers'; "
              f"hook cost {d} adds per token = {d / (2 * params):.1e} of a forward pass")
    out["steering"] = steer
    out["all_numeric_values_match"] = bool(all_ok)
    (HERE / "portal_results.json").write_text(json.dumps(out, indent=2))
    print(f"{'[PASS]' if all_ok else '[FAIL]'} every number shown by the portal DPO and SimPO labs is reproduced by our modules")
    return out


if __name__ == "__main__":
    main()
