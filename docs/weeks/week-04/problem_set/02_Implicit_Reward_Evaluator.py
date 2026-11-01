"""
Mission 04 — Part 2: Communications & Protocol Officer
Implicit Reward Extraction & Bradley-Terry Evaluation

Squadron Shannon · Station: Communications & Protocol Officer · Ankit Kumar & Garima Swami
Course: 14013 Conversational AI (BTU Cottbus-Senftenberg, WS 2026/27)
Sector: SEC-04-ORI (Betelgeuse) · AURA Subsystem 04: Plasma Ignition & Ethical Steering Core

After DPO the policy is its own reward model:  r_hat(x, y) = beta * log( pi_theta(y|x) / pi_ref(y|x) ),
and the Bradley-Terry model turns two rewards into a preference probability  P(y_w > y_l | x) = sigma(r_hat_w - r_hat_l).

This file extracts those rewards from the real DPO policy of Part 1, measures pairwise accuracy on held-out pairs
(against the noisy annotator labels and against the hidden true reward r*), and checks whether the probabilities
are calibrated: a reliability diagram against the empirical preference rate in each probability bin, the expected
calibration error (ECE), Brier score and NLL, plus temperature scaling fitted on a separate calibration split.

Run:  python 02_Implicit_Reward_Evaluator.py           (~20 s on CPU)
      python 02_Implicit_Reward_Evaluator.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                     # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


core = _load("dpo_core", "01_DPO_Loss_Core.py")


@torch.no_grad()
def evaluate_implicit_rewards(policy, ref, batch: Dict[str, List], beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """Implicit rewards for a batch {"prompts", "chosen", "rejected"} and the Bradley-Terry win probability."""
    pw, pl = core.pair_logps(policy, batch["prompts"], batch["chosen"], batch["rejected"])
    rw, rl = core.pair_logps(ref, batch["prompts"], batch["chosen"], batch["rejected"])
    reward_w, reward_l = beta * (pw - rw), beta * (pl - rl)
    delta = reward_w - reward_l
    return {"reward_w": reward_w, "reward_l": reward_l, "delta_r": delta, "p_win": torch.sigmoid(delta),
            "pairwise_accuracy": (reward_w > reward_l).float().mean(), "mean_margin": delta.mean()}


def calibration_curve(probs: Sequence[float], outcomes: Sequence[int], num_bins: int = 10) -> List[Dict[str, float]]:
    """Reliability diagram data: mean predicted probability vs empirical frequency in equal-width bins."""
    rows = []
    for i in range(num_bins):
        lo, hi = i / num_bins, (i + 1) / num_bins
        sel = [(p, o) for p, o in zip(probs, outcomes) if lo <= p < hi or (i == num_bins - 1 and p == 1.0)]
        n = len(sel)
        rows.append({"bin": f"[{lo:.1f}, {hi:.1f})", "count": n,
                     "avg_predicted_prob": sum(p for p, _ in sel) / n if n else (lo + hi) / 2,
                     "empirical_rate": sum(o for _, o in sel) / n if n else float("nan")})
    return rows


def calibration_metrics(probs: torch.Tensor, outcomes: torch.Tensor, num_bins: int = 10) -> Dict[str, float]:
    curve = calibration_curve(probs.tolist(), outcomes.tolist(), num_bins)
    n = len(probs)
    ece = sum(r["count"] / n * abs(r["avg_predicted_prob"] - r["empirical_rate"]) for r in curve if r["count"])
    p = probs.clamp(1e-7, 1 - 1e-7)
    return {"ece": ece, "brier": float(((probs - outcomes) ** 2).mean()),
            "nll": float(-(outcomes * p.log() + (1 - outcomes) * (1 - p).log()).mean()),
            "accuracy": float(((probs > 0.5).float() == outcomes).float().mean())}


def fit_temperature(logits: torch.Tensor, outcomes: torch.Tensor) -> float:
    """Temperature scaling: one scalar T minimising the NLL of sigma(logit / T) (Guo et al., 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits / log_t.exp(), outcomes)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def plot(curves: Dict[str, List[Dict]], deltas: torch.Tensor, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([0, 1], [0, 1], "--", c="grey", lw=1, label="perfect calibration")
    for (name, curve), col in zip(curves.items(), ["#b91c1c", "#0369a1", "#047857"]):
        pts = [(r["avg_predicted_prob"], r["empirical_rate"]) for r in curve if r["count"] >= 10]
        ax[0].plot(*zip(*pts), "o-", color=col, label=name)
    ax[0].set_xlabel("predicted P(y1 > y2) = sigma(r_hat1 - r_hat2)"); ax[0].set_ylabel("empirical rate y1 preferred")
    ax[0].set_title("Bradley-Terry reliability diagram (held-out)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].hist(deltas.numpy(), bins=40, color="#0369a1", alpha=.8)
    ax[1].set_title("implicit reward margins r_hat1 - r_hat2"); ax[1].set_xlabel("margin")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout(); FIGURES.mkdir(exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def main(quick: bool = False) -> Dict:
    print("=" * 100)
    print("AURA Subsystem 04 · Part 2 · Implicit Reward Extraction & Bradley-Terry Calibration (Squadron Shannon)")
    print("=" * 100)
    beta = 0.1
    ref = core.sft_reference(quick=quick)
    train = core.make_preferences(1000 if quick else 4000, seed=11)
    test = core.make_preferences(500 if quick else 2000, seed=12)
    policy, _ = core.train_dpo(ref, train, beta=beta, epochs=2 if quick else 4)

    batch = {"prompts": test.prompts, "chosen": test.chosen, "rejected": test.rejected}
    ev = evaluate_implicit_rewards(policy, ref, batch, beta)
    true_gap = torch.tensor([core.true_reward(x, w) - core.true_reward(x, l) for x, w, l in zip(test.prompts, test.chosen, test.rejected)])
    agree = ((ev["delta_r"] > 0) == (true_gap > 0))[true_gap != 0].float().mean()
    print(f"[1] Implicit rewards on {len(test.prompts)} held-out pairs (beta {beta}):")
    print(f"    pairwise accuracy vs annotator labels   = {float(ev['pairwise_accuracy']):.1%}")
    print(f"    pairwise accuracy vs hidden true reward = {float(agree):.1%}  (pairs with r*(y_w) != r*(y_l))")
    oracle_acc = sum(max(p, 1 - p) for p in test.p_first) / len(test.p_first)
    print(f"    ceiling: even the true annotator probabilities predict their noisy labels only {oracle_acc:.1%} of the time")
    print(f"    mean margin {float(ev['mean_margin']):+.3f} · mean P(y_w > y_l) {float(ev['p_win'].mean()):.1%}")

    # accuracy by difficulty: how far apart the two answers really are
    print("\n    accuracy vs true reward by difficulty |r*(y_w) - r*(y_l)|:")
    edges = [0.0, 0.25, 0.5, 1.0, 3.01]
    diff_rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (true_gap.abs() > lo) & (true_gap.abs() <= hi) if lo > 0 else (true_gap.abs() > 0) & (true_gap.abs() <= hi)
        acc = float(((ev["delta_r"] > 0) == (true_gap > 0))[sel].float().mean())
        diff_rows.append({"range": f"({lo}, {hi:.2f}]", "count": int(sel.sum()), "accuracy": acc})
        print(f"      ({lo:.2f}, {min(hi, 3):.2f}]  n = {int(sel.sum()):>4}  accuracy {acc:.1%}")

    print("\n    first 8 held-out pairs:")
    print(f"    {'pair':<5} {'r_hat(y_w)':>10} {'r_hat(y_l)':>10} {'margin':>8} {'P(y_w>y_l)':>10} {'r*(y_w)':>8} {'r*(y_l)':>8}")
    for i in range(8):
        print(f"    #{i + 1:<4} {float(ev['reward_w'][i]):>10.3f} {float(ev['reward_l'][i]):>10.3f} {float(ev['delta_r'][i]):>+8.3f} "
              f"{float(ev['p_win'][i]):>10.1%} {core.true_reward(test.prompts[i], test.chosen[i]):>8.2f} "
              f"{core.true_reward(test.prompts[i], test.rejected[i]):>8.2f}")

    # calibration in the original sampling order (y1, y2), so outcomes are not all 1
    first_won = torch.tensor(test.first_won, dtype=torch.float32)
    sign = first_won * 2 - 1                                       # +1 if y1 is the chosen one
    logit_y1 = ev["delta_r"] * sign
    half = len(logit_y1) // 2
    temp = fit_temperature(logit_y1[:half], first_won[:half])
    held = slice(half, None)
    variants = {"DPO implicit reward (raw)": torch.sigmoid(logit_y1[held]),
                f"temperature-scaled (T = {temp:.2f})": torch.sigmoid(logit_y1[held] / temp),
                "oracle annotator P (true BT)": torch.tensor(test.p_first)[held]}
    print(f"\n[2] Bradley-Terry calibration on the evaluation half ({len(first_won[held])} pairs, 10 bins; T fitted on the other half):")
    print(f"    {'predictor':<34} | {'ECE':>6} | {'Brier':>6} | {'NLL':>6} | {'acc':>6}")
    metrics, curves = {}, {}
    for name, probs in variants.items():
        m = calibration_metrics(probs, first_won[held])
        metrics[name] = m
        curves[name] = calibration_curve(probs.tolist(), first_won[held].tolist())
        print(f"    {name:<34} | {m['ece']:>6.3f} | {m['brier']:>6.3f} | {m['nll']:>6.3f} | {m['accuracy']:>6.1%}")
    print("\n    reliability bins of the raw DPO probabilities:")
    for r in curves["DPO implicit reward (raw)"]:
        if r["count"]:
            print(f"      {r['bin']}  n = {r['count']:>4}  predicted {r['avg_predicted_prob']:.3f}  empirical {r['empirical_rate']:.3f}")

    raw = metrics["DPO implicit reward (raw)"]
    ts = metrics[f"temperature-scaled (T = {temp:.2f})"]
    ok = float(agree) > 0.8 and ts["ece"] <= raw["ece"] + 1e-9 and ts["nll"] <= raw["nll"] + 1e-9
    direction = "over-confident (T > 1 softens it)" if temp > 1 else "under-confident (T < 1 sharpens it)"
    print(f"\n    -> with beta = {beta} the raw Bradley-Terry probabilities are {direction}; one scalar temperature fixes most of the ECE")
    print(f"\n{'[PASS]' if ok else '[FAIL]'} implicit rewards rank held-out pairs correctly and calibrate with one temperature")
    suffix = "_quick" if quick else ""
    plot(curves, logit_y1[held], FIGURES / f"02_bt_calibration{suffix}.png")
    result = {"beta": beta, "n_test": len(test.prompts), "pairwise_accuracy_labels": float(ev["pairwise_accuracy"]),
              "pairwise_accuracy_true_reward": float(agree), "annotator_ceiling": oracle_acc,
              "accuracy_by_difficulty": diff_rows, "temperature": temp, "calibration": metrics,
              "reliability_raw": curves["DPO implicit reward (raw)"], "all_passed": bool(ok)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"02_implicit_reward{suffix}.json").write_text(json.dumps(result, indent=2))
    print(f"Results -> results/02_implicit_reward{suffix}.json · figure -> figures/02_bt_calibration{suffix}.png")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
