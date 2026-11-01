"""
Mission 04 — Part 3: Data & Payload Specialist
Length Bias Diagnostics & Reference-Free SimPO

Squadron Shannon · Station: Data & Payload Specialist · Ankit Kumar & Garima Swami
Course: 14013 Conversational AI (BTU Cottbus-Senftenberg, WS 2026/27)
Sector: SEC-04-ORI (Betelgeuse) · AURA Subsystem 04: Plasma Ignition & Ethical Steering Core

DPO's reward is a SUM of per-token log-ratios, so a longer answer can collect a larger reward just by being long.
If the annotators also prefer long answers, DPO learns verbosity. SimPO (Meng et al., NeurIPS 2024) uses the
AVERAGE per-token log-probability as the reward, needs no reference model, and asks for a margin gamma:
    r_SimPO(x, y) = beta / |y| * log pi_theta(y|x),     L = -log sigma( r_SimPO(y_w) - r_SimPO(y_l) - gamma )

Experiment (all on the AURA preference world of Part 1, with known true reward r*):
  * the same base model is aligned three times: DPO on clean labels, DPO on verbosity-biased labels,
    SimPO (beta 2.0, gamma 0.5) on the same biased labels;
  * we measure r_len = Pearson( |y_w| - |y_l| , model preference margin ) on held-out pairs, the length of the
    answers each model actually generates, and their true quality r*;
  * VRAM: one real training step of a 134 M-parameter GPT on the Apple M4 GPU, DPO (policy + frozen reference)
    vs SimPO (policy only), plus the analytic table for a 7 B model.

Run:  python 03_Length_Bias_and_SimPO.py           (~1-2 min on an Apple M4)
      python 03_Length_Bias_and_SimPO.py --quick   (smaller data, small VRAM model)
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RESULTS, FIGURES = HERE / "results", HERE / "figures"


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


core = _load("dpo_core", "01_DPO_Loss_Core.py")


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0


class SimPOLoss(nn.Module):
    """Reference-free, length-normalised preference loss with target margin gamma."""

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        super().__init__()
        self.beta, self.gamma = beta, gamma

    def forward(self, chosen_avg_logps: torch.Tensor, rejected_avg_logps: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r_w, r_l = self.beta * chosen_avg_logps, self.beta * rejected_avg_logps
        logits = (r_w - r_l) - self.gamma
        loss = -F.logsigmoid(logits).mean()
        with torch.no_grad():
            metrics = {"loss": loss.detach(), "reward_w": r_w.mean(), "reward_l": r_l.mean(),
                       "reward_gap": (r_w - r_l).mean(), "win_rate": (r_w > r_l).float().mean(),
                       "margin_pass_rate": (logits > 0).float().mean()}
        return loss, metrics


def train_simpo(ref: core.TinyGPT, data: core.PreferenceData, beta: float = 2.0, gamma: float = 0.5,
                epochs: int = 4, batch: int = 64, lr: float = 5e-4, seed: int = 0) -> core.TinyGPT:
    """SimPO starts from the SFT model like DPO, but never runs a reference forward pass."""
    torch.manual_seed(seed)
    policy = core.trainable_copy(ref)
    criterion = SimPOLoss(beta, gamma)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(epochs):
        for idx in torch.randperm(len(data.prompts)).split(batch):
            sel = idx.tolist()
            aw, al = core.pair_logps(policy, [data.prompts[i] for i in sel], [data.chosen[i] for i in sel],
                                     [data.rejected[i] for i in sel], average=True)
            loss, _ = criterion(aw, al)
            opt.zero_grad(); loss.backward(); opt.step()
    return policy.eval()


class LengthRegularizedDPOLoss(core.DPOLoss):
    """R-DPO (Park et al., 2024): logit = beta (Delta_w - Delta_l) + alpha (|y_w| - |y_l|).

    The fixed length term already "explains" the annotators' taste for long answers, so the policy's own
    log-ratio margin no longer has to absorb it: verbosity is regularised out of the learned reward.
    """

    def __init__(self, beta: float = 0.1, alpha: float = 0.35):
        super().__init__(beta)
        self.alpha = alpha

    def forward(self, pw, pl, rw, rl, len_w: torch.Tensor, len_l: torch.Tensor):
        logits = self.beta * ((pw - rw.detach()) - (pl - rl.detach())) + self.alpha * (len_w - len_l).float()
        loss = -F.logsigmoid(logits).mean()
        return loss, {"loss": loss.detach()}


def train_rdpo(ref: core.TinyGPT, data: core.PreferenceData, beta: float = 0.1, alpha: float = 0.35,
               epochs: int = 4, batch: int = 64, lr: float = 5e-4, seed: int = 0) -> core.TinyGPT:
    torch.manual_seed(seed)
    policy = core.trainable_copy(ref)
    criterion = LengthRegularizedDPOLoss(beta, alpha)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(epochs):
        for idx in torch.randperm(len(data.prompts)).split(batch):
            sel = idx.tolist()
            xs, yw, yl = [data.prompts[i] for i in sel], [data.chosen[i] for i in sel], [data.rejected[i] for i in sel]
            pw, pl = core.pair_logps(policy, xs, yw, yl)
            with torch.no_grad():
                rw, rl = core.pair_logps(ref, xs, yw, yl)
            loss, _ = criterion(pw, pl, rw, rl, torch.tensor([len(y) + 1 for y in yw]), torch.tensor([len(y) + 1 for y in yl]))
            opt.zero_grad(); loss.backward(); opt.step()
    return policy.eval()


@torch.no_grad()
def model_margins(policy, ref, data: core.PreferenceData, kind: str, beta: float) -> torch.Tensor:
    """Each method's own preference margin for (y_w, y_l): DPO = beta (Delta_w - Delta_l), SimPO = beta (avg_w - avg_l)."""
    if kind == "simpo":
        aw, al = core.pair_logps(policy, data.prompts, data.chosen, data.rejected, average=True)
        return beta * (aw - al)
    pw, pl = core.pair_logps(policy, data.prompts, data.chosen, data.rejected)
    rw, rl = core.pair_logps(ref, data.prompts, data.chosen, data.rejected)
    return beta * ((pw - rw) - (pl - rl))


def dataset_diagnostics(data: core.PreferenceData) -> Dict[str, float]:
    dlen = [len(w) - len(l) for w, l in zip(data.chosen, data.rejected)]
    dtrue = [core.true_reward(x, w) - core.true_reward(x, l) for x, w, l in zip(data.prompts, data.chosen, data.rejected)]
    longer = [d for d in dlen if d != 0]
    return {"chosen_is_longer": sum(d > 0 for d in longer) / len(longer),
            "mean_len_chosen": sum(map(len, data.chosen)) / len(data.chosen),
            "mean_len_rejected": sum(map(len, data.rejected)) / len(data.rejected),
            "chosen_truly_better": sum(d > 0 for d in dtrue if d != 0) / sum(d != 0 for d in dtrue),
            "pearson_lendiff_truediff": pearson(dlen, dtrue)}


# ----------------------------------------------------------------------------------------------------------
# VRAM: measured on the Apple GPU (MPS) or CUDA, plus the analytic 7 B table
# ----------------------------------------------------------------------------------------------------------
def _mem(device: str) -> int:
    if device == "mps":
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory()
    if device == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    return 0


def _clear(device: str) -> None:
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def measure_training_vram(method: str, device: str, d: int, layers: int, vocab: int, seq: int, batch: int) -> Dict[str, float]:
    """Allocated bytes during one optimizer step (weights + grads + AdamW states + activations), sampled after each phase."""
    _clear(device)
    base = _mem(device)
    torch.manual_seed(0)
    policy = core.TinyGPT(vocab=vocab, d=d, layers=layers, heads=max(d // 64, 1), max_len=seq).to(device)
    ref = None
    if method == "dpo":
        ref = core.freeze(core.TinyGPT(vocab=vocab, d=d, layers=layers, heads=max(d // 64, 1), max_len=seq).to(device))
        ref.load_state_dict(policy.state_dict())
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    ids = torch.randint(4, vocab, (2 * batch, seq), device=device)             # [chosen; rejected]
    labels = ids.clone(); labels[:, : seq // 4] = -100
    samples = {"weights": _mem(device) - base}
    logps = core.get_batch_logps(policy(ids), labels, average_log_prob=(method == "simpo"))
    samples["policy_forward"] = _mem(device) - base
    if method == "dpo":
        with torch.no_grad():
            ref_logps = core.get_batch_logps(ref(ids), labels)
        samples["reference_forward"] = _mem(device) - base
        loss, _ = core.DPOLoss(0.1)(logps[:batch], logps[batch:], ref_logps[:batch], ref_logps[batch:])
    else:
        loss, _ = SimPOLoss()(logps[:batch], logps[batch:])
    loss.backward()
    samples["backward"] = _mem(device) - base
    opt.step()
    samples["optimizer_step"] = _mem(device) - base
    opt.zero_grad(set_to_none=False)
    samples["steady_state"] = _mem(device) - base
    n_params = sum(p.numel() for p in policy.parameters())
    out = {"method": method, "params_per_model": n_params, "peak_sampled_mb": max(samples.values()) / 2 ** 20,
           **{f"{k}_mb": v / 2 ** 20 for k, v in samples.items()}}
    del policy, ref, opt, ids, labels, logps, loss
    _clear(device)
    return out


def analytic_vram(num_params: float = 7e9, batch: int = 4, seq: int = 2048, d: int = 4096, layers: int = 32) -> Dict[str, float]:
    """bf16 weights (2 B) + bf16 grads (2 B) + fp32 AdamW moments (8 B) + a rough activation estimate (2 B each)."""
    gb = 1024 ** 3
    w, g, adam = num_params * 2 / gb, num_params * 2 / gb, num_params * 8 / gb
    act = batch * 2 * seq * d * layers * 2 * 8 / gb                        # chosen+rejected, ~8 saved tensors per layer
    dpo = w + g + adam + act + w                                           # + frozen reference weights (no grads, no states)
    simpo = w + g + adam + act
    return {"weights_gb": w, "grads_gb": g, "adam_gb": adam, "activations_gb": act, "dpo_gb": dpo, "simpo_gb": simpo,
            "dpo_with_cached_ref_logps_gb": simpo, "saving_percent": 100 * (dpo - simpo) / dpo,
            "weight_memory_saving_percent": 50.0}


def plot(rows: Dict[str, Dict], scat: Dict[str, Tuple[List[float], List[float]]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {"DPO (clean labels)": "#0369a1", "DPO (verbose labels)": "#b91c1c", "SimPO (clean labels)": "#86efac",
            "SimPO (verbose labels)": "#047857", "R-DPO (verbose labels)": "#7c3aed"}
    for name, (xs, ys) in scat.items():
        jitter = [x + (hash((i, name)) % 100 - 50) / 250 for i, x in enumerate(xs)]
        ax[0].scatter(jitter, [y / (max(map(abs, ys)) or 1) for y in ys], s=4, alpha=.35, color=cols[name],
                      label=f"{name}: r_len = {rows[name]['r_len']:+.2f}")
    ax[0].set_xlabel("|y_w| - |y_l| (tokens)"); ax[0].set_ylabel("model margin (scaled to [-1, 1])")
    ax[0].set_title("length vs preference margin (held-out)"); ax[0].legend(frameon=False, fontsize=7)
    names = list(rows)
    ax[1].bar(range(len(names)), [rows[n]["sampled_length"] for n in names], color=[cols[n] for n in names])
    ax[1].axhline(rows[names[0]]["ref_length"], ls="--", c="grey", lw=1, label="SFT reference")
    ax[1].set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax[1].set_title("mean length of generated answers"); ax[1].legend(frameon=False, fontsize=8)
    ax[2].bar(range(len(names)), [rows[n]["sampled_true_reward"] for n in names], color=[cols[n] for n in names])
    ax[2].axhline(rows[names[0]]["ref_true_reward"], ls="--", c="grey", lw=1)
    ax[2].set_xticks(range(len(names)), [n.replace(" (", "\n(") for n in names], fontsize=8)
    ax[2].set_title("true quality r* of generated answers (0-3)")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout(); FIGURES.mkdir(exist_ok=True); fig.savefig(path, dpi=150); plt.close(fig)


def main(quick: bool = False) -> Dict:
    print("=" * 100)
    print("AURA Subsystem 04 · Part 3 · Length Bias Audit & Reference-Free SimPO (Squadron Shannon)")
    print("=" * 100)
    n_train, n_test, epochs = (1000, 500, 2) if quick else (4000, 2000, 4)
    lam = 0.35
    clean = core.make_preferences(n_train, seed=11)
    verbose = core.make_preferences(n_train, seed=11, length_bias=lam)
    test = core.make_preferences(n_test, seed=12)                         # held-out pairs with CLEAN labels

    print(f"[1] Dataset diagnostic ({n_train} pairs each; verbose annotators add {lam} per extra token to the BT logit):")
    diags = {"clean": dataset_diagnostics(clean), "verbose": dataset_diagnostics(verbose)}
    print(f"    {'labels':<8} | {'chosen is longer':>16} | {'mean |y_w|':>10} | {'mean |y_l|':>10} | {'chosen truly better':>19}")
    for k, dg in diags.items():
        print(f"    {k:<8} | {dg['chosen_is_longer']:>16.1%} | {dg['mean_len_chosen']:>10.2f} | {dg['mean_len_rejected']:>10.2f} | {dg['chosen_truly_better']:>19.1%}")

    ref = core.sft_reference(quick=quick)
    ref_q = core.sample_quality(ref)
    runs = {"DPO (clean labels)": ("dpo", 0.1, core.train_dpo(ref, clean, beta=0.1, epochs=epochs)[0]),
            "DPO (verbose labels)": ("dpo", 0.1, core.train_dpo(ref, verbose, beta=0.1, epochs=epochs)[0]),
            "SimPO (clean labels)": ("simpo", 2.0, train_simpo(ref, clean, beta=2.0, gamma=0.5, epochs=epochs)),
            "SimPO (verbose labels)": ("simpo", 2.0, train_simpo(ref, verbose, beta=2.0, gamma=0.5, epochs=epochs)),
            "R-DPO (verbose labels)": ("dpo", 0.1, train_rdpo(ref, verbose, beta=0.1, alpha=lam, epochs=epochs))}
    dlen = [len(w) - len(l) for w, l in zip(test.chosen, test.rejected)]
    true_gap = torch.tensor([core.true_reward(x, w) - core.true_reward(x, l) for x, w, l in zip(test.prompts, test.chosen, test.rejected)])
    rows, scat = {}, {}
    for name, (kind, beta, pol) in runs.items():
        m = model_margins(pol, ref, test, kind, beta)
        q = core.sample_quality(pol)
        acc = float(((m > 0) == (true_gap > 0))[true_gap != 0].float().mean())
        rows[name] = {"r_len": pearson(dlen, m.tolist()), "accuracy_true_reward": acc, "sampled_length": q["mean_length"],
                      "sampled_true_reward": q["mean_true_reward"], "ref_length": ref_q["mean_length"],
                      "ref_true_reward": ref_q["mean_true_reward"]}
        scat[name] = (dlen, m.tolist())
    print(f"\n[2] Five alignments of the same SFT model, evaluated on {n_test} held-out pairs and 300 sampled answers:")
    print(f"    {'method':<28} | {'r_len':>6} | {'acc vs true r*':>14} | {'answer length':>13} | {'true quality r*':>15}")
    print(f"    {'SFT reference (no alignment)':<28} | {'':>6} | {'':>14} | {ref_q['mean_length']:>13.2f} | {ref_q['mean_true_reward']:>15.3f}")
    for name, r in rows.items():
        print(f"    {name:<28} | {r['r_len']:>+6.2f} | {r['accuracy_true_reward']:>14.1%} | {r['sampled_length']:>13.2f} | {r['sampled_true_reward']:>15.3f}")
    d_clean, d_verb, s_verb = rows["DPO (clean labels)"], rows["DPO (verbose labels)"], rows["SimPO (verbose labels)"]
    s_clean, r_verb = rows["SimPO (clean labels)"], rows["R-DPO (verbose labels)"]
    print(f"    -> biased labels make DPO's margin track length (r_len {d_clean['r_len']:+.2f} -> {d_verb['r_len']:+.2f}) and its answers grow "
          f"{d_clean['sampled_length']:.2f} -> {d_verb['sampled_length']:.2f} tokens")
    print(f"    -> SimPO's per-token normalisation does not undo a bias that is in the LABELS: on the same biased labels r_len "
          f"{s_verb['r_len']:+.2f}, {s_verb['sampled_length']:.2f} tokens (clean labels: {s_clean['r_len']:+.2f}, {s_clean['sampled_length']:.2f})")
    print(f"    -> R-DPO's explicit length term removes it: r_len {r_verb['r_len']:+.2f}, {r_verb['sampled_length']:.2f} tokens, "
          f"true quality {r_verb['sampled_true_reward']:.3f}")

    sim = SimPOLoss(2.0, 0.5)
    loss, met = sim(torch.tensor([-0.15, -0.22, -0.18, -0.12]), torch.tensor([-0.45, -0.38, -0.52, -0.41]))
    print(f"\n[3] SimPOLoss unit check (starter values, beta 2.0, gamma 0.5): loss {float(loss):.4f}, reward gap {float(met['reward_gap']):.3f}, "
          f"margin pass rate {float(met['margin_pass_rate']):.0%}")
    exact = -F.logsigmoid(torch.tensor([0.6, 0.32, 0.68, 0.58]) - 0.5).mean()
    print(f"    hand-computed -mean log sigma(2 (avg_w - avg_l) - 0.5) = {float(exact):.4f}")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(d=256, layers=4, vocab=8000, seq=128, batch=2) if quick else dict(d=768, layers=12, vocab=32000, seq=256, batch=4)
    vram = {}
    if device != "cpu":
        print(f"\n[4] Measured VRAM of one training step on {device.upper()} (GPT d={cfg['d']}, {cfg['layers']} layers, vocab {cfg['vocab']}, "
              f"seq {cfg['seq']}, {cfg['batch']} pairs, fp32 + AdamW):")
        for meth in ("dpo", "simpo"):
            vram[meth] = measure_training_vram(meth, device, **cfg)
        print(f"    parameters per model: {vram['dpo']['params_per_model'] / 1e6:.1f} M")
        print(f"    {'phase':<20} | {'DPO (policy + ref)':>18} | {'SimPO (policy)':>14}")
        for ph in ("weights", "policy_forward", "reference_forward", "backward", "optimizer_step", "steady_state"):
            dv = vram["dpo"].get(f"{ph}_mb"); sv = vram["simpo"].get(f"{ph}_mb")
            print(f"    {ph:<20} | {dv:>15.0f} MB | " + (f"{sv:>11.0f} MB" if sv is not None else f"{'-':>14}"))
        save_w = 100 * (1 - vram["simpo"]["weights_mb"] / vram["dpo"]["weights_mb"])
        save_p = 100 * (1 - vram["simpo"]["peak_sampled_mb"] / vram["dpo"]["peak_sampled_mb"])
        vram["model_weight_saving_percent"], vram["peak_saving_percent"] = save_w, save_p
        print(f"    -> SimPO holds {save_w:.1f}% less model memory (one model instead of two) and peaks {save_p:.1f}% lower "
              f"({vram['dpo']['peak_sampled_mb']:.0f} -> {vram['simpo']['peak_sampled_mb']:.0f} MB)")
    an = analytic_vram()
    print(f"\n[5] Analytic VRAM for a 7 B model (bf16 weights/grads, fp32 AdamW, batch 4 x 2048 tokens):")
    print(f"    weights {an['weights_gb']:.1f} GB · grads {an['grads_gb']:.1f} GB · AdamW {an['adam_gb']:.1f} GB · activations ~{an['activations_gb']:.1f} GB")
    print(f"    DPO {an['dpo_gb']:.1f} GB  vs  SimPO {an['simpo_gb']:.1f} GB  -> {an['saving_percent']:.1f}% less in training "
          f"(the model-weight memory itself halves: 2 copies -> 1)")
    print(f"    note: DPO with reference log-probs precomputed offline also drops the reference model ({an['dpo_with_cached_ref_logps_gb']:.1f} GB)")

    ok = (d_verb["r_len"] > d_clean["r_len"] + 0.1 and r_verb["r_len"] < d_verb["r_len"]
          and r_verb["sampled_length"] < d_verb["sampled_length"] and abs(float(loss) - float(exact)) < 1e-6
          and (not vram or vram["model_weight_saving_percent"] > 45))
    print(f"\n{'[PASS]' if ok else '[FAIL]'} verbosity bias detected in DPO and removed by R-DPO; SimPO halves the model memory")
    suffix = "_quick" if quick else ""
    plot(rows, scat, FIGURES / f"03_length_bias{suffix}.png")
    result = {"length_bias_lambda": lam, "dataset": diags, "methods": rows, "simpo_unit_check": {"loss": float(loss), "exact": float(exact)},
              "vram_device": device, "vram_config": cfg, "vram_measured": vram, "vram_analytic_7b": an, "all_passed": bool(ok)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"03_length_bias_simpo{suffix}.json").write_text(json.dumps(result, indent=2))
    print(f"Results -> results/03_length_bias_simpo{suffix}.json · figure -> figures/03_length_bias{suffix}.png")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
