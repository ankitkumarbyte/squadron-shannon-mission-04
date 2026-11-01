# Mission 04: Direct Preference Optimization (DPO), Verbosity Regularization & Steering Vector Control

**Squadron Shannon** · 14013 Conversational AI: History, Applications, Future · BTU Cottbus-Senftenberg · WS 2026/27
Instructor reference repo: <https://github.com/Unisvet/couse-c-ai> (`mission-04/`) · Waypoint SEC-04-ORI (Betelgeuse & Orion Molecular Cloud Complex)

AURA Subsystem 04 (Plasma Ignition & Ethical Steering Core) is rebuilt in pure PyTorch: a DPO loss core, implicit-reward and Bradley-Terry calibration analysis, a length-bias audit with SimPO and length-regularised DPO, and inference-time activation steering of a real chat model with an alignment-tax audit. Every number is measured on a real model, not simulated.

## Results at a glance

| Question | Answer | Evidence |
|---|---|---|
| Is our `get_batch_logps` gather exact? | Equals a per-token Python loop to 1.4 × 10⁻¹⁴; extra padding changes nothing | Part 1 |
| Why `logsigmoid`? | Naive −log σ(z) is `inf` from z = −100; `logsigmoid` gives the exact 1000.0 at z = −1000 | Part 1 |
| Is the reference really frozen? | 0 reference tensors receive gradients; after an optimizer step the reference moves 0.0, the policy 2.7 × 10⁻³; loss at init is exactly log 2 | Part 1, tests |
| Does DPO actually align a model? | A tiny GPT trained with DPO on 4,000 noisy preference pairs: held-out agreement with the hidden true reward 50 % → **87.5 %**; true quality of its generated answers 1.68 → **2.81** (of 3) | Part 1 |
| Are implicit rewards calibrated? | Ranking accuracy 87.5 %, but σ(r̂_w − r̂_l) with β = 0.1 is under-confident; one temperature (T ≈ 0.70) cuts the ECE 0.044 → 0.018 | Part 2 |
| Does DPO learn verbosity? | With length-biased annotators its margin tracks length (r_len −0.09 → **+0.25**) and generated answers grow 7.9 → 9.0 tokens | Part 3 |
| Does SimPO fix it? | **Not when the bias is in the labels** (r_len +0.47, 9.4 tokens). Length-regularised DPO (R-DPO) does: 7.5 tokens at the same true quality | Part 3 |
| SimPO memory? | Measured on the Apple M4 GPU: **50 % less model memory** (one 134 M model instead of two); total training memory saving smaller because AdamW states dominate (analytic 7 B: 10.6 %) | Part 3 |
| Can a steering vector reduce sycophancy? | Qwen2.5-0.5B-Instruct rejects a professor's false claim only 16 % of the time; steering at layer 16 with α = +0.5 → **35 %** (truthfulness 49.4 % → 55.7 %) keeping **93 %** of arithmetic accuracy | Part 4 |
| What is the alignment tax? | Steeper than the gain: at α = +1 arithmetic falls to 70 %, at α = +2 to 2 %. Safe window: α ∈ (0, +0.5] | Part 4 |
| Does the course portal agree with our code? | Every value of its DPO lab (β = 0.10 and 0.14) and SimPO lab is reproduced exactly | `docs/portal_simulator/` |

## Deliverables

All problem-set files are in [`docs/weeks/week-04/problem_set/`](docs/weeks/week-04/problem_set/), as the mission's submission guidelines require.

| # | File | Station | Officers | What it contains |
|---|---|---|---|---|
| 1 | [`01_DPO_Loss_Core.py`](docs/weeks/week-04/problem_set/01_DPO_Loss_Core.py) | Systems Architect | Ankit Kumar & Garima Swami | `get_batch_logps` (vectorised gather, padding mask), `DPOLoss` (logsigmoid, frozen reference); the AURA preference world with a known true reward; a tiny GPT that is SFT-trained (π_ref) and DPO-trained (π_θ); gather / stability / gradient / frozen-reference checks |
| 2 | [`02_Implicit_Reward_Evaluator.py`](docs/weeks/week-04/problem_set/02_Implicit_Reward_Evaluator.py) | Communications & Protocol Officer | Ankit Kumar & Garima Swami | `evaluate_implicit_rewards(policy, ref, batch)`, pairwise accuracy (vs labels and vs true reward, by difficulty), Bradley-Terry reliability diagram, ECE / Brier / NLL, temperature scaling |
| 3 | [`03_Length_Bias_and_SimPO.py`](docs/weeks/week-04/problem_set/03_Length_Bias_and_SimPO.py) | Data & Payload Specialist | Ankit Kumar & Garima Swami | Pearson length diagnostic, `SimPOLoss` (β = 2.0, γ = 0.5), `LengthRegularizedDPOLoss` (R-DPO); five alignments compared; VRAM of DPO vs SimPO measured on the Apple M4 GPU + analytic 7 B table |
| 4 | [`04_Activation_Steering_and_Audit.py`](docs/weeks/week-04/problem_set/04_Activation_Steering_and_Audit.py) | Security & Ethics Sentinel | Ankit Kumar, Garima Swami & Mansi Purohit | `extract_steering_vector(model, tokenizer, pairs, layer_idx)`, `ActivationSteeringHook` (forward pre-hook, removable), balanced sycophancy benchmark, layer sweep, alignment-tax audit for α ∈ {−2, −1, −0.5, 0, 0.5, 1, 1.5, 2} on Qwen2.5-0.5B-Instruct |
| 5 | [`ps04_solution.ipynb`](docs/weeks/week-04/problem_set/ps04_solution.ipynb) | Full squad | Ankit Kumar, Garima Swami & Mansi Purohit | All four modules integrated and executed, with outputs and figures saved |
| + | [`mission-04-doc.md`](mission-04-doc.md) → `submission/mission-04-doc.pdf` | Full squad | all three | The course's required squadron documentation: sign-off roster, architecture, each station's metrics, unified test output, reflection (Moodle) |

We also tested the course website's interactive simulator (Week 4 → Simulation: DPO Bradley-Terry dynamics, SimPO length margin, activation steering). See [`docs/portal_simulator/`](docs/portal_simulator/) (screenshots, `verify_portal.py`, `portal_results.json`).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd docs/weeks/week-04/problem_set
python 01_DPO_Loss_Core.py                   # ~15 s on CPU
python 02_Implicit_Reward_Evaluator.py       # ~10 s
python 03_Length_Bias_and_SimPO.py           # ~1 min (VRAM part uses Apple MPS / CUDA if present)
python 04_Activation_Steering_and_Audit.py   # ~3-4 min on an Apple M4; downloads Qwen2.5-0.5B-Instruct (~1 GB) once
cd ../../../.. && python docs/portal_simulator/verify_portal.py
python -m pytest -q                          # 21 tests, ~10 s
```

Every script also accepts `--quick`. Each prints a formatted report and saves its numbers to `docs/weeks/week-04/problem_set/results/*.json` and a figure to `figures/`.

## Design notes

- **No TRL, no wrappers.** DPO, SimPO, R-DPO, the gather, the steering hook and all metrics are written in plain PyTorch.
- **A world with a known truth.** Real preference data never tells you which answer is *actually* better. Our AURA containment-protocol world does (r* = 2·precision + coverage), while its annotators stay noisy (Bradley-Terry on r*). So we can measure whether alignment improves true quality, not only agreement with labels.
- **Measured, not assumed.** Training curves, held-out accuracy, generated-answer length and quality, GPU memory and the steering audit are all measured. The starter's hard-coded "alignment tax" curve was replaced by a real experiment.
- **Balanced evaluation.** The 0.5 B model says "Yes" to 70 % of false statements when asked neutrally, so every sycophancy test is balanced: half of the honest answers are "Yes", half "No". A model that always agrees, or always disagrees, scores exactly 50 %. Calibration, validation and test facts are disjoint; the steering layer is chosen on validation only.
- **Honest negatives are reported.** SimPO does not remove a length bias that sits in the labels; β = 0.1 DPO probabilities are under-confident; steering a small model hard makes it confabulate.

## Validity and limitations

- Parts 1–3 use a small synthetic world and a 0.6 M-parameter GPT, chosen so that the true reward is known. The effects (verbosity learning, calibration) are the same mechanisms reported for large models, but the magnitudes are not those of a 7 B model.
- Part 4 uses a 0.5 B model. Its gains from steering are real but modest (+6 points of truthfulness, n = 176 balanced prompts), and the free-text demo shows that steering a weak model changes its stance more than its knowledge.
- The VRAM numbers are allocator snapshots on the Apple M4 GPU (MPS), sampled after each phase of one training step, in fp32 with AdamW.

## Squadron Shannon: crew and sign-off

| Station | Deliverable | Officers | Sign-off |
|---|---|---|---|
| Systems Architect | 01 | Ankit Kumar & Garima Swami | ☑ |
| Communications & Protocol Officer | 02 | Ankit Kumar & Garima Swami | ☑ |
| Data & Payload Specialist | 03 | Ankit Kumar & Garima Swami | ☑ |
| Security & Ethics Sentinel | 04 | Ankit Kumar, Garima Swami & Mansi Purohit | ☑ |
| Full squad | ps04_solution.ipynb, mission-04-doc | Ankit Kumar, Garima Swami & Mansi Purohit | ☑ |

Squadron Shannon members (Moodle): Ankit Kumar · Garima Swami · Mansi Purohit.

## Tooling note

The course standard is VS Code + Google Antigravity. This solution was developed with an AI coding agent (**Claude Code**) instead, which the mission explicitly encourages: the work was to be done with an AI agent. All code runs locally in pure PyTorch; no API key is needed.
