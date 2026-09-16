# Adversarial-Robustness-of-Fraud-Detection-Models
Status: work in progress — preliminary results below. The threat model, attack implementation, and first evasion experiments are complete. The full degradation sweep and cross-model comparison are in progress.

A follow-on study to credit-card-fraud-detection, which built a leakage-safe fraud detection pipeline on the [IEEE-CIS](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset. That project asked how well can we detect fraud? This one asks the question that matters once a model is deployed: what happens when the fraudster knows the model exists and adapts?

## Motivation

Standard model evaluation assumes the data-generating process is static. Fraud violates that assumption by construction — fraudsters observe which transactions get blocked and adjust. A model with excellent held-out PR-AUC can still be trivially evadable if its decision boundary depends on features the attacker controls.

The question this project answers is not "is the model accurate?" but "how much accuracy does an adapting adversary cost us, and which model architecture degrades most gracefully?"

## Threat model

Adversarial ML research on images typically assumes an unconstrained perturbation budget under an L-p norm. That assumption is wrong for tabular fraud data, because most features are not attacker-controllable. A fraudster cannot retroactively change the card's transaction history, the issuing bank, or the device fingerprint.

This project therefore defines a realistic, constrained threat model:

Element	Specification
Attacker goal	Evasion — get a fraudulent transaction scored below the decision threshold
Attacker knowledge	White-box (worst case): full access to the trained model
Manipulable features	TransactionAmt only
Frozen features	Card-level aggregates, temporal features, categorical identifiers
Budget	Bounded perturbation of the transaction amount

Restricting the attack surface to a single manipulable feature is the point, not a limitation. It reflects what an attacker can actually do at transaction time, and it makes the resulting robustness estimate meaningful rather than pessimistic-by-construction.

## Methodology
Attack: vectorised grid search over the perturbation budget for TransactionAmt, evaluating all candidate perturbations in parallel rather than looping per transaction.
Compute: PyTorch with CUDA (NVIDIA GTX 1650).
Victim models: the three model families trained in the parent project — logistic regression, XGBoost, and a PyTorch MLP.
Preliminary findings

Evasion rates are low at moderate attack budgets. Perturbing the transaction amount within a realistic range does not reliably push fraudulent transactions below the decision threshold.

The mechanism appears to be feature dependence. XGBoost's decision boundary leans heavily on card-level aggregate features — the frozen part of the feature vector — rather than on the raw transaction amount. An attacker manipulating only the amount is pushing on a lever the model barely uses.

This connects directly to a finding from the parent project, where SHAP analysis showed the model's most important signals were aggregate and behavioural rather than transaction-local. Feature choices made for predictive reasons turned out to have robustness consequences that were not designed for.

These results are preliminary and cover a partial budget range. The full sweep is the next milestone.

## Roadmap
 Threat model definition
 GPU environment (PyTorch + CUDA)
 Vectorised grid-search attack
 Initial evasion experiments
 Full degradation curve across the attack budget range
 Security Evaluation Index (SEI) computation
 Cross-model robustness comparison (LR vs XGBoost vs MLP)
 Analysis of the accuracy–robustness trade-off
 
## Repository structure
```
.
├── notebooks/        # Threat model exploration and attack experiments
├── src/              # Attack implementation
├── results/          # Figures and metrics
├── requirements.txt
└── README.md
```
## Reproducing 
```bash
git clone https://github.com/lakshyakaviya/adversarial-robustness-fraud
cd adversarial-robustness-fraud
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Trained models and feature artifacts are produced by the parent project. The IEEE-CIS dataset is not committed to this repository; download it from the Kaggle competition page.

Reference

Jin Xiao, Yuhang Tian, Yanlin Jia, Xiaoyi Jiang, Lean Yu, Shouyang Wang (2023) Black-Box Attack-Based Security Evaluation Framework for Credit Card Fraud Detection Models. INFORMS Journal on Computing

Related
[credit-card-fraud-detection](https://github.com/lakshyakaviya/credit-card-fraud-detection) — the fraud detection pipeline this study attacks.
