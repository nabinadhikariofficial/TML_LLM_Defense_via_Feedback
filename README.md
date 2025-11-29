# Autonomous LLM Defense via Closed-Loop Adversarial Feedback

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20H200-76b900)](https://www.nvidia.com/)

> **A self-hardening safety pipeline that autonomously detects, generates, and patches LLM vulnerabilities in real-time.**

---

## 📖 Abstract

Large Language Models (LLMs) are brittle against adaptive adversarial attacks. Static defenses (like fixed safety filters) degrade quickly as attackers invent new jailbreaks. This project introduces a **closed-loop adversarial feedback system** where an LLM defense module autonomously:
1.  **Attacks itself** using a sophisticated Red-Team agent.
2.  **Detects failures** using a ground-truth Oracle.
3.  **Retrains itself** via online parameter-efficient fine-tuning (LoRA).

We demonstrate that training against a **Strong Adversary (70B)** significantly improves generalization to unseen attacks compared to a **Weak Adversary (8B)**, albeit with an observed "Alignment Tax" (increased False Positive Rate).

---

## 🏗️ System Architecture

The pipeline runs on a single **NVIDIA H200 (141GB)** using a 4-module architecture designed for high-throughput adversarial simulations.

| Module | Model | Role |
| :--- | :--- | :--- |
| **Target Model** | `meta-llama/Llama-3.1-8B-Instruct` | The protected asset (BF16). |
| **Adversary** | `meta-llama/Llama-3.1-70B-Instruct` | Generates diverse attacks using randomized strategies (Prefix, Style, Logic). Quantized to 4-bit. |
| **Defense** | `microsoft/deberta-v3-large` | The safety filter. Updated continuously via **LoRA**. |
| **Oracle** | `meta-llama/Llama-Guard-3-8B` | The objective judge providing ground-truth labels for retraining. |

### Key Engineering Features
* **Micro-Batching:** Decoupled logical batch size ($N=200$) from physical batch size ($N=32$) to prevent OOM errors on single-GPU setups while maintaining statistical significance.
* **Safety Calibration:** Implemented biased pre-training (90% Safe / 10% Harmful) to prevent initial "Brick Wall" behavior (100% False Positive Rate).
* **Dampened Learning:** Tuned learning rates and epochs (1 epoch per cycle) to prevent catastrophic forgetting and ensure stable convergence.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA 12.x compatible GPU (A100 80GB or H200 recommended)
- Hugging Face Access Token (for Llama 3.1 models)

### Installation
```bash
git clone https://github.com/nabinadhikariofficial/TML_LLM_Defense_via_Feedback.git
cd self-hardening-llm-defense
```

# Create environment
```bash
conda create -n defense_loop python=3.10
conda activate defense_loop
```
# Install dependencies
```bash
pip install torch transformers peft datasets bitsandbytes accelerate wandb
```
## 🚀 Usage

### **Prepare Datasets**
```bash
python prepare_data.py
```
### Execute the full defense loop:
```bash
python main.py
```

## 📊 Results & Analysis

We performed an ablation study comparing self-hardening performance when training defenses against a **Weak 8B adversary** versus a **Strong 70B adversary** across **10 iterative cycles**.

### **Comparison: 8B vs. 70B Adversary**

| Metric | 8B Adversary (Weak) | 70B Adversary (Strong) | Analysis |
|--------|----------------------|--------------------------|----------|
| **Initial Threat** | Low (7.5% ASR) | High (15.5% ASR) | 70B finds ~2× more initial vulnerabilities. |
| **Generalization** | Poor (+0.6%) | Strong (+1.2%) | 70B significantly reduces vulnerability to unseen attack types. |
| **Failure Mode** | Panic Overfitting | Gradual Trade-off | 8B causes sudden spike in FPR (25%). 70B yields smooth robustness–utility curve. |

### **Key Finding**

> **Adversarial capability determines defensive robustness.**  
Weak adversaries lead to overfitting and keyword-based heuristics, while strong adversaries force the model to learn genuine semantic safety features.

---

## 🔮 Future Work

### **Experience Replay**
Mitigate rising FPR ("Alignment Tax") by injecting stable **Golden Prompts** into retraining batches.

### **Direct Preference Optimization (DPO)**
Replace SFT with DPO to directly optimize the **safety–utility trade-off**.

### **Multi-Turn Defense**
Extend the `DefenseModule` context window to detect **Crescendo Attacks**, where malicious intent emerges gradually across multiple turns.

---

## 🤝 Contributing

Contributions are welcome!

Please see the **CONTRIBUTING.md** file for:
- Development guidelines  
- Code of conduct  
- Pull request workflow  

---
