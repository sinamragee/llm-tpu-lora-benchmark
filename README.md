# Parameter-Efficient Fine-Tuning (LoRA) & Memory Optimization for Open-Weight LLMs on TPUs

[![Framework: JAX / PyTorch-XLA](https://shields.io)](#)
[![Hardware: Cloud TPU](https://shields.io)](#)
[![License: Apache 2.0](https://shields.io)](#)

## 📌 Overview
This repository contains open-source training pipelines, configuration templates, and hardware utilization benchmarks for **memory-efficient fine-tuning (LoRA & Quantization)** of open-weight Large Language Models (**7B to 30B parameters**, including **Gemma** and **Qwen** architectures) on **TPU accelerators**.

The primary objective of this research project is to evaluate training throughput (tokens/sec), HBM memory scaling, and convergence stability across multilingual and reasoning datasets using **JAX** and **PyTorch/XLA**.

---

## 🗂️ Repository Structure
```text
├── configs/                # Hyperparameter & LoRA rank configurations (7B, 14B, 27B/32B)
├── scripts/
│   ├── preprocess_data.py  # Dataset tokenization & sequence packing pipeline
│   └── train_xla_lora.py   # Distributed PyTorch/XLA LoRA fine-tuning script
├── benchmarks/             # Throughput & HBM memory profiling logs
└── README.md
```

---

## 🚀 Project Roadmap
- [x] **Phase 1:** Dataset tokenization, sequence length optimization, and baseline memory profiling on Kaggle TPUs.
- [ ] **Phase 2:** Scaling distributed LoRA fine-tuning workflows to Cloud TPU slices.
- [ ] **Phase 3:** Comparative benchmarking between JAX and PyTorch/XLA on 27B–32B models.
- [ ] **Phase 4:** Releasing open-source training recipes, benchmark metrics, and evaluation scripts.

---

## 🛠️ Tech Stack
* **Accelerator:** Google Cloud TPU (XLA Architecture)
* **Core Frameworks:** JAX, PyTorch/XLA (`torch_xla`)
* **Libraries:** Hugging Face `transformers`, `peft`, `datasets`, `accelerate`
* **Target Models:** Gemma (9B–27B), Qwen (7B–32B)

---

## 📄 License & Acknowledgments
All scripts and benchmark configurations in this repository are released under the **Apache 2.0 License** for the open-source AI research community.
