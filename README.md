# 🚀 LlamaRefine - Refining TinyLlama with Supervised Fine-Tuning and Human Preferences (DPO)

This repository contains a complete **notebook-based pipeline** for fine-tuning the [TinyLlama](https://huggingface.co/TinyLlama) language model using **Direct Preference Optimization (DPO)** and **Supervised Fine-Tuning (SFT)** with **LoRA adapters**. It leverages **Unsloth** and **TRL** for efficient and scalable training on low-resource hardware.

---

## 📌 What is DPO (Direct Preference Optimization)?

**DPO** is a lightweight reinforcement learning technique that allows LLMs to learn **preferences between good and bad responses** using only ranked pairs — without needing a separate reward model or complex PPO setup.

> 📥 Input: Prompt + Chosen response + Rejected response  
> 🎯 Goal: Make the model assign higher likelihood to the "chosen" over the "rejected"

This approach directly aligns model behavior with **human preferences** and is highly efficient.

---

## 🧠 Final Unified Model Capabilities

After the full pipeline, your final model can:

- 🧠 Retain **TinyLlama’s original knowledge** (pretraining)
- 🎓 Follow instructions and tasks with **SFT**
- 📊 Prefer better responses with **DPO preference learning**

✅ You get a **single unified model**, merged and ready for inference, without needing LoRA adapters during runtime.

---

## 🚀 What This Project Does

- ✅ Fine-tunes TinyLlama using helpful outputs (SFT)
- ✅ Optimizes the model to prefer better responses (DPO)
- ✅ Uses LoRA for low-memory training
- ✅ Merges everything into one deployable model

---

## 🔧 Tech Stack
- 🤖 TinyLlama (base model)
- ⚡ Unsloth (fast fine-tuning)
- 🔁 TRL (Transformers RL) (DPO implementation)
- 📦 PEFT (LoRA adapters)
- 🤗 Hugging Face ecosystem (datasets, transformers, accelerate)

---

## 📒 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_SFT.ipynb` | Performs Supervised Fine-Tuning (SFT) using LoRA |
| `02_DPO.ipynb` | Performs Direct Preference Optimization (DPO) and merges adapters |
| `03_Inference.py` | Runs inference using the final merged model |

---

## 📁 Project Structure

```bash
.
├── 01_SFT.ipynb            # SFT with LoRA
├── 02_DPO.ipynb            # DPO training + merging adapters
├── 03_Inference.ipynb      # Inference from final model
└── README.md               # This file
```

---

## 📦 Dataset

- **Used for SFT**: Custom dataset from [`NvidiaDocumentationQandApairs.csv`](https://www.kaggle.com/datasets/gondimalladeepesh/nvidia-documentation-question-and-answer-pairs), consisting of high-quality Q&A pairs extracted from NVIDIA technical documentation.
- **Used for DPO**: [`yitingxie/rlhf-reward-datasets`](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets)  
  Format: Each entry contains a prompt, a chosen response (preferred), and a rejected response.

---

## 🏋️‍♂️ Training Pipeline

1. 🧪 Supervised Fine-Tuning (SFT)

```bash
run: 01_SFT.ipynb
```
- Fine-tunes TinyLlama with helpful responses using LoRA.
- Saves the LoRA adapter (sft_adapter).

2. 🎯 Direct Preference Optimization (DPO) and 🔀 Merge Adapters

```bash
Run: 02_DPO.ipynb
```

- Uses sft_adapter as a base.
- Trains the model to prefer "chosen" over "rejected" outputs.
- Produces a dpo_adapter.
- Merges base model + SFT adapter + DPO adapter into one final model.


3. 🤖 Inference

```bash
Run: Inference.py
```
Test the final model with custom prompts.

---

## 🧠 Model Behavior
After training, the final model:
- Understands natural prompts thanks to SFT
- Generates preferred and high-quality responses thanks to DPO
- Runs standalone (LoRA adapters merged)

---

## 💡 Use Case
This project is ideal for:

- Creating lightweight RLHF-style LLMs without PPO (Proximal Policy Optimization)
- Instruction-following LLMs tuned for preference-aligned outputs
- Prototyping custom assistants, chatbots, or tutor bots using small models

---

## 🧪 Proof of Concept (POC)

The following screenshots show the directory structure , model behavior before and after DPO preference tuning:
![image](https://github.com/user-attachments/assets/2238898a-f5a3-4572-8b08-38f9bcfb7b2d)


![image](https://github.com/user-attachments/assets/b4bba7a7-3efc-4aa2-9f52-cb02f3fdcc9c)

---

## 📤 Future Enhancements
- ✅ Upload final model to Hugging Face Hub
- 📈 Evaluate model using MT-Bench or AlpacaEval
- 💬 Add Gradio/Streamlit chatbot UI
- 🔗 Integrate with LangChain or RAG pipeline
