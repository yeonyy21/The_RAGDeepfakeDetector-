# The_RAGDeepfakeDetector-
🤖 Hybrid RAG-based Forensic Framework
A two-stage deepfake detection framework that combines pixel-level analysis (Stage 1) with contextual, logic-based analysis (Stage 2).

This system is designed to catch not only microscopic pixel artifacts but also macroscopic logical flaws (e.g., anatomical errors like 6 fingers) that simpler detectors might miss.

⚙️ How It Works
🔬 Stage 1: Pixel-Level Analysis
Models: MTCNN (Face Detection) + XceptionNet (Classification)

Purpose: Scans the detected face for technical, pixel-level manipulation artifacts.

Output: A Suspicion Score & Grad-CAM heatmap.

🧠 Stage 2: Contextual & RAG Analysis
Models: BLIP (Captioning) + LLaVA (Reasoning) + BGE (Embeddings)

Technique: RAG (Retrieval-Augmented Generation) with a FAISS-GPU vector index.

Purpose:

BLIP captions the image (e.g., "a man with six fingers").

RAG searches a forensic KB for relevant facts (e.g., "AI models fail to render hands").

LLaVA synthesizes all evidence (image + pixel score + KB) for a final, logic-based verdict.

✅ Final Synthesis
The LLaVA model combines evidence from both stages. A critical contextual failure (like 6 fingers) will override a low pixel score, ensuring logical flaws are prioritized.

💻 Installation (CUDA 12.1)
Bash

# PyTorch & GPU Core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu-cu121 bitsandbytes accelerate

# AI Models & Utilities
pip install transformers sentence-transformers timm facenet-pytorch opencv-python Pillowfailure (like an anatomical error) will override a low pixel-level score, ensuring high-level logical flaws are not missed.
