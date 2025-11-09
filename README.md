# The_RAGDeepfakeDetector-
A Hybrid RAG-based Forensic Framework for Deepfake Detection
This project is a two-stage deepfake detection framework. It combines pixel-level analysis (Stage 1) with contextual, logic-based analysis (Stage 2) to produce a robust forensic report.

The system is designed to catch not only microscopic pixel artifacts but also macroscopic logical flaws (e.g., anatomical errors like 6 fingers), which simple detectors might miss.

How It Works
Stage 1: Pixel-Level Analysis (PixelDetector)

Models: MTCNN (for face detection) + XceptionNet (for classification).

Purpose: Analyzes the detected face for technical, pixel-level manipulation artifacts.

Output: A Suspicion Score and a Grad-CAM heatmap for explainability.

Stage 2: Contextual Analysis (RAGDeepfakeDetector)

Models: BLIP (for image captioning) + LLaVA (for reasoning) + BGE (for embeddings).

Technique: RAG (Retrieval-Augmented Generation) with a FAISS-GPU vector index.

Purpose:

BLIP captions the image (e.g., "a man with six fingers").

RAG searches a forensic knowledge base for relevant facts (e.g., "AI models often fail to render hands correctly").

LLaVA synthesizes the image, the pixel score, and the retrieved knowledge to make a final, logic-based decision.

Final Synthesis

The LLaVA model combines evidence from both stages.

Key Feature: A critical contextual failure (like an anatomical error) will override a low pixel-level score, ensuring high-level logical flaws are not missed.
