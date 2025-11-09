# ==============================================================================
# A Hybrid RAG-based Forensic Framework for Deepfake Detection
# ==============================================================================
# This script implements the two-stage deepfake detection pipeline as described
# in the research paper.
# Stage 1: Pixel-level analysis using a CNN (XceptionNet) with Grad-CAM.
# Stage 2: Contextual analysis using a RAG framework with an LMM (LLaVA).
# ==============================================================================

import torch
import torch.nn.functional as F
import faiss
import numpy as np
import json
import cv2
from PIL import Image
from typing import List, Dict, Any

# --- Core AI Libraries ---
import timm
from facenet_pytorch import MTCNN
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# --- 1. Global Configuration ---
print("▶ [Config] Setting up global configuration...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Use bfloat16 for better performance on modern GPUs if supported, otherwise float16
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print(f"▶ [Config] Using device: {DEVICE} with dtype: {DTYPE}")

# --- 2. Forensic Knowledge Base (KB) ---
# This curated database powers the RAG component.
# In a real-world scenario, this would be a large, continuously updated database.
KNOWLEDGE_DATA = [
    {
        'doc_id': 'anat_001',
        'content': 'Forensic Principle: Generative models often fail to render complex human anatomy correctly, especially hands, fingers, and teeth. Look for an incorrect number of fingers, unnatural shapes, or twisted positions.'
    },
    {
        'doc_id': 'phys_004',
        'content': 'Forensic Principle: In face-swap deepfakes, inconsistent lighting between the synthetically added face and the original video\'s environment is a common artifact. Check for mismatched shadow directions and ambient light color.'
    },
    {
        'doc_id': 'ttp_sex_003',
        'content': 'Criminal TTP: A common deepfake sextortion workflow involves: 1) Collecting public photos from social media, 2) Creating non-consensual synthetic imagery (NCII), and 3) Threatening distribution to extort money.'
    },
    {
        'doc_id': 'tech_012',
        'content': 'Technical Artifact: High-frequency inconsistencies or "fizzing" artifacts are often visible around the edges of a manipulated facial area, especially in lower-quality deepfake videos.'
    }
]

# --- 3. Component Classes ---

class PixelDetector:
    """Component 1: Microscopic analysis via CNN for technical evidence."""
    def __init__(self, target_size=299):
        print("▶ [Component 1] Initializing PixelDetector (XceptionNet)...")
        self.target_size = target_size
        # Use keep_all=False to get only the most probable face
        self.face_detector = MTCNN(keep_all=False, device=DEVICE)
        self.model = timm.create_model('xception', pretrained=True, num_classes=2).to(DEVICE)
        self.model.eval()

        # Hooks for Grad-CAM
        self.gradients = None
        self.activations = None
        self.target_layer = self.model.conv4
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
        print("▶ [Component 1] PixelDetector is ready.")

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def _preprocess_face(self, face_img: Image.Image) -> torch.Tensor:
        face_img = face_img.resize((self.target_size, self.target_size), Image.BILINEAR)
        face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float() / 255.0
        # Normalization for XceptionNet
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        face_tensor = (face_tensor - mean) / std
        return face_tensor.unsqueeze(0).to(DEVICE)

    def analyze(self, image_path: str) -> Dict[str, Any]:
        print("🔬 [Stage 1] Running Pixel-Level Analysis...")
        try:
            img = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            return {"error": f"File not found: {image_path}"}

        boxes, _ = self.face_detector.detect(img)
        if boxes is None:
            return {"score": 0.0, "suspicious_area_desc": "No face detected.", "heatmap": None}

        # Crop the single detected face
        box = boxes[0]
        face = img.crop(box)
        input_tensor = self._preprocess_face(face)

        # 1. Perform the forward pass with gradients enabled to build the computation graph
        logits = self.model(input_tensor)

        # 2. Calculate suspicion score (this part doesn't need gradients)
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            suspicion_score = probs[0, 1].item() # Assuming class 1 is 'Fake'

        # 3. Generate Grad-CAM heatmap using the logits that have a grad_fn
        self.model.zero_grad()
        score_for_grad_cam = logits[:, 1] # Target the 'Fake' class for explanation
        score_for_grad_cam.backward()

        if self.gradients is None or self.activations is None:
            return {"error": "Could not generate Grad-CAM. Gradients or activations are missing."}

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        
        # Weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        heatmap_resized = cv2.resize(heatmap, (face.width, face.height))
        
        print(f"📊 [Stage 1] Pixel Analysis Result: Suspicion Score = {suspicion_score:.2f}")
        return {
            "score": suspicion_score,
            "suspicious_area_desc": "Pixel-level analysis indicates potential manipulation artifacts.",
            "heatmap": heatmap_resized
        }

class KnowledgeBase:
    """Manages forensic knowledge and FAISS-GPU vector search."""
    def __init__(self, embedding_model_name: str = 'BAAI/bge-large-en-v1.5'):
        print("▶ [KB] Initializing Knowledge Base...")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=DEVICE)
        self.documents = []
        self.gpu_index = None
        self.doc_id_map = {}
        print("▶ [KB] Knowledge Base ready for data.")

    def build(self, knowledge_data: List[Dict[str, str]]):
        print("🧠 [KB] Building Knowledge Base and FAISS-GPU Index...")
        self.documents = knowledge_data
        contents = [doc['content'] for doc in self.documents]
        # Generate embeddings on GPU, then move to CPU for FAISS
        embeddings = self.embedding_model.encode(contents, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()
        
        dimension = embeddings.shape[1]
        cpu_index = faiss.IndexFlatL2(dimension)
        cpu_index.add(embeddings)
        
        # Move index to GPU for fast searching
        res = faiss.StandardGpuResources()
        self.gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        
        for i, doc in enumerate(self.documents):
            self.doc_id_map[i] = doc['doc_id']
        print(f"✅ [KB] Knowledge Base indexed on GPU. {len(self.documents)} documents ready.")

    def search(self, query: str, k: int = 1) -> List[Dict[str, str]]:
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        # FAISS on GPU expects a torch tensor on GPU or numpy array on CPU
        distances, indices = self.gpu_index.search(query_embedding.cpu().numpy(), k)
        
        results = []
        for i in indices[0]:
            if i != -1: # FAISS returns -1 for no result
                doc_id = self.doc_id_map.get(i)
                if doc_id:
                    doc = next((doc for doc in self.documents if doc['doc_id'] == doc_id), None)
                    if doc:
                        results.append(doc)
        return results

class RAGDeepfakeDetector:
    """Component 2: Contextual analysis via RAG and LMM."""
    def __init__(self, knowledge_base: KnowledgeBase):
        print("▶ [Component 2] Initializing RAGDeepfakeDetector (BLIP, LLaVA)...")
        self.kb = knowledge_base
        
        # Image Captioning Model
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large", torch_dtype=DTYPE).to(DEVICE)
            
        # 4-bit Quantization for LLaVA to save memory
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE
        )
        
        # LMM for final report generation
        model_id = "llava-hf/llava-1.5-7b-hf"
        self.generator_processor = AutoProcessor.from_pretrained(model_id)
        self.generator_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("✅ [Component 2] RAG Detector is ready.")

    def _generate_dynamic_queries(self, image: Image.Image, pixel_analysis: Dict[str, Any]) -> List[str]:
        print("🖼️  [Stage 2] Generating image caption for situational awareness...")
        inputs = self.caption_processor(image, return_tensors="pt").to(DEVICE, DTYPE)
        out = self.caption_model.generate(**inputs, max_new_tokens=50)
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True).strip()
        print(f"🖼️  [Stage 2] Image Caption: \"{caption}\"")
        
        # Integrate pixel analysis findings into the queries
        pixel_clue = (f"A preliminary pixel-level analysis by a CNN model reported a high "
                      f"suspicion score of {pixel_analysis['score']:.2f}. The model indicated potential "
                      f"manipulation artifacts as a primary area of interest.")
        
        return [
            f"Given an image of '{caption}', analyze for anatomical errors, especially on hands and fingers. {pixel_clue}",
            f"Analyze the image described as '{caption}' for physical inconsistencies like unnatural lighting, shadows, and reflections.",
            "Check the image for digital forensic artifacts such as JPEG ghosts, compression anomalies, or unnatural textures."
        ]

    def generate_report(self, image: Image.Image, context_docs: List[Dict[str, str]], pixel_analysis: Dict[str, Any]) -> Dict[str, Any]:
        context_str = "\n\n".join([f"ID: {doc['doc_id']}\n{doc['content']}" for doc in context_docs])
        pixel_analysis_summary = (f"A pre-screening with a CNN-based detector yielded a suspicion score of {pixel_analysis['score']:.2f}. "
                                  f"The model's explainability heatmap highlighted potential pixel-level manipulation artifacts.")

        prompt = f"""USER: <image>
You are a world-class digital forensic analyst. Your task is to synthesize findings from a preliminary pixel-level analysis and a logical analysis based on a knowledge base to produce a final, evidence-based report on a given image.

**Instructions:**
1.  **Integrate Pixel Analysis:** Consider the preliminary findings from the CNN model as a key piece of evidence.
2.  **Analyze against Knowledge:** Compare the image against each principle in the "Retrieved Forensic Knowledge" section.
3.  **Synthesize and Conclude:** Form a final verdict by combining all evidence. If a critical rule (like human anatomy) is violated, the verdict MUST be "Fake," regardless of other findings.
4.  **Provide JSON Output:** Your entire response must be a single, valid JSON object. Do not add any text before or after the JSON. Follow the example format precisely.

---
**Preliminary Pixel Analysis:**
{pixel_analysis_summary}

**Retrieved Forensic Knowledge:**
{context_str}
---
**Example of a perfect JSON output:**
{{
  "verdict": "Fake",
  "confidence": 0.98,
  "pixel_analysis_summary": {{
    "suspicion_score": 0.85,
    "finding": "The CNN model detected high-frequency inconsistencies typical of generative models."
  }},
  "reasoning": "The final verdict is 'Fake' based on a critical contextual failure. The system identified an anatomical impossibility (six fingers on one hand), which is a known artifact of generative models as confirmed by retrieved knowledge document 'anat_001'. This high-level logical error strongly outweighs other findings."
}}
---
ASSISTANT:
"""
        inputs = self.generator_processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        print("\n📝 [Stage 2] Generating final report with LLaVA model...")
        
        # Generate the report
        output = self.generator_model.generate(**inputs, max_new_tokens=768, do_sample=False)
        response_text = self.generator_processor.decode(output[0], skip_special_tokens=True)
        
        # Clean and parse the JSON output
        try:
            # The model's response starts after the final "ASSISTANT:"
            json_str = response_text.split("ASSISTANT:")[-1].strip()
            # Remove potential markdown code blocks
            if json_str.startswith("```json"):
                json_str = json_str[len("```json"):].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-len("```")].strip()
            
            # Load the JSON string into a Python dictionary
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError) as e:
            print(f"❌ ERROR: Could not parse JSON from model output. Error: {e}\nRaw response: {response_text}")
            return {
                "verdict": "Parsing Error",
                "confidence": 0.0,
                "reasoning": "Failed to parse the JSON output from the LLaVA model.",
                "raw_output": response_text
            }

class HybridSystem:
    """Orchestrates the full two-stage detection pipeline."""
    def __init__(self):
        self.pixel_detector = PixelDetector()
        self.kb = KnowledgeBase()
        self.kb.build(KNOWLEDGE_DATA)
        self.rag_detector = RAGDeepfakeDetector(self.kb)

    def detect(self, image_path: str):
        print(f"\n🚀 Starting HYBRID forensic analysis for '{image_path}'...")
        
        # 1. Perform Stage 1: Pixel-level analysis
        pixel_analysis = self.pixel_detector.analyze(image_path)
        if "error" in pixel_analysis:
            print(f"❌ ERROR in Stage 1: {pixel_analysis['error']}")
            return

        # 2. Perform Stage 2: RAG-based contextual analysis
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"❌ ERROR: File not found for Stage 2: {image_path}")
            return
            
        queries = self.rag_detector._generate_dynamic_queries(image, pixel_analysis)
        
        print("\n🔍 [Stage 2] Searching Knowledge Base with dynamic queries...")
        # Use a set to avoid duplicate documents
        retrieved_docs_set = set()
        for q in queries:
            docs = self.kb.search(q, k=1)
            for doc in docs:
                retrieved_docs_set.add(json.dumps(doc))
        
        retrieved_docs = [json.loads(s) for s in retrieved_docs_set]
        print(f"🔍 [Stage 2] Retrieved {len(retrieved_docs)} unique forensic documents.")
        
        final_report = self.rag_detector.generate_report(image, retrieved_docs, pixel_analysis)
        
        print("\n--- ✅ FINAL HYBRID FORENSIC REPORT ---")
        print(json.dumps(final_report, indent=2, ensure_ascii=False))
        print("--- 🏁 ANALYSIS COMPLETE ---")

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    print("==============================================")
    print("= Hybrid Deepfake Detection Framework Start  =")
    print("==============================================")
    print("NOTE: The first run will download large AI models and may take a long time.")
    
    # Initialize the entire system
    hybrid_system = HybridSystem()
    
    # Specify the image to analyze
    # PLEASE REPLACE "path/to/your/image.png" WITH THE ACTUAL FILE PATH
    image_to_analyze = "fingersix.png"
    
    # Run the detection pipeline
    hybrid_system.detect(image_to_analyze)
