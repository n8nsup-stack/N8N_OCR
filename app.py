import os
import io
import json
import torch
import re
import numpy as np
import cv2
import hashlib
import imagehash
import sys
import types
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz, process
from supabase import create_client, Client
from PIL import Image

# --- COMPATIBILITY SHIM (Must be before PaddleOCR import) ---
try:
    import langchain_core.documents
    import langchain_text_splitters
    m_doc = types.ModuleType("langchain.docstore.document")
    m_doc.Document = langchain_core.documents.Document
    sys.modules["langchain.docstore.document"] = m_doc
    m_split = types.ModuleType("langchain.text_splitter")
    m_split.RecursiveCharacterTextSplitter = langchain_text_splitters.RecursiveCharacterTextSplitter
    sys.modules["langchain.text_splitter"] = m_split
except ImportError:
    pass

from paddleocr import PaddleOCR

os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_new_executor'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

app = FastAPI(title="OCR-NLP-Engine v3.1")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = None
model = None
reader = None 
MASTER_PRODUCTS = []
MASTER_SKUS = []
MASTER_EMBEDDINGS = None

class MatchRequest(BaseModel):
    lines: List[str]

@app.get("/")
def home():
    return {
        "status": "online", 
        "service": "OCR-NLP-Engine", 
        "version": "v3.1",
        "standardized_schema": ["total", "used", "remaining", "detected_phone"]
    }

@app.on_event("startup")
async def startup_event():
    global supabase, model, reader
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception: 
            pass
            
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    reader = PaddleOCR(use_angle_cls=True, lang='en') 
    await refresh_data_internal()

async def refresh_data_internal():
    global MASTER_PRODUCTS, MASTER_EMBEDDINGS, MASTER_SKUS
    if not supabase: return
    try:
        response = supabase.table("master_products").select("id, name, sku, embedding").execute()
        data = response.data
        if not data: return
        names, skus, embs = [], [], []
        for item in data:
            if item.get('name'):
                names.append(item['name'])
                skus.append(item.get('sku', 'N/A'))
                emb = item.get('embedding')
                if isinstance(emb, str):
                    emb = json.loads(emb.replace('{', '[').replace('}', ']'))
                embs.append(emb)
        MASTER_PRODUCTS, MASTER_SKUS = names, skus
        if embs:
            MASTER_EMBEDDINGS = torch.tensor(embs, dtype=torch.float)
    except Exception: pass

# --- Shared Utility Functions ---

async def get_clean_lines_from_file(file: UploadFile):
    """General OCR extraction for invoices and documents"""
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img)
    
    # PaddleOCR performs best on RGB/BGR directly
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    if img_bgr.shape[1] < 1200:
        scale = 1200.0 / img_bgr.shape[1]
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
    res = reader.ocr(img_bgr)
    raw_lines = [line[1][0] for line in res[0]] if res and res[0] else []
    
    clean_lines = []
    for text in raw_lines:
        text = re.sub(r'(?i)(\d+)\s*(om|omi|m1|om1|ml|mi)', r'\1ml', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'[^\w\sml]$|[\)\!\]\}]', '', text).strip()
        if len(text) > 3:
            clean_lines.append(text)
    
    await file.seek(0)
    return clean_lines

def run_match(text):
    if not MASTER_PRODUCTS or MASTER_EMBEDDINGS is None: return None
    
    input_emb = model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(input_emb, MASTER_EMBEDDINGS)
    best_idx = torch.argmax(scores).item()
    s_score = scores[0][best_idx].item()
    
    f_res = process.extractOne(text, MASTER_PRODUCTS, scorer=fuzz.token_set_ratio)
    f_name, f_score = f_res[0], f_res[1] / 100.0
    
    final_score, best_name = (s_score, MASTER_PRODUCTS[best_idx]) if s_score >= f_score else (f_score, f_name)

    if final_score >= 0.30:
        idx = MASTER_PRODUCTS.index(best_name)
        return {
            "master_name": best_name, 
            "sku": MASTER_SKUS[idx], 
            "confidence": round(final_score * 100, 2), 
            "original_text": text
        }
    return None

# --- Endpoints ---

@app.post("/get-raw-ocr")
async def get_raw_ocr(file: UploadFile = File(...)):
    lines = await get_clean_lines_from_file(file)
    return {"lines": lines}

@app.post("/process-quota")
async def process_quota(file: UploadFile = File(...)):
    """
    v3.1: Spatial & Context-Aware Multi-Provider OCR Engine
    - Preserves subtle/gray text (e.g. 'Dari 24.0 GB') without destructive thresholding
    - Uses spatial bounding boxes + keyword filtering to ignore bottom ad banners
      (e.g. 'Google Gemini Penyimpanan 200GB') and package headers (e.g. 'Happy 175GB 180 hr')
    - Supports 10 to 14 digit Indonesian phone numbers (including 13-digit Tri numbers)
    """
    content = await file.read()
    
    # 1. Hashes for duplicate detection
    md5_hash = hashlib.md5(content).hexdigest()
    
    # 2. Image Processing (3% top crop preserves greeting/phone numbers in Bima+ & MyTelkomsel)
    img_pil = Image.open(io.BytesIO(content)).convert("RGB")
    img_np = np.array(img_pil)
    
    h, w = img_np.shape[:2]
    crop_top = int(h * 0.03)
    crop_bottom = int(h * 0.96)
    img_cropped = img_np[crop_top:crop_bottom, :]
    
    phash_val = str(imagehash.phash(Image.fromarray(img_cropped)))
    
    # 3. High-Quality BGR preparation for PaddleOCR
    img_bgr = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)
    if img_bgr.shape[1] < 1080:
        scale = 1080.0 / img_bgr.shape[1]
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    proc_h = img_bgr.shape[0]
    
    # 4. PaddleOCR Execution
    res = reader.ocr(img_bgr)
    
    # Fallback to mild CLAHE contrast enhancement if text detection is sparse
    if not res or not res[0] or len(res[0]) < 3:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        res = reader.ocr(enhanced_bgr)
        
    ocr_items = res[0] if res and res[0] else []
    raw_lines = [item[1][0] for item in ocr_items]
    full_text = " ".join(raw_lines).lower().replace(',', '.')
    
    # Filter lines spatially and contextually for quota extraction:
    # Ignore lower-half ad banners (y > 55% of image height) and promo/package lines
    promo_keywords = r'(?:happy|penyimpanan|gemini|google|adobe|bonus|ekstra|undian|menangkan|cashback|180\s*hr|valid\s*\d+)'
    dashboard_lines = []
    for item in ocr_items:
        box = item[0]
        text_line = item[1][0].lower().replace(',', '.')
        y_center = sum(pt[1] for pt in box) / 4.0
        
        # Keep lines in the top 55% dashboard zone that do not match promo/package keywords
        if y_center <= (proc_h * 0.55) and not re.search(promo_keywords, text_line):
            dashboard_lines.append(text_line)
            
    dashboard_text = " ".join(dashboard_lines)
    
    # 5. SEMANTIC EXTRACTION
    total = 0.0
    remaining = 0.0
    used = 0.0
    
    # Pattern A: Check for explicit "Dari X GB" / "of X GB" / "/ X GB" (Telkomsel Halo, XL, Indosat)
    dari_match = re.search(r'(?:dari|of|\/)\s*(\d+(?:\.\d+)?)\s*(gb|mb)', dashboard_text)
    if not dari_match:
        dari_match = re.search(r'(?:dari|of|\/)\s*(\d+(?:\.\d+)?)\s*(gb|mb)', full_text)
        
    # Pattern B: Check for explicit "Terpakai X GB"
    used_match = re.search(r'(?:terpakai|pemakaian|used)\s*[:\s]*(\d+(?:\.\d+)?)\s*(gb|mb)', dashboard_text)
    
    # Collect valid GB / MB values from the clean dashboard zone
    gb_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(gb|mb)', dashboard_text)
    parsed_values = []
    for val_str, unit in gb_matches:
        val = float(val_str)
        if unit == 'mb':
            val = round(val / 1024.0, 2)
        if 0.05 <= val <= 3000:
            parsed_values.append(val)
            
    if dari_match:
        # Explicit Total Quota found (e.g. "Sisa Kuota 17.4 GB Dari 24.0 GB")
        tot_val = float(dari_match.group(1))
        if dari_match.group(2) == 'mb':
            tot_val = round(tot_val / 1024.0, 2)
        total = round(tot_val, 2)
        
        remaining_candidates = [v for v in parsed_values if v < total]
        if remaining_candidates:
            remaining = round(max(remaining_candidates), 2)
        else:
            remaining = total
            
        used = round(max(0.0, total - remaining), 2)
        
    elif used_match and len(parsed_values) >= 2:
        # Explicit Used Quota in GB found
        u_val = float(used_match.group(1))
        if used_match.group(2) == 'mb':
            u_val = round(u_val / 1024.0, 2)
        used = round(u_val, 2)
        total = round(max(parsed_values), 2)
        remaining = round(max(0.0, total - used), 2)
        
    else:
        # Single-value dashboard card (e.g. Bima+ Tri "228.6GB Kuota")
        if parsed_values:
            remaining = round(max(parsed_values), 2)
            total = remaining
            used = 0.0

    # 6. Expiry Date Extraction
    date_match = re.search(r'(?:hingga|sampai|s\/d)\s+(\d{1,2}\s+[a-z]{3,9}\s+\d{4})', full_text)
    if not date_match:
        date_match = re.search(r'berlaku\s+sampai\s+(\d{1,2}\s+[a-z]{3,9}\s+\d{4})', full_text)
    expiry = date_match.group(1).title() if date_match else "Unknown"
    
    # 7. Phone Number Extraction (Supports 10 to 14 digit Indonesian numbers, e.g. 13-digit Tri)
    compact_text = re.sub(r'[\s\-]+', '', " ".join(raw_lines))
    phone_match = re.search(r'(?:08|628)\d{8,12}', compact_text)
    detected_phone = phone_match.group(0) if phone_match else None

    return {
        # Standardized schema
        "total": total,
        "used": used,
        "remaining": remaining,
        "detected_phone": detected_phone,
        
        # Backward-compatible aliases for existing workflows
        "total_quota": total,
        "used_quota": used,
        "remaining_quota": remaining,
        
        # Metadata
        "expiry_date": expiry,
        "md5": md5_hash,
        "phash": phash_val,
        "version": "v3.1"
    }

@app.post("/match-items")
async def match_items(data: MatchRequest):
    return {"matches": [m for m in (run_match(line) for line in data.lines) if m]}

@app.post("/process-invoice")
async def process_invoice(file: UploadFile = File(...)):
    lines = await get_clean_lines_from_file(file)
    return {"matches": [m for m in (run_match(line) for line in lines) if m]}

@app.post("/refresh-data")
async def refresh_data():
    await refresh_data_internal()
    return {"status": "success"}
