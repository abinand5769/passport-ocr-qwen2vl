# streamlit_passport_ocr.py
import streamlit as st
import json
import os
from datetime import datetime
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

schema = {
    "document_type": "passport",
    "country_code": "",
    "full_name_en": "",
    "full_name_ar": "",
    "passport_number": "",
    "nationality_en": "",
    "nationality_ar": "",
    "date_of_birth": "",
    "place_of_birth_en": "",
    "place_of_birth_ar": "",
    "sex": "",
    "date_of_issue": "",
    "date_of_expiry": "",
    "issuing_authority_en": "",
    "issuing_authority_ar": "",
    "holder_signature": "",
    "mrz_lines": ["", ""],
    "extracted_fields_count": 0
}

class PassportOCRExtractor:
    def _init_(self):
        self.vision_model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @st.cache_resource
    def load_models(_self):
        try:
            with st.spinner("Loading Qwen2-VL-2B model... This will take several minutes on first run."):
                logger.info("Loading model...")
                
                _self.vision_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2-VL-2B-Instruct",
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=True
                )
                
                _self.processor = AutoProcessor.from_pretrained(
                    "Qwen/Qwen2-VL-2B-Instruct",
                    trust_remote_code=True
                )
                
            st.success(f"Model loaded successfully on {_self.device}!")
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
        
    def preprocess_image(self, uploaded_file):
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            
            alpha = 1.2
            beta = 10
            enhanced = cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta)
            pil_image = Image.fromarray(enhanced).convert("RGB")
            
            return pil_image, image
            
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None, None

    def create_extraction_prompt(self):
        return f"""You are an expert multilingual OCR system specializing in passport document analysis.

TASK: Extract ALL information from this passport image containing Arabic and English text.

REQUIREMENTS:
1. Read text in BOTH Arabic and English accurately
2. Extract passport fields from both language sections
3. For dates, use DD/MM/YYYY format
4. For MRZ lines, capture complete machine-readable zone

Return ONLY a valid JSON object with this structure:
{json.dumps(schema, ensure_ascii=False, indent=2)}

IMPORTANT: Response must be ONLY the JSON object, no additional text."""

    def extract_with_qwen2vl(self, image):
        try:
            prompt = self.create_extraction_prompt()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                generated_ids = self.vision_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=False
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            st.error(f"Error during extraction: {e}")
            return None

    def parse_json_response(self, response):
        try:
            response = response.strip()
            
            if "json" in response:
                start = response.find("json") + 7
                end = response.find("", start)
                response = response[start:end].strip()
            elif "" in response:
                start = response.find("") + 3
                end = response.rfind("")
                response = response[start:end].strip()
            
            if not response.startswith("{"):
                start = response.find("{")
                if start != -1:
                    response = response[start:]
            
            if not response.endswith("}"):
                end = response.rfind("}") + 1
                if end > 0:
                    response = response[:end]
            
            data = json.loads(response)
            
            for key in schema.keys():
                if key not in data:
                    data[key] = schema[key]
            
            non_empty_fields = sum(1 for value in data.values() 
                                 if value and value != "" and value != ["", ""])
            data["extracted_fields_count"] = non_empty_fields
            
            return data
            
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {e}")
            with st.expander("View Raw Response"):
                st.text(response[:1000])
            return None

    def extract_passport_data(self, uploaded_file):
        enhanced_image, original_image = self.preprocess_image(uploaded_file)
        if enhanced_image is None:
            return None, None
        
        with st.spinner("Extracting passport data..."):
            response = self.extract_with_qwen2vl(enhanced_image)
        
        if response is None:
            return None, None
        
        with st.expander("View Raw Model Response"):
            st.text(response[:1000])
        
        passport_data = self.parse_json_response(response)
        if passport_data is None:
            return None, None
        
        return passport_data, original_image

def create_downloadable_json(data, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "extraction_timestamp": timestamp,
        "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_used": "Qwen/Qwen2-VL-2B-Instruct",
        "source_image": filename,
        "extracted_fields": data.get("extracted_fields_count", 0),
        "passport_data": data
    }
    
    return json.dumps(results, ensure_ascii=False, indent=2)

def main():
    st.set_page_config(
        page_title="Passport OCR Extractor",
        page_icon="🛂",
        layout="wide"
    )
    
    st.title("🛂 Multilingual Passport OCR Extractor")
    st.markdown("Upload a passport image to extract Arabic and English text")
    
    if 'extractor' not in st.session_state:
        st.session_state.extractor = PassportOCRExtractor()
    
    if st.session_state.extractor.vision_model is None:
        if not st.session_state.extractor.load_models():
            st.stop()
    
    uploaded_file = st.file_uploader(
        "Choose a passport image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"]
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"File: {uploaded_file.name}", use_column_width=True)
            st.info(f"*File:* {uploaded_file.name}\n*Size:* {image.size}")
        
        with col2:
            st.subheader("🔍 Extraction Results")
            
            if st.button("🚀 Extract Passport Data", type="primary"):
                passport_data, original_image = st.session_state.extractor.extract_passport_data(uploaded_file)
                
                if passport_data is not None:
                    st.session_state.passport_data = passport_data
                    st.session_state.filename = uploaded_file.name
                    st.success("Extraction completed successfully!")
        
        if 'passport_data' in st.session_state:
            st.subheader("📋 Extracted JSON Data")
            st.json(st.session_state.passport_data)
            
            st.divider()
            
            json_content = create_downloadable_json(
                st.session_state.passport_data, 
                st.session_state.filename
            )
            
            st.download_button(
                label="📥 Download JSON Results",
                data=json_content,
                file_name=f"{os.path.splitext(st.session_state.filename)[0]}passport_data{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                type="primary"
            )

if _name_ == "_main_":
    main()