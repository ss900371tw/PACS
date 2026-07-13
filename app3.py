import os
import time
import socket
import threading
from io import BytesIO

# 防止特定環境引發相容性崩潰
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ====== Web 相關庫 ======
import requests
from flask import Flask, request, jsonify

# ====== AI / 深度學習相關庫 ======
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel  

# ====== 檢查與指派裝置 ======
device_m1 = "cpu" if torch.cuda.is_available() else "cpu"
# 如果只有單張卡，模型二就必須與模型一共享或依賴 BitsAndBytes 量化
device_m2 = "cpu" if torch.cuda.is_available() else "cpu" 
print(f"🎬 模型一配置裝置: {device_m1.upper()} | 模型二配置裝置: {device_m2.upper()}")

HF_TOKEN = os.getenv("HF_TOKEN", "")

# ====== 模型一設定 (VinDr 疾病分類) ======
m1_base_id = "google/medgemma-1.5-4b-it"
m1_lora_id = "ss900371tw/medgemma-VinDr-lora"  # 已依需求更新為 VinDr-lora

# ====== 模型二設定 (27B 診斷原因推導) ======
m2_base_id = "google/medgemma-1.5-4b-it"

print("==========================================")
print("正在載入 模型一 (MedGemma 4B + VinDr LoRA)...")
processor_m1 = AutoProcessor.from_pretrained(m1_base_id, token=HF_TOKEN)
m1_dtype = torch.bfloat16 if "cuda" in device_m1 else torch.float32

base_m1 = AutoModelForImageTextToText.from_pretrained(
    m1_base_id, torch_dtype=m1_dtype, low_cpu_mem_usage=True, token=HF_TOKEN
).to(device_m1)

model_m1 = PeftModel.from_pretrained(base_m1, m1_lora_id, token=HF_TOKEN)
model_m1 = model_m1.merge_and_unload()
model_m1.eval()

print("==========================================")
print("正在載入 模型二 (MedGemma 27B)...")
# 💡 針對 27B 龐大模型，強烈建議在單卡或顯存吃緊時啟用 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
) if "cuda" in device_m2 else None

processor_m2 = AutoProcessor.from_pretrained(m2_base_id, token=HF_TOKEN)

model_m2 = AutoModelForImageTextToText.from_pretrained(
    m2_base_id,
    torch_dtype=torch.bfloat16 if "cuda" in device_m2 else torch.float32,
    quantization_config=bnb_config,
    device_map=device_m2 if bnb_config is None else "auto", # 若量化則交由 auto 管理分配
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)
model_m2.eval()
print("🎉 雙 AI 模型載入與初始化成功！")


# ====== Flask & Orthanc 初始化 ======
app = Flask(__name__)
ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

processed_sops = {}
CACHE_EXPIRY_SEC = 300  
lock = threading.Lock()
active_processing_locks = {}
processing_dict_lock = threading.Lock()

uploaded_counter = 0
successful_new_instance_ids = []  
counter_lock = threading.Lock()


def clean_expired_cache():
    while True:
        time.sleep(60)
        now = time.time()
        with lock:
            expired = [sop for sop, ts in processed_sops.items() if now - ts > CACHE_EXPIRY_SEC]
            for sop in expired:
                del processed_sops[sop]

threading.Thread(target=clean_expired_cache, daemon=True).start()


def get_instance_uuid_by_sop(sop_instance_uid):
    try:
        lookup_url = f"{ORTHANC_URL}/tools/lookup"
        response = requests.post(lookup_url, data=sop_instance_uid, auth=AUTH, timeout=5)
        if response.status_code == 200:
            res_json = response.json()
            if res_json and len(res_json) > 0:
                return res_json[0]["ID"]
    except Exception as e:
        print(f"Lookup UID 失敗: {e}")
    return None


def analyze_image_with_dual_ai(instance_id):
    """
    管線化串接多階段模型：
    1. 使用模型一辨識 CXR 疾病 (Diagnosis)
    2. 將疾病帶入模型二，引導 27B 模型深入分析影像特徵與形成主因 (Reasoning)
    3. 將疾病帶入模型一 (MedGemma 4B)，獨立推論藥物治療 (Medications) 與治療建議 (Treatment)
    """
    try:
        preview_url = f"{ORTHANC_URL}/instances/{instance_id}/preview"
        response = requests.get(preview_url, auth=AUTH, timeout=10)
        if response.status_code != 200:
            return "Analysis Failed", "N/A", "N/A", "N/A"

        image = Image.open(BytesIO(response.content)).convert("RGB")

        # -----------------------------------------------------------------
        # 🚀 階段一：模型一進行疾病分類 (VinDr LoRA) -> Diagnosis
        # -----------------------------------------------------------------
        m1_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Identify the most likely disease in this image."},
                {"type": "image"}
            ]
        }]
        m1_prompt = processor_m1.apply_chat_template(m1_messages, tokenize=False, add_generation_prompt=True)
        m1_inputs = processor_m1(text=[m1_prompt], images=[[image]], return_tensors="pt").to(device_m1)

        with torch.no_grad():
            m1_outputs = model_m1.generate(
                **m1_inputs, 
                max_new_tokens=20,
                pad_token_id=processor_m1.tokenizer.pad_token_id or processor_m1.tokenizer.eos_token_id
            )
        m1_input_len = m1_inputs["input_ids"].shape[-1]
        diagnosis = processor_m1.decode(m1_outputs[0][m1_input_len:], skip_special_tokens=True).strip()
        print(f" [模型一 診斷結果]: {diagnosis}")

        # -----------------------------------------------------------------
        # 🚀 階段二：模型二進行臨床原因推導 (MedGemma 27B) -> Reasoning
        # -----------------------------------------------------------------
        reasoning_prompt_text = (
            f"This chest X-ray (CXR) image is diagnosed with '{diagnosis}'. "
            f"Based on the visual evidence in this image, explain why this diagnosis was made "
            f"and describe the key radiographic findings supporting it."
        )
        
        m2_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": reasoning_prompt_text},
                {"type": "image"}
            ]
        }]
        m2_prompt = processor_m2.apply_chat_template(m2_messages, tokenize=False, add_generation_prompt=True)
        m2_inputs = processor_m2(text=[m2_prompt], images=[[image]], return_tensors="pt").to(model_m2.device)

        with torch.no_grad():
            m2_outputs = model_m2.generate(
                **m2_inputs,
                max_new_tokens=250, 
                pad_token_id=processor_m2.tokenizer.pad_token_id or processor_m2.tokenizer.eos_token_id
            )
        m2_input_len = m2_inputs["input_ids"].shape[-1]
        reasoning = processor_m2.decode(m2_outputs[0][m2_input_len:], skip_special_tokens=True).strip()
        print(f" [模型二 推理原因]: {reasoning}")

        # -----------------------------------------------------------------
        # 🚀 階段三：使用 MedGemma 4B 推論藥物與治療建議 (並要求明確標記以供解析)
        # -----------------------------------------------------------------
        treatment_prompt_text = (
f"A patient is diagnosed with '{diagnosis}' based on chest X-ray findings, "
            f"with supporting radiographic evidence indicating: '{reasoning}'.\n"
            f"Please provide clinical recommendations strictly using the following format:\n\n"
            f"[Medications]\n(Provide recommended medications, drug classes, or first-line choices here)\n\n"
            f"[Treatment]\n(Provide general treatment, supportive care, and clinical management plans here)\n\n"
            f"Keep the output concise, structured, and professionally written."
        )
        
        m3_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": treatment_prompt_text}
            ]
        }]
        m3_prompt = processor_m1.apply_chat_template(m3_messages, tokenize=False, add_generation_prompt=True)
        m3_inputs = processor_m1(text=[m3_prompt], return_tensors="pt").to(device_m1)

        with torch.no_grad():
            m3_outputs = model_m1.generate(
                **m3_inputs,
                max_new_tokens=350, 
                pad_token_id=processor_m1.tokenizer.pad_token_id or processor_m1.tokenizer.eos_token_id
            )
        m3_input_len = m3_inputs["input_ids"].shape[-1]
        tx_raw = processor_m1.decode(m3_outputs[0][m3_input_len:], skip_special_tokens=True).strip()
        
        # 🧠 解析文字，將 Medications 與 Treatment 分流
        medications = "N/A"
        treatment = "N/A"
        
        try:
            if "[Medications]" in tx_raw and "[Treatment]" in tx_raw:
                parts = tx_raw.split("[Treatment]")
                treatment = parts[1].strip()
                med_part = parts[0].replace("[Medications]", "").strip()
                medications = med_part
            else:
                # 備用切分防呆機制
                medications = tx_raw
        except Exception as parse_err:
            print(f"文字切分失敗，採用防呆輸出: {parse_err}")
            medications = tx_raw

        return diagnosis, reasoning, medications, treatment

    except Exception as e:
        print(f"多階段模型推理過程中發生錯誤: {e}")
        return "Analysis Failed (AI Error)", "N/A", "N/A", "N/A"


import uuid
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID, generate_uid
from pydicom.filewriter import dcmwrite
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_and_upload_encapsulated_pdf(instance_id, diagnosis, reasoning, medications, treatment):
    """將四大核心維度（Diagnosis, Reasoning, Medications, Treatment）精準排版至 PDF 並上傳"""
    try:
        tags_url = f"{ORTHANC_URL}/instances/{instance_id}/tags"
        tags_res = requests.get(tags_url, auth=AUTH, timeout=5)
        if tags_res.status_code != 200:
            return None
        
        tags_json = tags_res.json()
        study_uid = tags_json.get("0020,000d", {}).get("Value")
        patient_id = tags_json.get("0010,0020", {}).get("Value", "UNKNOWN")
        patient_name = tags_json.get("0010,0010", {}).get("Value", "UNKNOWN")
        patient_birth_date = tags_json.get("0010,0030", {}).get("Value", "")
        patient_sex = tags_json.get("0010,0040", {}).get("Value", "")
        accession_number = tags_json.get("0008,0050", {}).get("Value", "")
        
        if not study_uid:
            return None

        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer, pagesize=letter,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
        )
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, leading=22, spaceAfter=15, textColor='#1A365D')
        section_style = ParagraphStyle('SecStyle', parent=styles['Heading2'], fontSize=12, leading=16, spaceBefore=12, spaceAfter=5, textColor='#2C5282')
        body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=11, leading=16, spaceAfter=10)
        
        story = []
        story.append(Paragraph("<b>MedGemma Multi-Stage AI Clinical Report</b>", title_style))
        story.append(Paragraph(f"<b>Patient ID:</b> {patient_id} &nbsp;&nbsp;&nbsp;&nbsp; <b>Name:</b> {patient_name}", body_style))
        story.append(Paragraph(f"<b>Generated Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Spacer(1, 15))
        
        # 1. Diagnosis
        story.append(Paragraph("<b>[1. Diagnosis]</b>", section_style))
        story.append(Paragraph(f"<font color='#C53030'><b>{diagnosis}</b></font>", body_style))
        story.append(Spacer(1, 3))
        
        # 2. Reasoning
        story.append(Paragraph("<b>[2. Reasoning]</b>", section_style))
        formatted_reasoning = reasoning.replace('\n', '<br/>')
        story.append(Paragraph(formatted_reasoning, body_style))
        story.append(Spacer(1, 3))
        
        # 3. Medications
        story.append(Paragraph("<b>[3. Medications]</b>", section_style))
        formatted_meds = medications.replace('\n', '<br/>')
        story.append(Paragraph(formatted_meds, body_style))
        story.append(Spacer(1, 3))

        # 4. Treatment
        story.append(Paragraph("<b>[4. Treatment]</b>", section_style))
        formatted_tx = treatment.replace('\n', '<br/>')
        story.append(Paragraph(formatted_tx, body_style))
        
        doc.build(story)
        pdf_data = pdf_buffer.getvalue()
        pdf_buffer.close()

        # DICOM 封裝
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 222
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.104.1')
        new_sop_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = UID(new_sop_uid)
        file_meta.TransferSyntaxUID = UID('1.2.840.10008.1.2.1')
        file_meta.ImplementationClassUID = UID('1.2.840.10008.2026.1')

        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = patient_birth_date
        ds.PatientSex = patient_sex
        ds.StudyInstanceUID = study_uid
        ds.AccessionNumber = accession_number
        
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPClassUID = UID('1.2.840.10008.5.1.4.1.1.104.1')
        ds.SOPInstanceUID = UID(new_sop_uid)
        
        now_struct = time.localtime()
        ds.StudyDate = time.strftime("%Y%m%d", now_struct)
        ds.StudyTime = time.strftime("%H%M%S", now_struct)
        ds.ContentDate = time.strftime("%Y%m%d", now_struct)
        ds.ContentTime = time.strftime("%H%M%S", now_struct)
        
        ds.Modality = "DOC"
        ds.SeriesNumber = "100"
        ds.InstanceNumber = "1"
        
        ds.InstanceMIMETypeInEncapsulatedDocument = "application/pdf"
        ds.DocumentTitle = f"MedGemma Full AI Report ({diagnosis})"
        ds.EncapsulatedDocument = pdf_data

        fp = BytesIO()
        dcmwrite(fp, ds, write_like_original=False)
        dicom_binary = fp.getvalue()
        fp.close()

        with lock:
            processed_sops[new_sop_uid] = time.time()

        upload_url = f"{ORTHANC_URL}/instances"
        headers = {"Content-Type": "application/dicom"}
        upload_res = requests.post(upload_url, data=dicom_binary, auth=AUTH, headers=headers, timeout=10)
        
        if upload_res.status_code == 200:
            return upload_res.json().get("ID")
        else:
            print(f"上傳 Encapsulated PDF 失敗: {upload_res.status_code}")
            return None
            
    except Exception as e:
        print(f"建立或上傳 Encapsulated PDF 失敗: {e}")
        return None


def process_webhook_task(sop_id):
    global uploaded_counter, successful_new_instance_ids
    
    with lock:
        if sop_id in processed_sops:
            return

    with processing_dict_lock:
        if sop_id not in active_processing_locks:
            active_processing_locks[sop_id] = threading.Lock()
        sop_lock = active_processing_locks[sop_id]

    if not sop_lock.acquire(blocking=False):
        return

    try:
        with lock:
            if sop_id in processed_sops:
                return

        time.sleep(1) 
        instance_uuid = get_instance_uuid_by_sop(sop_id)
        if not instance_uuid:
            return

        # 🧠 呼叫優化後的雙階段 AI 模型
        diagnosis, reasoning, medications, treatment = analyze_image_with_dual_ai(instance_uuid)
        new_instance_id = create_and_upload_encapsulated_pdf(instance_uuid, diagnosis, reasoning, medications, treatment)

        if new_instance_id:
            print(f"✅ [管線完成] 新結構化 PDF 報告上傳成功！內部 ID: {new_instance_id}")
            with counter_lock:
                uploaded_counter += 1
                successful_new_instance_ids.append(new_instance_id)  
            
    except Exception as e:
        print(f"💥 處理任務時發生錯誤: {e}")
    finally:
        sop_lock.release()
        with processing_dict_lock:
            if sop_id in active_processing_locks:
                try:
                    del active_processing_locks[sop_id]
                except KeyError:
                    pass


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    sop_id = data.get("sopInstanceId")
    if sop_id and sop_id != "UnknownSOP":
        threading.Thread(target=process_webhook_task, args=(sop_id,)).start()
    return jsonify({"status": "received"}), 200


def run_flask():
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    print("====== 正在初始化兩階段 AI 影像診斷服務 ======")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("🚀 背景 Flask Webhook 服務已啟動 (Port: 5000)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n 服務已停止")
