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
# 對齊微調腳本，改用 AutoModelForImageTextToText 確保架構與 Chat Template 完美相容
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel  # 引入 PEFT 用來掛載 LoRA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎬 目前偵測到的運算裝置: {device.upper()}")


# 🌟 安全管理：優先從環境變數讀取 Token
HF_TOKEN = os.getenv("HF_TOKEN", "")

# 基底模型 (PaliGemma / Gemma 3 骨幹的 MedGemma 1.5 4b 版本)
base_model_id = "google/medgemma-1.5-4b-it"
# 您的 LoRA 微調權重儲存庫
lora_model_id = "ss900371tw/medgemma-vqa-lora"

print("正在載入影像處理器...")
processor = AutoProcessor.from_pretrained(base_model_id, token=HF_TOKEN)

print(f"正在載入 Google MedGemma 基底模型 (使用 {device.upper()})...")
# 如果是 GPU 使用 bfloat16 提升速度並節省顯存，CPU 則維持 float32
current_dtype = torch.bfloat16 if device == "cuda" else torch.float32

base_model_obj = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=current_dtype,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
).to(device)

print("正在掛載 MedGemma LoRA 微調權重...")
model = PeftModel.from_pretrained(
    base_model_obj, 
    lora_model_id, 
    token=HF_TOKEN
)

print("正在進行權重融合 (Merge & Unload)...")
model = model.merge_and_unload()
model.eval()                       # 將模型切換至評估模式
print("🎉 AI 模型與 LoRA 權重載入與融合成功！")


# ====== Flask 實例初始化 ======
app = Flask(__name__)

# ====== Orthanc 設定 ======
ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

# ====== 防止無窮迴圈的快取與執行緒鎖機制 ======
processed_sops = {}
CACHE_EXPIRY_SEC = 300  # 5分鐘後過期釋放記憶體
lock = threading.Lock()

# 新增一個專門用來鎖定「特定 SOP 處理流程」的字典，避免同一個檔案的併發 Webhook 重複執行
active_processing_locks = {}
processing_dict_lock = threading.Lock()

# ====== 上傳計數器、新 Instance ID 快取與定時器變數 ======
uploaded_counter = 0
successful_new_instance_ids = []  # 用來儲存新產生的 Orthanc 內部新 instance_id
counter_lock = threading.Lock()



def clean_expired_cache():
    """定期清理過期的快取防止記憶體無限增長"""
    while True:
        time.sleep(60)
        now = time.time()
        with lock:
            expired = [sop for sop, ts in processed_sops.items() if now - ts > CACHE_EXPIRY_SEC]
            for sop in expired:
                del processed_sops[sop]

# 啟動背景清理執行緒
threading.Thread(target=clean_expired_cache, daemon=True).start()




def get_instance_uuid_by_sop(sop_instance_uid):
    """向 Orthanc 查詢 SOPInstanceUID 對應的內部 UUID"""
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


def analyze_image_with_ai(instance_id):
    """從 Orthanc 獲取該實例的預覽圖，並套用標準 Chat Template 進行推理預測"""
    try:
        preview_url = f"{ORTHANC_URL}/instances/{instance_id}/preview"
        response = requests.get(preview_url, auth=AUTH, timeout=10)
        
        if response.status_code != 200:
            print(f"無法從 Orthanc 取得影像預覽: {instance_id}")
            return "Analysis Failed (Preview Fetch Error)"

        # 讀取影像並確保轉換為 RGB 格式
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 🌟 修正重點：與 Finetuning (task.py) 結構完全對齊的 Chat Template 格式
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "Identify the most likely disease in this image."
                    },
                    {
                        "type": "image"
                    }
                ]
            }
        ]
        
        # 將格式轉換為 Gemma 3 / MedGemma 標準的 Prompt 文本
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 🌟 修正重點：Gemma3 / MedGemma 處理器要求 nested list (即 [[img]])
        # 同時傳入套用 Template 後的文字與格式化影像
        inputs = processor(
            text=[prompt_text], 
            images=[[image]], 
            return_tensors="pt"
        ).to(device)

        # 進行推理
        with torch.no_grad():
            output_ids = model.generate(
            **inputs, 
            max_new_tokens=20,
            # 🌟 明確指定 pad_token_id，如果處理器沒有定義，則安全地對齊 eos_token_id
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
            )

        
        # 解碼預測結果 (剪切掉輸入 Prompt 的長度，僅保留輸出的 Answer)
        input_len = inputs["input_ids"].shape[-1]
        diagnosis = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        
        print(f"[AI 實時診斷結果 - ID: {instance_id}]: {diagnosis}")
        return diagnosis

    except Exception as e:
        print(f"AI 推理過程中發生錯誤: {e}")
        return "Analysis Failed (AI Error)"


import uuid
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID, generate_uid

# 替換掉原本的 update_instance_tag_keep_same_id 函數
from pydicom.filewriter import dcmwrite  # 🌟 必須引入 dcmwrite 函數

# 修正後的 create_and_upload_dicom_sr 函數
def create_and_upload_dicom_sr(instance_id, report_text):
    """
    根據原 CXR 影像的中繼資料，建構一個標準的 DICOM SR (Structured Report) 檔案，
    並將 MedGemma 的長報告寫入，最後獨立上傳至 Orthanc。
    """
    try:
        # 1. 從 Orthanc 取得原影像的 Tags 與中繼資料
        tags_url = f"{ORTHANC_URL}/instances/{instance_id}/tags"
        tags_res = requests.get(tags_url, auth=AUTH, timeout=5)
        if tags_res.status_code != 200:
            print(f"無法取得原影像 Tags: {instance_id}")
            return None
        
        tags_json = tags_res.json()
        
        # 擷取原影像關鍵 UID 與病人資訊
        study_uid = tags_json.get("0020,000d", {}).get("Value")
        patient_id = tags_json.get("0010,0020", {}).get("Value", "UNKNOWN")
        patient_name = tags_json.get("0010,0010", {}).get("Value", "UNKNOWN")
        patient_birth_date = tags_json.get("0010,0030", {}).get("Value", "")
        patient_sex = tags_json.get("0010,0040", {}).get("Value", "")
        accession_number = tags_json.get("0008,0050", {}).get("Value", "")
        
        if not study_uid:
            print("原影像缺少 StudyInstanceUID，無法建立 SR 關聯。")
            return None

        # 2. 初始化 DICOM Dataset
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 222
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        # 設定為 Basic Text SR Storage
        file_meta.MediaStorageSOPClassUID = UID('1.2.840.10008.5.1.4.1.1.88.11')
        
        # 為這個 SR 報告生成全新的獨立 SOPInstanceUID
        new_sop_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = UID(new_sop_uid)
        file_meta.TransferSyntaxUID = UID('1.2.840.10008.1.2.1') # Explicit VR Little Endian
        file_meta.ImplementationClassUID = UID('1.2.840.10008.2026.1')

        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # 3. 複製病人與檢查關聯資訊 (必須與原圖一致)
        ds.PatientName = patient_name
        ds.PatientID = patient_id
        ds.PatientBirthDate = patient_birth_date
        ds.PatientSex = patient_sex
        ds.StudyInstanceUID = study_uid
        ds.AccessionNumber = accession_number
        
        # 4. 設定 SR 專屬的 Series 與 Instance 識別碼
        ds.SeriesInstanceUID = generate_uid()  # 報告必須屬於獨立的新 Series
        ds.SOPClassUID = UID('1.2.840.10008.5.1.4.1.1.88.11')  # Basic Text SR
        ds.SOPInstanceUID = UID(new_sop_uid)
        ds.DataType = "TEXT"
        
        # 設定報告時間
        now_struct = time.localtime()
        ds.StudyDate = time.strftime("%Y%m%d", now_struct)
        ds.StudyTime = time.strftime("%H%M%S", now_struct)
        ds.ContentDate = time.strftime("%Y%m%d", now_struct)
        ds.ContentTime = time.strftime("%H%M%S", now_struct)
        
        ds.Modality = "SR"                  # 儀器類型改為 SR
        ds.SeriesNumber = "99"              # 通常給予一個較大的 Series 號碼與影像區隔
        ds.InstanceNumber = "1"
        ds.ValueType = "TEXT"               # 說明內容物為文字

        # 5. 填入 MedGemma 的長報告文字
        ds.TextValue = str(report_text)     # 寫入長報告
        
        # 概念名稱設定（Findings）
        concept_name = Dataset()
        concept_name.CodeValue = "121070"
        concept_name.CodingSchemeDesignator = "DCM"
        concept_name.CodeMeaning = "Findings"
        ds.ConceptNameCodeSequence = [concept_name]

        # 6. 🌟 修正重點：使用 dcmwrite 函數將其寫入 BytesIO
        fp = BytesIO()
        dcmwrite(fp, ds, write_like_original=False)
        dicom_binary = fp.getvalue()
        fp.close()

        # 7. 在上傳前，預先將該「新生成的 SR SOPInstanceUID」寫入快取，避免 Webhook 造成重複無限迴圈
        with lock:
            processed_sops[new_sop_uid] = time.time()

        # 8. 直接上傳至 Orthanc (不刪除原圖，僅新增 SR)
        upload_url = f"{ORTHANC_URL}/instances"
        headers = {"Content-Type": "application/dicom"}
        upload_res = requests.post(upload_url, data=dicom_binary, auth=AUTH, headers=headers, timeout=10)
        
        if upload_res.status_code == 200:
            return upload_res.json().get("ID")
        else:
            print(f"上傳 SR 失敗，Orthanc 回傳狀態碼: {upload_res.status_code}")
            return None
            
    except Exception as e:
        print(f"建立或上傳 DICOM SR 失敗: {e}")
        return None


def process_webhook_task(sop_id):
    global uploaded_counter, successful_new_instance_ids
    print(f"📥 [Webhook 偵測] 收到來自 Orthanc 的 SOP ID: {sop_id}")
    
    # 1. 檢查是否已經存在於已處理快取中
    with lock:
        if sop_id in processed_sops:
            print(f"⚠️ [跳過] SOP ID {sop_id} 已在快取中，防止重複處理。")
            return

    # 2. 獲取或建立針對該單一 SOP 檔案的執行緒鎖
    with processing_dict_lock:
        if sop_id not in active_processing_locks:
            active_processing_locks[sop_id] = threading.Lock()
        sop_lock = active_processing_locks[sop_id]

    # 使用該檔案專屬的鎖進行處理
    if not sop_lock.acquire(blocking=False):
        print(f"⚠️ [跳過] SOP ID {sop_id} 目前正由另一個執行緒處理中。")
        return

    try:
        # 再次雙重檢查
        with lock:
            if sop_id in processed_sops:
                return

        print(f"🔍 [步驟 1] 正在向 Orthanc 查詢 SOP ID 對應的內部 UUID...")
        time.sleep(1) # 💡 加上 1 秒緩衝，避免 Orthanc 還沒寫入完畢
        instance_uuid = get_instance_uuid_by_sop(sop_id)
        
        if not instance_uuid:
            print(f"❌ [失敗] Orthanc 查無此 SOP ID 的 UUID，流程中斷。")
            return
        print(f"   -> 成功取得 Orthanc 內部 UUID: {instance_uuid}")

        # 呼叫 AI 進行即時推理
        print(f"🧠 [步驟 2] 開始呼叫 MedGemma AI 進行 CPU 推理...")
        diagnosis_result = analyze_image_with_ai(instance_uuid)
        print(f"   -> AI 診斷完成，結果為: {diagnosis_result}")

        # 修改 Tag 並重新上傳
        print(f"💾 [步驟 3] 正在將 MedGemma 報告轉換為 DICOM SR 並上傳至 Orthanc...")
        new_instance_id = create_and_upload_dicom_sr(instance_uuid, report_text=diagnosis_result)
 
        if new_instance_id:
            print(f"✅ [成功] 影像已重新覆蓋上傳！新 Instance ID: {new_instance_id}")
            with counter_lock:
                uploaded_counter += 1
                successful_new_instance_ids.append(new_instance_id)  
        
        else:
            print(f"❌ [失敗] 修改 Tag 或重新上傳失敗，未觸發彈窗。")
            
    except Exception as e:
        print(f"💥 [異常] 處理任務時發生錯誤: {e}")
    finally:
        # 釋放鎖定
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
        # 丟到背景執行緒，不卡住 Webhook 回應
        threading.Thread(target=process_webhook_task, args=(sop_id,)).start()
    return jsonify({"status": "received"}), 200


def run_flask():
    """在背景子執行緒運行 Flask 服務"""
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)

# =====================================================================
# 主程式進入點 (確保 Tkinter 事件循環不休眠)
# =====================================================================
if __name__ == "__main__":
    print("====== 正在初始化服務 ======")
    
    # 1. 啟動背景 Flask 執行緒
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("🚀 背景 Flask Webhook 服務已啟動 (Port: 5000)")

    # 2. 主執行緒常駐給 Tkinter 核心使用
    root = tk.Tk()
    root.title("MedGemma AI Listener")
    
    # 將其縮小並放到角落，確保 active 狀態且不干擾主要操作
    root.geometry("250x60+0+0") 
    
    # 加上簡單的狀態標籤，讓介面知道它還活著
    status_label = tk.Label(root, text="AI 服務運行中...\n請勿關閉此視窗", font=("Arial", 10))
    status_label.pack(pady=10)
    
    print("🎉 Tkinter 主事件循環已就緒，等待 DICOM 上傳 Webhook...")
    
    # 開始進入 Tkinter 主循環
    root.mainloop()
