import threading
import time
import queue
import requests
from flask import Flask, jsonify, request
import tkinter as tk
from PIL import Image
from io import BytesIO

# ====== 替換 Unsloth：改用 Hugging Face 原生庫 (支援 CPU) ======
print("正在載入 MedGemma AI 模型 (CPU/Transformers)，請稍候...")
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM
# ====== 替換原本的模型載入區塊 ======
import torch
from transformers import AutoProcessor, AutoModelForPreTraining  # 視覺語言模型通常使用 PreTraining 或 CausalLM
from peft import PeftModel

import os
import torch

# ====== 1. Hugging Face 認證與模型設定 ======
# 寫入你的 Token，解決 Unauthenticated 警告並解鎖 Gated 模型下載權限


import os
import torch
from huggingface_hub import hf_hub_download  # 用於自動下載單個 pth 檔案

# ====== 1. Hugging Face 認證與模型設定 ======
os.environ["HF_TOKEN"] = ""

BASE_MODEL = "google/medgemma-1.5-4b-it" 
REPO_ID = "ss900371tw/medgemma-vqa-lora"  # 根據截圖中的正確倉庫名稱
PTH_FILENAME = "lora_round_10.pth"

print("正在載入 MedGemma AI 模型 (純 CPU + 自定義 PTH 模式)，請稍候...")
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model

device = "cpu"
print(f"目前使用的硬體設備為: {device.upper()}")

# ====== 2. 載入處理器 (Processor) ======
print(f"正在從基礎模型 {BASE_MODEL} 載入處理器...")
processor = AutoProcessor.from_pretrained(BASE_MODEL)

# ====== 3. 載入基礎模型 ======
print("正在載入基礎模型權重...")
base_model = PaliGemmaForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# ====== 4. 初始化 LoRA 配置並掛載空的外掛結構 ======
print("正在初始化 LoRA 配置結構...")
# 注意：這裡的 r, lora_alpha, target_modules 必須與你當初微調時的設定完全一致
# 以下為 Unsloth 視覺微調常見的標準預設值
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM"
)

# 讓基礎模型擁有 LoRA 的骨架
model = get_peft_model(base_model, lora_config)

# ====== 5. 自動下載並手動載入 .pth 權重 ======
print(f"正在從 Hugging Face 下載 {PTH_FILENAME}...")
try:
    # 這會自動下載 pth 到快取並回傳本地路徑
    local_pth_path = hf_hub_download(repo_id=REPO_ID, filename=PTH_FILENAME)
    print(f"載入本地權重檔案: {local_pth_path}")
    
    # 用 PyTorch 強制載入至 CPU
    lora_state_dict = torch.load(local_pth_path, map_location=torch.device('cpu'))
    
    # 將權重注入進剛剛建立的空骨架中
    # strict=False 允許忽略一些非必要的優化器參數（如 optimizer 狀態）
    injected_keys = model.load_state_dict(lora_state_dict, strict=False)
    print("LoRA 權重手動融合成功！")
except Exception as e:
    print(f"手動載入 .pth 權重失敗，這通常是微調時的 State Dict 結構與標準 Peft 不同。錯誤訊息: {e}")
    print("警告：模型將以未微調的狀態（Base Model）繼續執行。")

model.eval() # 切換至推理模式
print("AI 模型載入程序完成！")
# ==================================



app = Flask(__name__)

# ====== Orthanc 設定 ======
ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

# ====== 防止無窮迴圈的快取與執行緒鎖機制 ======
processed_sops = {}
CACHE_EXPIRY_SEC = 300  # 5分鐘後過期釋放記憶體
lock = threading.Lock()

active_processing_locks = {}
processing_dict_lock = threading.Lock()

# ====== 上傳計數器、新 Instance ID 快取與定時器變數 ======
uploaded_counter = 0
successful_new_instance_ids = []  
counter_lock = threading.Lock()
popup_timer = None  

# ====== Tkinter 全域 root ======
root = None


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


def trigger_popup_with_debounce():
    """使用防抖機制，延遲觸發 Tkinter 彈窗"""
    global popup_timer
    
    with counter_lock:
        if popup_timer is not None:
            popup_timer.cancel()
        
        # 1.5 秒內都沒有新影像上傳成功，才執行 _execute_popup
        popup_timer = threading.Timer(1.5, _execute_popup)
        popup_timer.start()


def _execute_popup():
    """準備彈窗文字，並安全地將彈窗任務交給 Tkinter 主執行緒"""
    global uploaded_counter, successful_new_instance_ids, popup_timer, root
    
    with counter_lock:
        count = uploaded_counter
        instance_list = list(set(successful_new_instance_ids))  
        
        # 狀態歸零，供下一波上傳使用
        uploaded_counter = 0
        successful_new_instance_ids = []
        popup_timer = None
    
    if count == 0:
        return

    # ====== 動態組合文字與連結 ======
    if count == 1:
        unit = "dicom"
        pronoun = "it"
        new_id = instance_list[0] if instance_list else "UNKNOWN"
        message_text = (
            f"You successfully uploaded 1 {unit}. "
            f"You can open {pronoun} from {ORTHANC_URL}/wsi/app/viewer.html?instance={new_id}"
        )
    else:
        unit = "dicoms"
        pronoun = "them"
        message_text = f"You successfully uploaded {count} {unit}. You can open {pronoun} from:\n"
        for new_id in instance_list:
            message_text += f"{ORTHANC_URL}/wsi/app/viewer.html?instance={new_id}\n"

    # 使用 root.after 叫醒主執行緒來渲染 UI，避免多執行緒死結
    if root:
        root.after(0, lambda: create_toplevel_window(message_text, count, len(instance_list)))


def create_toplevel_window(message_text, count, list_條數):
    """真正由主執行緒渲染的視窗函式"""
    global root
    
    top = tk.Toplevel(root)
    top.title("Upload Success")
    top.attributes("-topmost", True)
    
    # 根據文字多寡動態調整視窗高度
    window_height = 140 + (max(0, list_條_數 - 1) * 20) if count > 1 else 130
    top.geometry(f"580x{window_height}")  
    top.resizable(True, True)  
    
    top.update_idletasks()
    x = (top.winfo_screenwidth() - top.winfo_reqwidth()) // 2
    y = (top.winfo_screenheight() - top.winfo_reqheight()) // 2
    top.geometry(f"+{x}+{y}")

    # 使用 Text 元件方便複製網址
    text_area = tk.Text(top, wrap="word", font=("Arial", 10), bd=0, bg=top.cget("bg"))
    text_area.insert("1.0", message_text)
    text_area.configure(state="disabled")  
    text_area.pack(pady=15, padx=20, fill="both", expand=True)

    # 點擊 OK 只銷毀 Toplevel 視窗，不要摧毀 root 主核心
    btn = tk.Button(top, text="OK", width=10, command=top.destroy)
    btn.pack(pady=(0, 12))


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
    """從 Orthanc 獲取該實例的預覽圖，並使用原生 Transformers 進行 CPU 推理預測"""
    try:
        preview_url = f"{ORTHANC_URL}/instances/{instance_id}/preview"
        response = requests.get(preview_url, auth=AUTH, timeout=10)
        
        if response.status_code != 200:
            print(f"無法從 Orthanc 取得影像預覽: {instance_id}")
            return "Analysis Failed (Preview Fetch Error)"

        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 設定與微調時一致的 Prompt 格式
        instruction = "Identify the most likely disease in this image."
        prompt = f"<image>\nUser: {instruction}\nAssistant: "

        # 使用 processor 處理影像與文字，並確保張量送至正確的設備 (CPU/GPU)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        # 關閉梯度計算進行推理
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=20,
                use_cache=True
            )
        
        # 僅解碼新生成的 Token
        input_len = inputs["input_ids"].shape[1]
        diagnosis = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
        
        print(f"[AI 實時診斷結果 - ID: {instance_id}]: {diagnosis}")
        return diagnosis

    except Exception as e:
        print(f"AI 推理過程中發生錯誤: {e}")
        return "Analysis Failed (AI Error)"


def update_instance_tag_keep_same_id(instance_id, comment_text):
    """修改標籤並強制覆蓋原 ID，成功時回傳「新生成的 Orthanc 內部 Instance ID」"""
    try:
        tags_url = f"{ORTHANC_URL}/instances/{instance_id}/tags"
        tags_res = requests.get(tags_url, auth=AUTH, timeout=5)
        if tags_res.status_code != 200:
            return None
        
        tags_json = tags_res.json()
        sop_instance_uid = tags_json.get("0008,0018", {}).get("Value")
        study_uid = tags_json.get("0020,000d", {}).get("Value")
        series_uid = tags_json.get("0020,000e", {}).get("Value")

        modify_url = f"{ORTHANC_URL}/instances/{instance_id}/modify"
        payload = {
            "Replace": {
                "PatientComments": str(comment_text), # 寫入 AI 診斷結果
                "SOPInstanceUID": sop_instance_uid,     
                "StudyInstanceUID": study_uid,          
                "SeriesInstanceUID": series_uid         
            },
            "Force": True
        }
        
        modified_res = requests.post(modify_url, json=payload, auth=AUTH, timeout=10)
        if modified_res.status_code != 200:
            return None
            
        dicom_binary = modified_res.content

        if sop_instance_uid:
            with lock:
                processed_sops[sop_instance_uid] = time.time()

        # 刪除舊的，上傳修改後的
        delete_url = f"{ORTHANC_URL}/instances/{instance_id}"
        requests.delete(delete_url, auth=AUTH, timeout=5)
        
        upload_url = f"{ORTHANC_URL}/instances"
        headers = {"Content-Type": "application/dicom"}
        upload_res = requests.post(upload_url, data=dicom_binary, auth=AUTH, headers=headers, timeout=10)
        
        if upload_res.status_code == 200:
            return upload_res.json().get("ID")
        else:
            return None
    except Exception as e:
        print(f"更新 DICOM Tag 失敗: {e}")
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

        instance_uuid = get_instance_uuid_by_sop(sop_id)
        if not instance_uuid:
            return

        diagnosis_result = analyze_image_with_ai(instance_uuid)

        new_instance_id = update_instance_tag_keep_same_id(instance_uuid, comment_text=diagnosis_result)
        
        if new_instance_id:
            with counter_lock:
                uploaded_counter += 1
                successful_new_instance_ids.append(new_instance_id)  
            trigger_popup_with_debounce()
            
    finally:
        sop_lock.release()
        with processing_dict_lock:
            if sop_id in active_processing_locks:
                del active_processing_locks[sop_id]


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    sop_id = data.get("sopInstanceId")
    if sop_id and sop_id != "UnknownSOP":
        threading.Thread(target=process_webhook_task, args=(sop_id,)).start()
    return jsonify({"status": "received"}), 200


def run_flask():
    """在背景子執行緒運行 Flask 服務"""
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    root = tk.Tk()
    root.withdraw()  
    
    root.mainloop()
