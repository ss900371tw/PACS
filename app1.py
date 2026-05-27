import threading
import time
import queue
import os
from io import BytesIO
import tkinter as tk

# ====== Web 相關庫 ======
import requests
from flask import Flask, jsonify, request

# ====== AI / Image 相關庫 ======
import torch
from PIL import Image

# ====== Unsloth / Hugging Face 模型載入 ======
print("正在載入 MedGemma AI 模型，請稍候...")
from unsloth import FastVisionModel

# 🌟 安全管理：優先從環境變數讀取 Token
HF_TOKEN = os.getenv("HF_TOKEN", "")

# 這裡直接載入你合併微調後的完整全參數模型
model_id = "ss900371tw/medgemma-vqa-lora"

# Unsloth 載入多模態模型時，會同時回傳 model 與 processor (包含 tokenizer 與 image_processor)
model, processor = FastVisionModel.from_pretrained(
    model_name = model_id, 
    load_in_4bit = True,
    token = HF_TOKEN
)
FastVisionModel.for_inference(model)
print("🎉 AI 模型載入成功！")

app = Flask(__name__)

# ====== Orthanc 設定 ======
ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

# ====== 防止無窮迴圈的快取與執行緒鎖機制 ======
processed_sops = {}
CACHE_EXPIRY_SEC = 300  # 5分鐘後過期釋放記憶體
lock = threading.Lock()

# ====== 🚀 關鍵修正：引入任務隊列 (Task Queue) ======
task_queue = queue.Queue()

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
        
        popup_timer = threading.Timer(1.5, _execute_popup)
        popup_timer.start()


def _execute_popup():
    """準備彈窗文字，並安全地將彈窗任務交給 Tkinter 主執行緒"""
    global uploaded_counter, successful_new_instance_ids, popup_timer, root
    
    with counter_lock:
        count = uploaded_counter
        instance_list = list(set(successful_new_instance_ids))  
        
        uploaded_counter = 0
        successful_new_instance_ids = []
        popup_timer = None
    
    if count == 0:
        return

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

    if root:
        root.after(0, lambda: create_toplevel_window(message_text, count, len(instance_list)))


def create_toplevel_window(message_text, count, list_count):
    """真正由主執行緒渲染的視窗函式"""
    global root
    
    top = tk.Toplevel(root)
    top.title("Upload Success")
    top.attributes("-topmost", True)
    
    window_height = 140 + (max(0, list_count - 1) * 20) if count > 1 else 130
    top.geometry(f"580x{window_height}")  
    top.resizable(True, True)  
    
    top.update_idletasks()
    x = (top.winfo_screenwidth() - top.winfo_reqwidth()) // 2
    y = (top.winfo_screenheight() - top.winfo_reqheight()) // 2
    top.geometry(f"+{x}+{y}")

    text_area = tk.Text(top, wrap="word", font=("Arial", 10), bd=0, bg=top.cget("bg"))
    text_area.insert("1.0", message_text)
    text_area.configure(state="disabled")  
    text_area.pack(pady=15, padx=20, fill="both", expand=True)

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
    """從 Orthanc 獲取該實例的預覽圖，並使用 MedGemma 進行分析預測"""
    try:
        preview_url = f"{ORTHANC_URL}/instances/{instance_id}/preview"
        response = requests.get(preview_url, auth=AUTH, timeout=10)
        
        if response.status_code != 200:
            print(f"無法從 Orthanc 取得影像預覽: {instance_id}")
            return "Analysis Failed (Preview Fetch Error)"

        # 確保影像轉換為 RGB
        image = Image.open(BytesIO(response.content)).convert("RGB")

        instruction = "Identify the most likely disease in this image."
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=image, text=input_text, return_tensors="pt").to("cuda")

        # 進行推論
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=20)
        
        # 解碼預測結果
        input_len = inputs["input_ids"].shape[-1]
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
                "PatientComments": str(comment_text), 
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

        # 在刪除與重傳前，先把舊的 SOP 記錄到快取防止無限迴圈
        if sop_instance_uid:
            with lock:
                processed_sops[sop_instance_uid] = time.time()

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


# ====== 🚀 關鍵修正：獨立的 AI 消費者執行緒 ======
def ai_worker_loop():
    """單一執行緒：依序從隊列中取出 SOP ID 進行 AI 推論，避免 GPU 競爭"""
    global uploaded_counter, successful_new_instance_ids
    print("🚀 AI 推論後台排隊執行緒已啟動...")
    
    while True:
        sop_id = task_queue.get()  # 如果隊列是空的，會在這裡乖乖等
        try:
            # 檢查是否已經被處理過 (二次防線)
            with lock:
                if sop_id in processed_sops:
                    continue

            instance_uuid = get_instance_uuid_by_sop(sop_id)
            if not instance_uuid:
                continue

            # 呼叫 AI（現在只有這一個執行緒在呼叫，絕對不會漏圖了！）
            diagnosis_result = analyze_image_with_ai(instance_uuid)

            new_instance_id = update_instance_tag_keep_same_id(instance_uuid, comment_text=diagnosis_result)
            
            if new_instance_id:
                with counter_lock:
                    uploaded_counter += 1
                    successful_new_instance_ids.append(new_instance_id)  
                trigger_popup_with_debounce()
                
        except Exception as e:
            print(f"處理任務時發生非預期錯誤: {e}")
        finally:
            task_queue.task_done()


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    sop_id = data.get("sopInstanceId")
    
    if sop_id and sop_id != "UnknownSOP":
        # 檢查快取，如果不是剛被修改標籤重傳的，就塞進排隊隊列中
        with lock:
            is_processed = sop_id in processed_sops
            
        if not is_processed:
            print(f"📥 收到新 DICOM Webhook，加入排隊隊列: {sop_id}")
            task_queue.put(sop_id)
            
    return jsonify({"status": "received"}), 200


def run_flask():
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)


if __name__ == "__main__":
    # 1. 啟動 Flask 接收端 (多執行緒)
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # 2. 🚀 啟動唯一的 AI 推論消費端 (單執行緒排隊)
    ai_thread = threading.Thread(target=ai_worker_loop, daemon=True)
    ai_thread.start()

    # 3. 啟動 Tkinter GUI 主執行緒
    root = tk.Tk()
    root.withdraw()  
    root.mainloop()
    flask_thread.start()

    root = tk.Tk()
    root.withdraw()  
    root.mainloop()
