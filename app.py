import threading
import time
import queue
import requests
from flask import Flask, jsonify, request
import tkinter as tk

app = Flask(__name__)

# ====== Orthanc 設定 ======
ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

# ====== 防止無窮迴圈的快取機制 ======
processed_sops = {}
CACHE_EXPIRY_SEC = 300  # 5分鐘後過期釋放記憶體
lock = threading.Lock()

# ====== 上傳計數器、新 Instance ID 快取與定時器變數 ======
uploaded_counter = 0
successful_new_instance_ids = []  # 用來儲存新產生的 Orthanc 內部新 instance_id
counter_lock = threading.Lock()
popup_timer = None  # 用來記錄目前的定時器物件

# ====== Tkinter 專用執行緒安全佇列與全域 root ======
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

    # 【核心修正】使用 root.after 叫醒主執行緒來渲染 UI，避免多執行緒死結
    if root:
        root.after(0, lambda: create_toplevel_window(message_text, count, len(instance_list)))


def create_toplevel_window(message_text, count, list_條數):
    """真正由主執行緒渲染的視窗函式"""
    global root
    
    top = tk.Toplevel(root)
    top.title("Upload Success")
    top.attributes("-topmost", True)
    
    # 根據文字多寡動態調整視窗高度
    window_height = 140 + (max(0, list_條數 - 1) * 20) if count > 1 else 130
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
        pass
    return None


def update_instance_tag_keep_same_id(instance_id, comment_text="No pneumothorax"):
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
        return None


def process_webhook_task(sop_id):
    global uploaded_counter, successful_new_instance_ids
    
    with lock:
        if sop_id in processed_sops:
            return
    with lock:
        processed_sops[sop_id] = time.time()

    instance_uuid = get_instance_uuid_by_sop(sop_id)
    if not instance_uuid:
        with lock:
            if sop_id in processed_sops: del processed_sops[sop_id]
        return

    new_instance_id = update_instance_tag_keep_same_id(instance_uuid, comment_text="No pneumothorax")
    
    if new_instance_id:
        with counter_lock:
            uploaded_counter += 1
            successful_new_instance_ids.append(new_instance_id)  
        trigger_popup_with_debounce()
    else:
        with lock:
            if sop_id in processed_sops: del processed_sops[sop_id]


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
    # 1. 啟動背景 Flask 執行緒
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # 2. 主執行緒常駐給 Tkinter 核心使用
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗，我們只用 Toplevel 來跳通知
    
    # 開始進入 Tkinter 主循環，永不退出，直到手動關閉命令列
    root.mainloop()
