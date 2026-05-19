import threading
import time
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

# ====== Tkinter 全域主視窗物件與文字元件 ======
root = None
text_area = None


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
    """準備彈窗文字，並安全地呼叫 root 更新介面"""
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

    # 安全地讓主執行緒去控制 root 彈出
    if root:
        root.after(0, lambda: show_root_window(message_text, count, len(instance_list)))


def show_root_window(message_text, count, list_条数):
    """將隱藏的 root 視窗更新文字、調整大小並移到螢幕中央顯示"""
    global root, text_area
    
    # 1. 更新文字內容
    text_area.configure(state="normal")
    text_area.delete("1.0", tk.END)
    text_area.insert("1.0", message_text)
    text_area.configure(state="disabled")
    
    # 2. 根據文字多寡動態調整視窗高度
    window_height = 140 + (max(0, list_条数 - 1) * 20) if count > 1 else 130
    
    # 3. 計算螢幕中央座標
    root.update_idletasks()
    x = (root.winfo_screenwidth() - 580) // 2
    y = (root.winfo_screenheight() - window_height) // 2
    
    # 4. 重新設定大小並移回螢幕中央 (取消一開始的隱形狀態)
    root.geometry(f"580x{window_height}+{x}+{y}")
    root.deiconify()  # 確保視窗不是最小化狀態
    root.attributes("-topmost", True)  # 強制置頂
    root.focus_force()  # 強制取得焦點


def hide_root_window():
    """使用者點擊 OK 或關閉視窗時，只把 root 移到螢幕外隱藏，而不摧毀它"""
    global root
    # 把視窗縮小並丟到外太空（螢幕外），保留 mainloop 活體
    root.geometry("0x0+9999+9999")


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

    # 2. 主執行緒直接初始化 root 視窗與佈局
    root = tk.Tk()
    root.title("Upload Success")
    
    # 關鍵：一開始先把它藏到螢幕外 (+9999)，等 Webhook 觸發時才拉回中央
    root.geometry("0x0+9999+9999")
    root.resizable(True, True)

    # 預先在 root 建立好 Text 元件（用來裝網址）
    text_area = tk.Text(root, wrap="word", font=("Arial", 10), bd=0, bg=root.cget("bg"))
    text_area.pack(pady=15, padx=20, fill="both", expand=True)

    # 預先建立好 OK 按鈕，點擊時呼叫 hide_root_window 移到螢幕外
    btn = tk.Button(root, text="OK", width=10, command=hide_root_window)
    btn.pack(pady=(0, 12))

    # 攔截右上角的 "X" 關閉按鈕，讓它同樣執行隱藏，而不是摧毀主視窗
    root.protocol("WM_DELETE_WINDOW", hide_root_window)

    # 進入 Tkinter 主循環
    root.mainloop()
