import threading
import time
import tkinter as tk
from flask import Flask, jsonify, request
import requests  # 記得確保有安裝 requests 模組

app = Flask(__name__)

root = tk.Tk()
root.withdraw()

# ====== Orthanc 設定 ======
ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")  # 請確保帳密正確

# ====== debounce 狀態 ======
lock = threading.Lock()
pending_SOPs = set()  # 用 set 儲存不重複的 SOPInstanceUID
last_event_time = 0
debounce_ms = 800  # 0.8 秒內合併


def show_custom_popup(title, msg):
    """自訂可以反白複製文字的彈出視窗"""
    popup = tk.Toplevel(root)
    popup.title(title)
    popup.geometry("500x250")  # 稍微加寬加高以容納 preview 網址
    popup.attributes("-topmost", True)

    frame = tk.Frame(popup, padx=15, pady=15)
    frame.pack(fill=tk.BOTH, expand=True)

    bg_color = popup.cget("bg")
    text_area = tk.Text(
        frame,
        wrap=tk.WORD,
        bg=bg_color,
        relief="flat",
        font=("Microsoft JhengHei", 10),
    )
    text_area.insert(tk.END, msg)
    text_area.config(state=tk.DISABLED)
    text_area.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    btn_ok = tk.Button(
        frame, text="OK", width=10, command=popup.destroy, relief="groove"
    )
    btn_ok.pack(side=tk.BOTTOM, pady=(5, 0))

    # 視窗居中
    popup.update_idletasks()
    width = popup.winfo_width()
    height = popup.winfo_height()
    x = (popup.winfo_screenwidth() // 2) - (width // 2)
    y = (popup.winfo_screenheight() // 2) - (height // 2)
    popup.geometry(f"{width}x{height}+{x}+{y}")


def get_preview_url(sop_instance_uid):
    """向 Orthanc 查詢 SOPInstanceUID 對應的預覽網址"""
    try:
        lookup_url = f"{ORTHANC_URL}/tools/lookup"
        # 注意：Orthanc lookup 接收的是純文字字串作為 data
        response = requests.post(
            lookup_url, data=sop_instance_uid, auth=AUTH, timeout=5
        )

        if response.status_code == 200:
            res_json = response.json()
            if res_json and len(res_json) > 0:
                instance_uuid = res_json[0]["ID"]
                return f"{ORTHANC_URL}/instances/{instance_uuid}/preview"
    except Exception as e:
        print(f"查詢 Orthanc 失敗 ({sop_instance_uid}): {e}")

    return f"無法取得該影像的預覽連結 (UID: {sop_instance_uid})"


def flush_popup():
    global pending_SOPs

    with lock:
        sops_list = list(pending_SOPs)
        pending_SOPs.clear()

    count = len(sops_list)
    if count == 0:
        return

    # 批次向 Orthanc 換取 preview_url
    preview_urls = []
    for sop_id in sops_list:
        url = get_preview_url(sop_id)
        preview_urls.append(url)

    # 格式化顯示文字
    if count == 1:
        msg = f"You uploaded a file.\nYou can preview it from:\n{preview_urls[0]}"
    else:
        urls_str = "\n".join(preview_urls)
        msg = f"You uploaded {count} unique instances.\nYou can preview them from:\n{urls_str}"

    # 呼叫 Tkinter GUI 執行緒顯示彈窗
    root.after(0, lambda: show_custom_popup("Orthanc 通知", msg))


def debounce_worker():
    """背景檢查是否要觸發 popup"""
    global last_event_time

    while True:
        time.sleep(0.2)

        with lock:
            if len(pending_SOPs) > 0:
                idle_time = time.time() - last_event_time

                if idle_time > debounce_ms / 1000:
                    root.after(0, flush_popup)


@app.route("/webhook", methods=["POST"])
def webhook():
    global last_event_time

    data = request.get_json(silent=True) or {}
    # 對應 Lua 發送的 payload 欄位名稱
    sop_id = data.get("sopInstanceId")

    if sop_id and sop_id != "UnknownSOP":
        with lock:
            pending_SOPs.add(sop_id)
            last_event_time = time.time()

    return jsonify({"status": "success"}), 200


def run_flask():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    # Flask thread
    threading.Thread(target=run_flask, daemon=True).start()

    # debounce thread
    threading.Thread(target=debounce_worker, daemon=True).start()

    root.mainloop()
