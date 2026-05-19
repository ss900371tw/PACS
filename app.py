import threading
import time
import re
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

# ====== Orthanc 設定 ======
ORTHANC_URL = "http://localhost:8042"
AUTH = ("orthanc", "orthanc")

# ====== 防止無窮迴圈的快取機制 ======
# 快取格式: { "SOPInstanceUID": 寫入時的時間戳記 }
processed_sops = {}
CACHE_EXPIRY_SEC = 300  # 5分鐘後過期釋放記憶體
lock = threading.Lock()

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
        pass
    return None


def update_instance_tag_keep_same_id(instance_id, comment_text="No pneumothorax"):
    """
    修改標籤並強制覆蓋原 ID
    """
    try:
        # 1. 獲取原始影像的所有原生 UID 資訊
        tags_url = f"{ORTHANC_URL}/instances/{instance_id}/tags"
        tags_res = requests.get(tags_url, auth=AUTH, timeout=5)
        if tags_res.status_code != 200:
            return False
        
        tags_json = tags_res.json()
        sop_instance_uid = tags_json.get("0008,0018", {}).get("Value")
        study_uid = tags_json.get("0020,000d", {}).get("Value")
        series_uid = tags_json.get("0020,000e", {}).get("Value")

        # 2. 呼叫 modify 生成帶有新標籤的 DICOM 二進位檔案
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
            return False
            
        dicom_binary = modified_res.content

        # 3. 終極魔法：先刪除舊的
        delete_url = f"{ORTHANC_URL}/instances/{instance_id}"
        requests.delete(delete_url, auth=AUTH, timeout=5)
        upload_url = f"{ORTHANC_URL}/instances"
        headers = {"Content-Type": "application/dicom"}
        upload_res = requests.post(upload_url, data=dicom_binary, auth=AUTH, headers=headers, timeout=10)
        
        if upload_res.status_code == 200:
            return True
        else:
            return False

    except Exception as e:
        return False


def process_webhook_task(sop_id):
    with lock:
        if sop_id in processed_sops:
            return

    with lock:
        processed_sops[sop_id] = time.time()

    instance_uuid = get_instance_uuid_by_sop(sop_id)
    if not instance_uuid:
        with lock:
            if sop_id in processed_sops: del processed_sops[sop_id] # 失敗則移除快取
        return

    success = update_instance_tag_keep_same_id(instance_uuid, comment_text="No pneumothorax")
    if not success:
        with lock:
            if sop_id in processed_sops: del processed_sops[sop_id]


@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(silent=True) or {}
    sop_id = data.get("sopInstanceId")
    if sop_id and sop_id != "UnknownSOP":
        threading.Thread(target=process_webhook_task, args=(sop_id,)).start()
    return jsonify({"status": "received"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
