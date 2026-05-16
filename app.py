import tkinter as tk
from tkinter import messagebox
from flask import Flask, jsonify
import threading

app = Flask(__name__)

def show_popup():
    # 建立一個隱藏的 Tkinter 主視窗
    root = tk.Tk()
    root.withdraw()
    # 確保視窗跳在最上層
    root.attributes('-topmost', True)
    # 彈出提示訊息
    messagebox.showinfo("Orthanc 通知", "You uploaded a file")
    root.destroy()

@app.route('/webhook', methods=['POST'])
def webhook():
    # 使用線程 (Threading) 異步觸發彈窗，避免卡住 HTTP 請求
    threading.Thread(target=show_popup).start()
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    print("Windows 監聽伺服器已啟動，等待 Orthanc 上傳檔案...")
    # 監聽在本地 5000 端口
    app.run(host='0.0.0.0', port=5000)