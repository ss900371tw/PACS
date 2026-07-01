LINUX: 

cd PACS

docker compose up -d

python app.py

open http://localhost:8042/ui/app/index.html#/

upload dicom

show tkinter

<img width="582" height="276" alt="image" src="https://github.com/user-attachments/assets/0dec6f50-7984-4a5d-8c3d-83185abb8987" />

Windows: 

Open Docker Desktop

pacs-trigger.lua 中的 local url = 'http://172.17.0.1:5000/webhook' 改成 local url = 'http://host.docker.internal:5000/webhook'

cd PACS

docker-compose up -d

python app.py

open http://localhost:8042/ui/app/index.html#/

upload dicom

show tkinter

<img width="582" height="276" alt="image" src="https://github.com/user-attachments/assets/0dec6f50-7984-4a5d-8c3d-83185abb8987" />








pyinstaller --noconfirm --onedir --name "MedGemma_AI_Listener" \
    --exclude-module PyQt5 \
    --exclude-module PyQt6 \
    --exclude-module PyQt5-sip \
    --exclude-module PyQt6-sip \
    --exclude-module sphinx \
    --exclude-module logging \
    app2.py

＃ 刪除 orthanc 

Linux

docker compose down

Windows:

docker-compose down

