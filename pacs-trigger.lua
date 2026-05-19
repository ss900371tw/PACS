function OnStoredInstance(instanceId, tags, metadata)
    -- 1. 取得 SOPInstanceUID (單張影像的唯一 ID)
    local sopInstanceId = tags['SOPInstanceUID'] or instanceId or 'UnknownSOP'
    local studyId = tags['StudyInstanceUID'] or tags['StudyID'] or 'UnknownStudy'
    
    -- 檢查是否為 Unknown，如果是就不用傳送了
    if sopInstanceId == 'UnknownSOP' then
        return
    end

    print('偵測到新檔案: ' .. sopInstanceId .. '，正在通知後端 AI 自動標籤服務...')
    
    -- 2. 使用你的電腦實體 IP (或 localhost 如果在同一台)
    local url = 'http://192.168.2.110:5000/webhook'
    
    -- 3. 打包 JSON payload
    local payload = '{"studyId": "' .. studyId .. '", "sopInstanceId": "' .. sopInstanceId .. '"}'
    
    -- 4. 設定 Header
    local headers = {
        ["Content-Type"] = "application/json"
    }
    
    -- 5. 發送 POST 請求
    -- 使用 pcall (protected call) 防止網路異常導致整個 Orthanc 崩潰
    pcall(function()
        HttpPost(url, payload, headers)
    end)
end

    

    

    

    

