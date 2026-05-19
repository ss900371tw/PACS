function OnStoredInstance(instanceId, tags, metadata)
    print('偵測到新檔案上傳，正在發送影像資訊給 Windows 端...')
    
    -- 使用你的電腦實體 IP
    local url = 'http://localhost:5000/webhook'
    
    -- 1. 取得 StudyInstanceUID (若無則用備用欄位)
    local studyId = tags['StudyInstanceUID'] or tags['StudyID'] or 'UnknownStudy'
    
    -- 2. 取得 SOPInstanceUID (這代表單張影像的唯一 ID)
    -- 如果 DICOM 內找不到，可以用 Orthanc 傳進來的 instanceId 丟給後端做備用
    local sopInstanceId = tags['SOPInstanceUID'] or instanceId or 'UnknownSOP'
    
    -- 3. 將多個欄位包裝成標準 JSON 字串
    -- 注意：這裡手動拼接字串，請確保欄位與逗號標點符號正確
    local payload = '{"studyId": "' .. studyId .. '", "sopInstanceId": "' .. sopInstanceId .. '"}'
    
    -- 4. 設定 Header 告訴 Flask 這是 JSON
    local headers = {
        ["Content-Type"] = "application/json"
    }
    
    -- 5. 發送 POST 請求
    HttpPost(url, payload, headers)
end


    

    

