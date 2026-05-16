function OnStoredInstance(instanceId, tags, metadata)
    print('偵測到新檔案上傳，正在發送 ID 給 Windows 端...')
    
    -- 使用你的電腦實體 IP
    local url = 'http://10.242.13.179:5000/webhook'
    
    -- 將 instanceId 包裝成 JSON 字串
    local payload = '{"instanceId": "' .. instanceId .. '"}'
    
    -- 發送 POST 請求，並帶上 JSON 資料
    HttpPost(url, payload)
end