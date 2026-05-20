function OnStoredInstance(instanceId, tags, metadata)
    -- 【核心安全檢查】如果是 Orthanc 內部 Modify 產生來的，直接跳過，斷絕無窮迴圈
    if metadata and (metadata['ModifiedFrom'] or metadata['AnonymizedFrom']) then

        return
    end

    -- 【終極修正】強迫向 Orthanc 請求最完整、最真實的 DICOM 標籤資訊
    local response = ParseJson(RestApiGet('/instances/' .. instanceId .. '/tags?short'))
    
    -- DICOM 標準標籤：0008,0018 是 SOPInstanceUID
    local sopInstanceId = response['0008,0018']
    -- DICOM 標準標籤：0020,000d 是 StudyInstanceUID
    local studyId = response['0020,000d'] or 'UnknownStudy'

    -- 如果真的拿不到真實 SOPUID，再退求其次使用傳入的 tags
    if not sopInstanceId then
        sopInstanceId = tags['SOPInstanceUID'] or instanceId or 'UnknownSOP'
    end

    if sopInstanceId == 'UnknownSOP' then
        return
    end

    
    -- 【終極突破】在不改 docker-compose 的情況下，直接走 Docker 預設網關連回外面的實體機
    local url = 'http://172.17.0.1:5000/webhook'
    -- local url = 'http://host.docker.internal:5000/webhook'

    local payload = '{"studyId": "' .. studyId .. '", "sopInstanceId": "' .. sopInstanceId .. '"}'
    local headers = {
        ["Content-Type"] = "application/json"
    }
    
    pcall(function()
        HttpPost(url, payload, headers)
    end)
end


    

    

    

