import re

def classify_with_regex(log_message):
    label_patterns = {
        r"User User\d+ logged (in|out).":"User Action",
        r"Backup (started|ended) at.*":"System Notification",
        r"Backup completed successfully.":"System Notification",
        r"System updated to version.*":"System Notification",
        r"File.*uploaded successfully by user.*":"System Notification",
        r"Disk cleanup completed successfully.":"System Notification",
        r"System reboot initiated by user.*":"System Notification",
        r"Account with ID .*created by.*":"User Action",
                
        
    }
    for pattern, label in label_patterns.items():
        #if re.search(pattern,log_message):
        if re.search(pattern,log_message,re.IGNORECASE):
                return label
    return None
        
if __name__=='__main__':
    print(classify_with_regex('User User123 logged in.'))
    print(classify_with_regex('Backup started at 12:00.'))
    print(classify_with_regex('Backup completed successfully.'))
    print(classify_with_regex('System updated to version 1.0.0.'))
    print(classify_with_regex('Hey bro whats up'))
    print(classify_with_regex('nova.osapi_compute.wsgi.server [req-b9718cd8-f65e-49cc-8349-6cf7122af137 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 ""GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1"" status: 200 len: 1893 time: 0.2675118.'))