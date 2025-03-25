from sentence_transformers import SentenceTransformer
import joblib

import os
file_path = r'D:/Self Study/6 Log Classification Model/models/logistic_classifier.joblib'


transformer_model = SentenceTransformer('all-MiniLM-L6-v2') #SBERT
#msg_embedding = transformer_model.encode(log_msg)
classifier_model = joblib.load(file_path)

def classify_with_bert(log_msg):
    msg_embedding = transformer_model.encode(log_msg)
    predict_probability = classifier_model.predict_proba([msg_embedding])[0]
    predicted_class=None
    if max(predict_probability)>0.5:
        predicted_class = classifier_model.predict([msg_embedding])[0]
    return (predicted_class)


if __name__ == '__main__':
    logs = ['nova.osapi_compute.wsgi.server [req-b9718cd8-f65e-49cc-8349-6cf7122af137 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 ""GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1"" status: 200 len: 1893 time: 0.2675118.',
            'Email service experiencing issues with sending.',
            'Lead conversion failed for prospect ID 7842 due to missing contact information.',
            "nova.osapi_compute.wsgi.server [req-01d570b0-78a7-4719-b7a3-429fd7dc5a3f 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 ""POST /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers HTTP/1.1"" Status code -  202 len: 733 time: 0.5130808.",
            'Hey bro whats up.',
            'User User123 logged in.']
    for log in logs:
        #print(log,'->',classify_with_bert(log))
        print(classify_with_bert(log))
    