from processor_regex import classify_with_regex
from sentence_transformers import SentenceTransformer
from processor_bert_logistic import classify_with_bert
import joblib
from processor_llm import classify_with_llm
import pandas as pd

def classify(logs):
    labels = []
    for source, log in logs:
        label = log_classify(source,log)
        labels.append(label)
    return labels
        
        
def log_classify(source,log_msg):
    if source == 'LegacyCRM':
         label = classify_with_llm(log_msg) #llm
    else:
        label = classify_with_regex(log_msg)
        if label is None:
            label = classify_with_bert(log_msg)
            if label is None:
                label = "Unclassified"
    return label

def classify_with_csv(input_csv):
    df = pd.read_csv(input_csv, encoding='latin1', delimiter=',')

    df['predicted_label'] = classify(list(zip(df['source'],df['log_message'])))
    output_file = 'resources/output.csv'
    df.to_csv(output_file,index=False)


if __name__=='__main__':
    
    classify_with_csv('resources/test.csv')
    
    # logs = [('ModernCRM','nova.osapi_compute.wsgi.server [req-b9718cd8-f65e-49cc-8349-6cf7122af137 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 ""GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail HTTP/1.1"" status: 200 len: 1893 time: 0.2675118.'),
    #         ('ModernCRM','Email service experiencing issues with sending.'),
    #         ('LegacyCRM','Lead conversion failed for prospect ID 7842 due to missing contact information.'),
    #         ('AnalyticsEngine',"nova.osapi_compute.wsgi.server [req-01d570b0-78a7-4719-b7a3-429fd7dc5a3f 113d3a99c3da401fbd62cc2caa5b96d2 54fadb412c4e40cdbaed9335e4c35a9e - - -] 10.11.10.1 ""POST /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers HTTP/1.1"" Status code -  202 len: 733 time: 0.5130808."),
    #         ('','Hey bro whats up.'),
    #         ('','User User123 logged in.')]
    
    # rst = classify(logs)
    # print(rst)
        