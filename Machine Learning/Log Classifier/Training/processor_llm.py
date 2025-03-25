from groq import Groq
from dotenv import load_dotenv
load_dotenv()

groq=Groq()

def classify_with_llm(log_msg):
    
    prompt = f'''Classify the log messages into one these two categories: (1) Workflow Error,
    (2) Deprecation Warning.
    if you can't figure it out a category, return 'Unclassified'.
    Only return the category name. No preamble.
    log message:{log_msg}'''
    
    chat_completion = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ])

    return(chat_completion.choices[0].message.content)

if __name__=="__main__":
    print(classify_with_llm('Lead conversion failed for prospect ID 7842 due to missing contact information.'))
