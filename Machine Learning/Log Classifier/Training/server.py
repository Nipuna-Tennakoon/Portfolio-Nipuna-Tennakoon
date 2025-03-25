import pandas as pd
import os
from fastapi import FastAPI, UploadFile, HTTPException,File
from fastapi.responses import FileResponse
from processor_main import classify

app = FastAPI()

@app.post('/classify/')
async def classify_logs(file: UploadFile = File(...)):
    # Ensure the file is a CSV
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='File must be a CSV.')

    try:
        # Read CSV into a DataFrame
        df = pd.read_csv(file.file, encoding='latin1', delimiter=',')

        # Check required columns
        if "source" not in df.columns or "log_message" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'source' and 'log_message' columns.")

        # Run classification (ensure classify() function is defined)
        df['target_label'] = classify(list(zip(df['source'], df['log_message'])))

        # Define output file path
        output_dir = os.path.abspath(os.path.join(os.getcwd(), "../resources"))  # Move up one level
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        output_file = os.path.join(output_dir, "output.csv")
        #os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure directory exists
        df.to_csv(output_file, index=False)

        # Return classified CSV
        return FileResponse(output_file, media_type="text/csv", filename="classified_output.csv")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        file.file.close()
