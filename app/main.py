import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForTokenClassification, BertTokenizerFast

# Define paths based on your local folder structure
model_path = "../model/ner_bert_model"  # Path to saved model
tokenizer_path = "../model/tokenizer"   # Path to saved tokenizer

# Load fine-tuned model and tokenizer
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
model_fine_tuned = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

# Load pipeline with optimized device setting
pipe = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer, device=device)

# Initialize FastAPI app
app = FastAPI(
    title="NER BERT API"
)


# Confirmation that app has started
print("FastAPI app is running...")

class RequestModel(BaseModel):
    input: str

@app.post("/predict")
def get_response(request: RequestModel):
    sentence = request.input
    ner_results = pipe(sentence, batch_size=8)

    # Format output properly
    formatted_results = [
        {
            "word": entity["word"],
            "entity": entity["entity"],
            "confidence": float(entity["score"])  # Convert numpy.float32 to Python float
        }
        for entity in ner_results
    ]
    
    return {"sentence": sentence, "entities": formatted_results}
