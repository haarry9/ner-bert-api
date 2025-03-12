# Named Entity Recognition (NER) API with BERT

![techstack](tech_stack.png)

## Overview

The aim of the project is to implement a Named Entity Recognition (NER) API using a fine-tuned BERT model trained on the CoNLL-2003 dataset. The trained model is integrated into a FastAPI-based REST service that exposes an endpoint /predict that takes input text as JSON and returns the recognized entities. 

## Features
- **Data Preprocessing**: Preprocessed the data in Google Colab, including tokenization, ddataset formatting, including padding and word embeddings.
- **Model Training**: Fine-tuned the pretrained BERT model from Hugging Face on the CoNLL-2003 dataset for NER using Google Colab.
- **Model Evaluation**: Assessed model performance using Precision, Recall, and F1-score.
- **Model Serialization & Deployment**: The trained model and tokenizers were saved to local environment for inference.
- **API Integration**: Implemented a FastAPI `/predict` endpoint for real-time entity recognition.
- **Future Enhancements**: Dockerization and deployment on AWS for scalable access.

## Dataset
![conll-2003 dataset](conll-2003.png)

The model is trained on the [CoNLL-2003 dataset](https://huggingface.co/datasets/eriktks/conll2003), a standard benchmark for Named Entity Recognition tasks. It consists of newswire articles where entities are labeled into four categories: Person (PER), Organization (ORG), Location (LOC), and Miscellaneous (MISC). 

## Project Structure

```
├── app/                 # FastAPI service
│   ├── main.py          # API script
├── notebooks/           # Notebooks containing Data Preprocessing and model training (Google Colab)
├── scripts/             # Utility scripts (e.g., model downloader script)
├── .gitignore           # Excludes model files from GitHub
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```


## Installation

### Prerequisities
- Python 3.8+
- - Virtual environment (recommended)

### Setup

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv venv
# On Linux/Mac
source venv/bin/activate  
# On Windows
venv\Scripts\activate

pip install -r requirements.txt
```
## Model Folder Storage and Retrieval
Since the size of the model files exceed 400 MB's, they are not stored in the repository. The model folder is uploaded to Google Drive [here](https://drive.google.com/file/d/1TjtdVLCQ1So2TC6ylo6Rttyp2l_yXpvX/view?usp=drive_link). The model folder is ignored in `.gitignore`, and a python script is provided to download the model before running the API.

### Downloading the Model Folder
Before running the API, download the model by executing:
```bash
python scripts/download_model.py
```
This script will download the fine-tuned model from Google Drive and store it in the appropriate directory.
The project directory after running the scripts to download the model folder will be as:
```
  ├── model/               # Serialized model and tokenizer
  ├── app/                 # FastAPI service
  │   ├── main.py          # API script
  ├── notebooks/           # Notebooks containg Data Preprocessing and model training (Google Colab)
  ├── scripts/             # Utility scripts (e.g., model download script)
  ├── .gitignore           # Excludes model files from GitHub
  ├── README.md            # Project documentation
  └── requirements.txt     # Python dependencies
```
Here, the model folder contains the following files:
```
model/
├── ner_bert_model/
│   ├── config.json
│   ├── model.safetensors
├── tokenizer/
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── vocab.txt
```
## Running the NER BERT API
```bash
cd app
uvicorn main:app --reload
```

## Test the API endpoint:
```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  --data-raw '{
    "input": "NASA and SpaceX launched the Artemis II mission, with astronauts training at Kennedy Space Center before heading to the ISS."
  }'
```

## Testing API via FastAPI's Interactive Docs
- Open `http://127.0.0.1:8000/docs` in a browser.
- Click POST /predict, enter a sentence, and click Execute.

### FastAPI Swagger UI 
![swagger UI 1](ui1.png)
![swagger UI 2](ui2.png)



# TODO:
- [ ] **Containerization**: Using a Docker-based deployment strategy.
- [ ] **Cloud Deployment**: Deploing the API to AWS Lambda/ECS for accessibility.


## Future Enhancements:
- Implement API authentication and request validation.
- Introduce logging and exception handling for robustness.
- Develop a Streamlit-based UI for interactive entity recognition.

## Acknowledgments
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Google Colab](https://colab.research.google.com/)
- [CoNLL-2003 Dataset](https://www.kaggle.com/datasets/alaakhaled/conll003-ner)
