import fitz  # PyMuPDF
import spacy
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from ..utils.config import load_params

# Load models once (global)
params = load_params()
nlp = spacy.load("en_core_sci_sm")
tokenizer = AutoTokenizer.from_pretrained(params["nlp"]["model_name"])
scibert_model = AutoModel.from_pretrained(params["nlp"]["model_name"])

def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF."""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def tokenize_text(text):
    """Tokenize and clean text using SciSpaCy."""
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def get_scibert_embedding(text):
    """Generate SciBERT embeddings for text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=params["nlp"]["max_token_length"]
    )
    with torch.no_grad():
        outputs = scibert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def process_papers(pdf_dir, output_dir):
    """Process all PDFs into embeddings."""
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        text = extract_text_from_pdf(pdf_path)
        tokens = tokenize_text(text)
        embedding = get_scibert_embedding(tokens)
        results.append({
            "pdf_path": str(pdf_path),
            "text": text,
            "embedding": embedding
        })

    df = pd.DataFrame(results)
    csv_path = output_dir / "processed_papers.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved processed papers to {csv_path}")

if __name__ == "__main__":
    params = load_params()
    process_papers(
        pdf_dir=params["arxiv"]["save_dir"],
        output_dir=params["features"]["output_dir"]
    )