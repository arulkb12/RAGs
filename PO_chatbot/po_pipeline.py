import faiss
import numpy as np
import pickle
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

STORE_PATH = "po_store.pkl"
DIM = 1536


# -------------------------
# Embedding
# -------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# -------------------------
# 🔥 UPDATED: Structured Extraction (Robust Version)
# -------------------------
def extract_invoice_json(text):

    # 🔥 Guard against empty OCR text
    if not text or len(text.strip()) < 20:
        return {
            "invoice_number": None,
            "po_number": None,
            "vendor_name": None,
            "invoice_date": None,
            "total_amount": None,
            "currency": None,
            "shipping_address": None,
            "error": "Insufficient or unreadable invoice text"
        }

    prompt = f"""
You are a financial document parser.

Convert the unstructured invoice text into structured JSON.

Return ONLY valid JSON.
Do not include explanation.
Do not include markdown.
Do not wrap in ```.

Required fields:
- invoice_number (string or null)
- po_number (string or null)
- vendor_name (string or null)
- invoice_date (string or null)
- total_amount (number or null)
- currency (string or null)
- shipping_address (string or null)

If any field is missing, return null.

Invoice Text:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},  # 🔥 FORCE JSON MODE
        messages=[
            {"role": "system", "content": "Return structured invoice data as valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        structured_data = json.loads(raw_output)

        # 🔥 Ensure all required fields exist
        required_fields = [
            "invoice_number",
            "po_number",
            "vendor_name",
            "invoice_date",
            "total_amount",
            "currency",
            "shipping_address"
        ]

        for field in required_fields:
            if field not in structured_data:
                structured_data[field] = None

        return structured_data

    except Exception:
        print("⚠ JSON Parsing Failed. Raw Model Output:")
        print(raw_output)

        return {
            "invoice_number": None,
            "po_number": None,
            "vendor_name": None,
            "invoice_date": None,
            "total_amount": None,
            "currency": None,
            "shipping_address": None,
            "error": "Invalid JSON returned from model"
        }


# -------------------------
# Storage
# -------------------------
def load_store():
    if os.path.exists(STORE_PATH):
        with open(STORE_PATH, "rb") as f:
            return pickle.load(f)
    else:
        index = faiss.IndexFlatL2(DIM)
        return {"index": index, "data": []}


def save_store(store):
    with open(STORE_PATH, "wb") as f:
        pickle.dump(store, f)


# -------------------------
# 🔥 UPDATED: Add Invoice (Now stores structured format only)
# -------------------------
def add_invoice(text):

    store = load_store()

    invoice_json = extract_invoice_json(text)

    # 🔥 Convert structured JSON to clean summary string
    summary = json.dumps(invoice_json, indent=2)

    embedding = get_embedding(summary)

    store["index"].add(np.array([embedding]).astype("float32"))
    store["data"].append(invoice_json)

    save_store(store)

    return invoice_json


# -------------------------
# Query
# -------------------------
def ask_question(question):

    store = load_store()

    if store["index"].ntotal == 0:
        return "No invoices uploaded."

    q_embedding = get_embedding(question)

    D, I = store["index"].search(
        np.array([q_embedding]).astype("float32"), 3
    )

    matched = [store["data"][i] for i in I[0]]

    context = json.dumps(matched, indent=2)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[{
            "role": "user",
            "content": f"""
Answer using ONLY the structured invoice JSON data below.

{context}

Question:
{question}
"""
        }]
    )

    return response.choices[0].message.content
