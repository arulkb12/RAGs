from pypdf import PdfReader
from docx import Document
import tempfile
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_image(path):
    # Convert image to base64
    with open(path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all invoice details clearly from this image. Return plain readable text."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


def extract_text(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        path = tmp.name

    text = ""

    if ext == "pdf":
        text = extract_text_from_pdf(path)

    elif ext in ["doc", "docx"]:
        text = extract_text_from_docx(path)

    elif ext in ["png", "jpg", "jpeg"]:
        text = extract_text_from_image(path)

    else:
        text = ""

    os.unlink(path)
    return text.strip()
