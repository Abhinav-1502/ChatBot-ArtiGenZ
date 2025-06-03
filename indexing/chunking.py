import fitz
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
    
def extract_paragraphs(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        blocks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 30]  # Ignore tiny lines
        for block in blocks:
            paragraphs.append({
                "text": block,
                "page": page_num
            })
    return paragraphs

def group_paragraphs(paragraphs, group_size=3):
    grouped = []
    for i in range(0, len(paragraphs), group_size):
        group = paragraphs[i:i+group_size]
        content = "\n\n".join([p["text"] for p in group])
        page_range = [p["page"] for p in group]
        metadata = {
            "page_start": page_range[0],
            "page_end": page_range[-1]
        }
        grouped.append(Document(page_content=content.strip(), metadata=metadata))
    return grouped

def chunk_documents(docs, chunk_size=700, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)



def save_chunks_to_json(chunks, output_path="chunks.json"):
    serialized = []
    for chunk in chunks:
        serialized.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "size": len(chunk.page_content)
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(serialized)} chunks to {output_path}")

# 📦 Full pipeline
if __name__ == "__main__":
    pdf_path = "resources/manual.pdf"
    paragraphs = extract_paragraphs(pdf_path)
    grouped_docs = group_paragraphs(paragraphs, group_size=3)
    chunks = chunk_documents(grouped_docs)
    
    save_chunks_to_json(chunks, "resources/chunks.json")
    