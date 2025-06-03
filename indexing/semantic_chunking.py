import pdfplumber
from sentence_transformers import SentenceTransformer, util
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

def is_heading(text):
    return text.strip().isupper() or text.strip().startswith(tuple(str(i) for i in range(1, 100)))

def extract_paragraphs(text):
    lines = text.split('\n')
    paragraphs = []
    current_para = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        else:
            current_para.append(stripped)

    if current_para:
        paragraphs.append(" ".join(current_para))

    return paragraphs

def chunk_pdf(file_path, similarity_threshold=0.75, max_tokens=700):
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        current_section = None
        buffer = []
        paragraph_meta = []

        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            paras = extract_paragraphs(text)
            for para in paras:
                if is_heading(para):
                    if buffer:
                        chunks.extend(create_semantic_chunks(
                            buffer, current_section, paragraph_meta, model, similarity_threshold, max_tokens
                        ))
                        buffer = []
                        paragraph_meta = []
                    current_section = para
                else:
                    buffer.append(para)
                    paragraph_meta.append({"section": current_section, "page": page_num})
        
        # flush remaining buffer
        if buffer:
            chunks.extend(create_semantic_chunks(buffer, current_section, paragraph_meta, model, similarity_threshold, max_tokens))

    return chunks

def create_semantic_chunks(paragraphs, section_title, metadata_list, model, similarity_threshold, max_tokens):
    chunks = []
    current_chunk = []
    chunk_meta = []

    para_embeddings = model.encode(paragraphs, convert_to_tensor=True)

    for i, paragraph in enumerate(paragraphs):
        if not current_chunk:
            current_chunk.append(paragraph)
            chunk_meta.append(metadata_list[i])
            continue

        sim_score = util.cos_sim(para_embeddings[i], para_embeddings[i - 1])
        current_len = len(" ".join(current_chunk).split())

        if sim_score >= similarity_threshold or current_len < 200:
            current_chunk.append(paragraph)
            chunk_meta.append(metadata_list[i])
        else:
            chunks.append({
                "section": section_title,
                "chunk": " ".join(current_chunk),
                "meta": chunk_meta.copy()
            })
            # overlap logic
            current_chunk = [current_chunk[-1], paragraph]
            chunk_meta = [chunk_meta[-1], metadata_list[i]]

    if current_chunk:
        chunks.append({
            "section": section_title,
            "chunk": " ".join(current_chunk),
            "meta": chunk_meta.copy()
        })

    return chunks


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

# ======= Run it ===========
if __name__ == "__main__":
    pdf_path = "resources/manual.pdf"  # change this
    semantic_chunks = chunk_pdf(pdf_path)
    # save_chunks_to_json(semantic_chunks, "resources/semantic_chunks.pdf")

    with open("oracle_semantic_chunks.json", "w") as f:
        json.dump(semantic_chunks, f, indent=2)

    print(f"Extracted {len(semantic_chunks)} semantic chunks.")