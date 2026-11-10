
from importlib import import_module
import numpy as np

try:
    SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
except ModuleNotFoundError as error:
    if error.name != "sentence_transformers":
        raise
    raise SystemExit(
        "Missing dependency. Install it with: "
        "python -m pip install sentence-transformers"
    ) from error

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read knowledge file
with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split text into chunks
def chunk_text(text, chunk_size=45, overlap=10):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks

chunks = chunk_text(text)

print(f"\nCreated {len(chunks)} chunks.")

# Generate embeddings
embeddings = model.encode(chunks)

# Three test questions
questions = [
    "What is Artificial Intelligence?",
    "What is the difference between Machine Learning and Deep Learning?",
    "Who invented Artificial Intelligence?"
]

# Retrieve relevant chunks
for question in questions:
    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    question_embedding = model.encode([question])[0]

    similarities = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(question_embedding)
    )

    top_indices = np.argsort(similarities)[::-1][:2]

    print("\nRETRIEVED CHUNKS:")

    for index in top_indices:
        print(f"\nSimilarity: {similarities[index]:.4f}")
        print(chunks[index])

    if similarities[top_indices[0]] < 0.35:
        print("\nThe file may not contain enough information to answer this question.")

print("\nRAG retrieval completed successfully!")

