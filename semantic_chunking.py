from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# Same Tesla text
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""


# FREE LOCAL EMBEDDING MODEL
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Semantic Chunker
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)


# Create semantic chunks
chunks = semantic_splitter.split_text(
    tesla_text
)


# Display results
print("SEMANTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(chunks, 1):

    print(
        f"Chunk {i}: ({len(chunk)} chars)"
    )

    print(f'"{chunk}"')
    print()