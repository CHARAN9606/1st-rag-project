from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ============================================================
# 1. LOAD FREE LOCAL MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading free local model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded successfully!")


# ============================================================
# 2. TESLA TEXT TO CHUNK
# ============================================================

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


# ============================================================
# 3. CREATE AGENTIC CHUNKING PROMPT
# ============================================================

prompt = f"""
You are a text chunking expert.

Split the following text into logical chunks.

Rules:

- Each chunk should be around 200 characters or less.
- Split at natural topic boundaries.
- Keep related information together.
- Do not rewrite the original text.
- Do not summarize the text.
- Keep the original wording.
- Put <<<SPLIT>>> between chunks.
- Return ONLY the chunked text.

Text:

{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split.
"""


# ============================================================
# 4. CREATE CHAT INPUT
# ============================================================

messages = [
    {
        "role": "system",
        "content": (
            "You are an expert document chunking assistant. "
            "Follow the user's chunking instructions exactly."
        )
    },
    {
        "role": "user",
        "content": prompt
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


# ============================================================
# 5. TOKENIZE
# ============================================================

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)


# ============================================================
# 6. ASK THE LOCAL MODEL TO CHUNK
# ============================================================

print("\n🤖 Asking local AI model to chunk the text...")

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False
    )


# ============================================================
# 7. REMOVE THE ORIGINAL PROMPT
# ============================================================

generated_tokens = outputs[
    0
][
    inputs["input_ids"].shape[1]:
]

marked_text = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()


# ============================================================
# 8. SPLIT USING <<<SPLIT>>>
# ============================================================

chunks = marked_text.split(
    "<<<SPLIT>>>"
)


# ============================================================
# 9. CLEAN CHUNKS
# ============================================================

clean_chunks = []

for chunk in chunks:

    cleaned = chunk.strip()

    if cleaned:
        clean_chunks.append(cleaned)


# ============================================================
# 10. SHOW RESULTS
# ============================================================

print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)

for i, chunk in enumerate(
    clean_chunks,
    1
):

    print(
        f"Chunk {i}: ({len(chunk)} chars)"
    )

    print(
        f'"{chunk}"'
    )

    print()