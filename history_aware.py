import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# =====================================================
# 2. SELECT DEVICE
# =====================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

    print("CUDA GPU detected.")

else:

    DEVICE = "cpu"

    print("CUDA GPU not detected. Using CPU.")


print(f"Using device: {DEVICE}")


# =====================================================
# 3. LOAD TOKENIZER
# =====================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

print("Tokenizer loaded successfully.")


# =====================================================
# 4. LOAD QWEN MODEL
# =====================================================

print("\nLoading Qwen model...")

if DEVICE == "cuda":

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16
    )

    model = model.to("cuda")

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32
    )

    model = model.to("cpu")


model.eval()

print("Qwen model loaded successfully.")


# =====================================================
# 5. GENERATE RESPONSE
# =====================================================

def generate_response(
    messages,
    max_new_tokens=100
):

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    # -----------------------------------------------
    # Tokenize
    # -----------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )


    # -----------------------------------------------
    # Move inputs to same device as model
    # -----------------------------------------------

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    # -----------------------------------------------
    # Generate
    # -----------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )


    # -----------------------------------------------
    # Remove prompt from output
    # -----------------------------------------------

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[-1]:
    ]


    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    return response.strip()


# =====================================================
# 6. HISTORY-AWARE QUESTION REWRITER
# =====================================================

def make_history_aware_question(
    user_question,
    chat_history
):

    # -----------------------------------------------
    # If there is no history, return original question
    # -----------------------------------------------

    if not chat_history:

        return user_question


    # -----------------------------------------------
    # Build conversation history
    # -----------------------------------------------

    history = ""

    for chat in chat_history:

        history += f"""
User:
{chat["question"]}

Assistant:
{chat["answer"]}

"""


    # -----------------------------------------------
    # Create Qwen messages
    # -----------------------------------------------

    messages = [

        {
            "role": "system",
            "content": """
You are a history-aware question rewriter.

Your job is to rewrite the user's latest question
into a standalone question for vector database
retrieval.

Rules:

1. Use the conversation history to understand
   references such as:
   "it", "this", "that", "they", "he", "she",
   "the above", and "what about it".

2. Include necessary context from the previous
   conversation.

3. Do NOT answer the question.

4. Do NOT add information that is not present
   in the conversation.

5. Return ONLY the rewritten standalone question.

6. If the question is already standalone,
   return it unchanged.
"""
        },

        {
            "role": "user",
            "content": f"""
Conversation history:

{history}

Latest user question:

{user_question}

Standalone retrieval question:
"""
        }

    ]


    # -----------------------------------------------
    # Generate rewritten question
    # -----------------------------------------------

    rewritten_question = generate_response(
        messages,
        max_new_tokens=100
    )


    return rewritten_question.strip()
