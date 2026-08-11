import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# =====================================================
# 1. LOAD EMBEDDING MODEL
# =====================================================

persistent_directory = "db/chroma_db"


print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# =====================================================
# 2. LOAD CHROMA DATABASE
# =====================================================

print("Loading Chroma database...")


db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={
        "hnsw:space": "cosine"
    }
)


print("Chroma loaded successfully")



# =====================================================
# 3. LOAD QWEN MODEL
# =====================================================

print("\nLoading Qwen model...")


model_id = "Qwen/Qwen2.5-1.5B-Instruct"



tokenizer = AutoTokenizer.from_pretrained(
    model_id
)



model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)



print("Qwen loaded successfully")



# =====================================================
# 4. CHAT MEMORY
# =====================================================


chat_history = []



# =====================================================
# 5. FUNCTION TO GENERATE ANSWER
# =====================================================


def generate_answer(messages):

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)



    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )



    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )


    return answer.strip()




# =====================================================
# 6. ASK QUESTION FUNCTION
# =====================================================


def ask_question(user_question):


    global chat_history



    print("\n==============================")
    print("QUESTION")
    print("==============================")

    print(user_question)



    # -----------------------------
    # Retrieve documents
    # -----------------------------


    retriever = db.as_retriever(
        search_kwargs={
            "k": 3
        }
    )


    relevant_docs = retriever.invoke(
        user_question
    )



    print("\nDocuments found:", len(relevant_docs))



    context = "\n\n".join(
        [
            doc.page_content
            for doc in relevant_docs
        ]
    )



    # -----------------------------
    # Previous conversation
    # -----------------------------


    history = ""

    for chat in chat_history:

        history += f"""
User:
{chat['question']}

Assistant:
{chat['answer']}

"""



    # -----------------------------
    # Create Qwen prompt
    # -----------------------------


    messages = [

        {
            "role": "system",
            "content":
            """
You are a helpful RAG assistant.

Rules:
1. Answer only using the provided documents.
2. Use conversation history if needed.
3. If information is missing, say:
"I don't have enough information."
"""
        },


        {
            "role": "user",
            "content":
            f"""

Conversation history:

{history}


Current question:

{user_question}



Documents:

{context}



Answer:
"""
        }

    ]



    # -----------------------------
    # Generate answer
    # -----------------------------


    answer = generate_answer(messages)



    print("\n==============================")
    print("ANSWER")
    print("==============================")

    print(answer)



    # -----------------------------
    # Save memory
    # -----------------------------


    chat_history.append(
        {
            "question": user_question,
            "answer": answer
        }
    )



    return answer




# =====================================================
# 7. CHAT LOOP
# =====================================================


def start_chat():


    print("\n================================")
    print("RAG CHATBOT READY")
    print("Type 'quit' to exit")
    print("================================")



    while True:


        question = input("\nYou: ")



        if question.lower() == "quit":

            print("Goodbye!")
            break



        ask_question(question)




# =====================================================
# START CHATBOT
# =====================================================


if __name__ == "__main__":

    start_chat()