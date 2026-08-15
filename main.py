from pathlib import Path
from build_embeddings import EmbeddingBuilder
from rag_agent import RAGAgent
from llm_wrapper import OpenRouterLLM
from retriever import Retriever

# User and system prompts
user_prompt = (
    "از متن‌های زیر برای پاسخ دقیق به سؤال استفاده کن. "
    "اگر پاسخ در متن‌ها نبود، صادقانه بگو که اطلاعات کافی وجود ندارد. "
    "در پایان هر پاسخ، از کاربر بپرس آیا سؤال دیگری دارد یا می‌خواهد ادامه دهد."
)

system_prompt = "شما یک دستیار فارسی هستید که پاسخ‌ها را به صورت خلاصه اما دقیق ارائه می‌دهد."

# Embedding file paths (must match EmbeddingBuilder definitions)
CHUNKS_PATH = Path("Data/chunks.json")
METADATA_PATH = Path("Data/metadata.json")
EMBEDDINGS_PATH = Path("Data/embeddings.npy")

print("\n🔹 Checking for existing embeddings...")

if all(p.exists() for p in [CHUNKS_PATH, METADATA_PATH, EMBEDDINGS_PATH]):
    print("✅ Precomputed embeddings found. Skipping build.\n")
else:
    print("⚙️ Embedding files missing. Starting build...")
    embd = EmbeddingBuilder()
    embd.build()
    print("✅ Embeddings built successfully.\n")

# Initialize retriever and LLM
print("🔹 Initializing retriever & LLM...")
ret = Retriever()
llm = OpenRouterLLM(model="deepseek-v4-flash", system_prompt=system_prompt)
print(f"✅ Agent powered by: {llm.model}\n")

rag_agent = RAGAgent(ret, llm, user_prompt=user_prompt)

# Query loop
while True:
    user_query = input("Please ask your question (type 'exit' to quit):\n ")
    if user_query.lower().strip() in ["exit", "quit", "خروج"]:
        print("Goodbye 👋")
        break

    answer = rag_agent.answer(user_query)
    print(f"\nAgent's Answer:\n{answer}\n")

#hghghjhmjhj,