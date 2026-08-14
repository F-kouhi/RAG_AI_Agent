class RAGAgent:
    """
    RAG pipeline: query → retrieve → LLM → answer
    """

    def __init__(self, retriever, llm, user_prompt):
        self.retriever = retriever
        self.llm = llm
        self.user_prompt = user_prompt
        

    def answer(self, query: str, top_k: int = 5):
        results = self.retriever.retrieve(query, top_k=top_k)

        if not results:
            return "متأسفم، اطلاعات مرتبطی پیدا نشد."

        context_blocks = []
        for i, r in enumerate(results, 1):
            context_blocks.append(f"[{i}] {r['text']}")

        context_text = "\n\n".join(context_blocks)

        prompt = f"""
{self.user_prompt}
متن‌ها:
{context_text}

سؤال:
{query}
"""

        return self.llm.generate(prompt)
