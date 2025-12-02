# src/rag/rag_chain.py

from typing import Dict, Any, List

from src.core.vectorstore import get_vectorstore
from src.core.llm import get_llm


RAG_PROMPT = """
You are an intelligent assistant that answers questions based on document context.

Rules:
- First, carefully read the context.
- If the context directly answers the question, use it.
- If the context is partial or fragmented, infer a clear answer from it.
- Only if there is truly no relevant information, say exactly:
  "No relevant information found in the uploaded documents."
- Prefer concise answers. If the user asks for bullet points, use bullet points.
- Do NOT talk about the retrieval process itself.

Context:
{context}

Question:
{question}

Answer:
"""


def build_rag_chain():
    vs = get_vectorstore()
    llm = get_llm()

    def rag(question: str) -> Dict[str, Any]:
        # 1) Retrieve top-k similar chunks
        docs: List = vs.similarity_search(question, k=8)

        # 2) Build context string with some source info, but keep it under a limit
        context_parts = []
        total_chars = 0
        max_chars = 4000  # keep prompt manageable

        for d in docs:
            meta = d.metadata or {}
            header = (
                f"[Source: {meta.get('file_name', '?')} - "
                f"page {meta.get('page_number', '?')} - "
                f"modality={meta.get('modality', '?')}]\n"
            )
            body = d.page_content.strip()
            if not body:
                continue

            text = header + body
            if total_chars + len(text) > max_chars and context_parts:
                break

            context_parts.append(text)
            total_chars += len(text)

        if context_parts:
            context = "\n\n".join(context_parts)
        else:
            context = "No relevant context found."

        # 3) Build the final prompt and call local LLM (Ollama)
        prompt = RAG_PROMPT.format(context=context, question=question)
        response = llm.invoke(prompt)

        # ChatOllama returns an AIMessage; extract plain text
        answer_text = getattr(response, "content", str(response))

        return {
            "answer": answer_text,
            "source_documents": docs,
        }

    return rag
