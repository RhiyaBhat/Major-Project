# src/rag/rag_chain.py

from typing import Dict, Any, List

from src.core.vectorstore import get_vectorstore
from src.core.llm import get_llm


RAG_PROMPT = """
You are a helpful AI assistant answering questions about the LlamaChain project and its documentation.

**Critical Instructions:**
1. The user is asking about the LlamaChain project specifically (not papers in the literature review).
2. Look for information in sections titled "Objectives", "Introduction", "Methodology", or "Conclusion".
3. If you see references to other papers (like TOMDS, MuDoC), those are from the literature review - ignore them unless specifically asked.
4. Answer clearly and directly based on the LlamaChain project context provided below.
5. Use bullet points or numbered lists when listing multiple items.

**Context from LlamaChain Project Documents:**
{context}

**User Question:**
{question}

**Your Answer (about LlamaChain project only):**
"""


def build_rag_chain():
    vs = get_vectorstore()
    llm = get_llm()

    def rag(question: str) -> Dict[str, Any]:
        # Enhanced retrieval: search with multiple query variations
        queries = [question]
        
        # Add query expansion for common questions
        if "objective" in question.lower():
            queries.extend([
                "LlamaChain objectives aims goals",
                "project objectives section 5",
                "what does LlamaChain aim to achieve"
            ])
        
        # Collect documents from multiple query variations
        all_docs = []
        seen_ids = set()
        
        for q in queries:
            docs = vs.similarity_search(q, k=6)
            for doc in docs:
                doc_id = id(doc)
                if doc_id not in seen_ids:
                    all_docs.append(doc)
                    seen_ids.add(doc_id)
        
        # Prioritize documents from "Objectives" section
        def doc_priority(doc):
            content_lower = doc.page_content.lower()
            meta = doc.metadata or {}
            
            # Highest priority: contains "Objectives" heading
            if "objectives" in content_lower and ("5." in doc.page_content or "objective" in content_lower[:100]):
                return 0
            # High priority: page 3 (where objectives are)
            elif meta.get("page_number") == 3:
                return 1
            # Medium: mentions project goals/aims
            elif any(word in content_lower for word in ["llama", "llamachain", "system will", "project will"]):
                return 2
            # Low: literature review content
            elif any(word in content_lower for word in ["authors:", "publication date:", "this paper", "this research"]):
                return 10
            else:
                return 5
        
        all_docs.sort(key=doc_priority)
        docs = all_docs[:10]  # Take top 10 after sorting
        
        print(f"\n[DEBUG] Retrieved {len(docs)} documents after deduplication and ranking")

        # Build context string
        context_parts = []
        total_chars = 0
        max_chars = 5000

        for i, d in enumerate(docs):
            meta = d.metadata or {}
            page_num = meta.get('page_number', '?')
            
            header = (
                f"[Source {i+1}: {meta.get('file_name', '?')} - "
                f"Page {page_num} - "
                f"{meta.get('modality', '?')}]\n"
            )
            body = d.page_content.strip()
            if not body:
                continue

            # Debug first few chunks
            if i < 3:
                print(f"[DEBUG] Chunk {i+1} (Page {page_num}): {body[:150]}...")

            text = header + body
            if total_chars + len(text) > max_chars and context_parts:
                break

            context_parts.append(text)
            total_chars += len(text)

        if context_parts:
            context = "\n\n".join(context_parts)
            print(f"[DEBUG] Total context length: {len(context)} chars, {len(context_parts)} chunks used")
        else:
            context = "No relevant context found."
            print("[DEBUG] WARNING: No context parts generated!")

        # Build prompt and call LLM
        prompt = RAG_PROMPT.format(context=context, question=question)
        response = llm.invoke(prompt)
        answer_text = getattr(response, "content", str(response))

        print(f"[DEBUG] Answer preview: {answer_text[:200]}...\n")

        return {
            "answer": answer_text,
            "source_documents": docs[:5],  # Return top 5 for display
        }

    return rag