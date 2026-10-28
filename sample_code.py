from langchain_community.retrievers import BM25Retriever

retriever_bm25 = BM25Retriever.from_documents(chunks)
retriever_bm25.k = 5

retriever_vectorial = vectorstore.as_retriever(
    search_kwargs={"k": 5, "namespace": NAMESPACE},
)


from langchain_classic.retrievers import EnsembleRetriever

retriever_hibrido = EnsembleRetriever(
    retrievers=[retriever_bm25, retriever_vectorial],
    weights=[0.5, 0.5],
)


from typing import List, Dict


class RAGSystem:
    """Encapsula el EnsembleRetriever y expone un método simple para obtener el top-k."""

    def __init__(self, retriever, k: int = 5):
        self.retriever = retriever
        self.k = k

    def obtener_top_k(self, query: str) -> List[Dict]:
        docs = self.retriever.invoke(query)[: self.k]
        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("source", "desconocida"),
                "categoria": d.metadata.get("categoria", "desconocida"),
            }
            for d in docs
        ]


rag_system = RAGSystem(retriever_hibrido, k=5)