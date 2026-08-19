import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class Query:

    def __init__(self):
        load_dotenv()

        # LangSmith
        langsmith_key = os.getenv("langsmith_apikey")
        if langsmith_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = langsmith_key
            os.environ["LANGSMITH_PROJECT"] = "UniFit"

        # Embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        print("Embedding model loaded.")

        # Chroma
        self.db = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embedding_model
        )
        print("Chroma database initialized.")

        self.college_name = "Dublin City University - DCU"

        # Vector retriever
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {
                    "college": self.college_name
                }
            }
        )

        # Loading same chunks used in embedding for BM25
        self.chunks = self._load_chunks_from_chroma()

        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(
            [
                doc for doc in self.chunks
                if doc.metadata.get("college")
                == self.college_name
            ]
        )
        self.bm25_retriever.k = 5

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            api_key=os.getenv("api_key"),
            temperature=0
        )

    def _load_chunks_from_chroma(self):
        data = self.db.get(include=["documents", "metadatas"])
        return [
            Document(page_content=content, metadata=meta)
            for content, meta in zip(data["documents"], data["metadatas"])
        ]


    # ---------- VECTOR SEARCH ----------

    def similarity_search(self, user_query):
        return self.retriever.invoke(user_query)

    # ---------- BM25 SEARCH ----------

    def bm25_search(self, user_query):
        return self.bm25_retriever.invoke(user_query)

    def hybrid_search(self, user_query, k = 5):
        vector_results = self.similarity_search(user_query)
        bm25_results = self.bm25_search(user_query)

        scores = {}
        documents = {}

        for rank, doc in enumerate(vector_results, start = 1):
            key = (doc.metadata.get("source"), doc.page_content)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            documents[key] = doc

        for rank, doc in enumerate(bm25_results, start = 1):
            key = (doc.metadata.get("source"), doc.page_content)
            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            documents[key] = doc

        ranked_keys = sorted(
            scores,
            key = scores.get,
            reverse=True
        )

        return [documents[key] for key in ranked_keys[:k]]

    # ---------- MULTI-QUERY SEARCH ----------

    def multi_query_search(self, user_query):

        query_rewriting_prompt = ChatPromptTemplate.from_template(
            """Generate 3 alternative versions of the question for retrieval.
            Preserve the original intent.
            The documents contain information about university programs.

            Question:
            {question}

            Return exactly one query per line."""
        )

        generate_queries = (
            query_rewriting_prompt
            | self.llm
            | StrOutputParser()
            | (lambda x: [q.strip() for q in x.splitlines() if q.strip()])
        )

        retrieval_chain = (
            generate_queries
            | self.retriever.map()
            | self.get_unique_union
        )

        return retrieval_chain.invoke({
            "question": user_query
        })

    # ---------- HELPERS ----------

    def get_unique_union(self, documents):
        unique_docs = {}

        for sublist in documents:
            for doc in sublist:
                key = (
                    doc.metadata.get("source"),
                    doc.page_content
                )
                unique_docs[key] = doc

        return list(unique_docs.values())

    def format_docs(self, docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    # ---------- ANSWERING ----------

    def answer_query(self, user_query, strategy="similarity"):

        if strategy == "similarity":
            retrieved_docs = self.similarity_search(user_query)

        elif strategy == "bm25":
            retrieved_docs = self.bm25_search(user_query)

        elif strategy == "multi_query":
            retrieved_docs = self.multi_query_search(user_query)

        elif strategy == "hybrid_search":
            retrieved_docs = self.hybrid_search(user_query)

        else:
            raise ValueError(
                f"Unknown retrieval strategy: {strategy}"
            )

        context = self.format_docs(retrieved_docs)

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
            Answer the question using only the provided context.

            If the context does not contain enough information to answer
            the question, say that the information was not found in the
            available documents.

            Context:
            {context}

            Question:
            {question}
            """
        )

        answer_chain = prompt | self.llm | StrOutputParser()

        answer = answer_chain.invoke({
            "context": context,
            "question": user_query
        })

        return answer, retrieved_docs

    # ---------- CLI ----------

    def query(self):

        while True:
            user_query = input(
                "\nEnter your query (or type 'exit' to quit): "
            )

            if user_query.lower() == "exit":
                break

            strategy = "hybrid_search"

            answer, documents = self.answer_query(
                user_query,
                strategy=strategy
            )

            print("\nAnswer:\n")
            print(answer)

            print("\nSources:")

            sources = {
                doc.metadata.get("source", "Unknown")
                for doc in documents
            }

            for source in sources:
                print("-", source)


if __name__ == "__main__":
    obj = Query()
    obj.query()