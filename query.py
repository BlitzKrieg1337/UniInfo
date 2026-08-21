import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


class Query:

    def __init__(self, college_name, program_name):
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

        self.college_name = college_name
        self.program_name = program_name

        # Vector retriever
        self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "filter": {
                    "$and": [
                        {"college": self.college_name},
                        {"program": self.program_name}
                    ]
                }
            }
        )

        # Loading same chunks used in embedding for BM25
        self.chunks = self._load_chunks_from_chroma()

        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(
            [
                doc for doc in self.chunks
                if doc.metadata.get("college") == self.college_name and doc.metadata.get("program") == self.program_name
            ]
        )
        self.bm25_retriever.k = 8

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

    def hybrid_search(self, user_query, k = 15):
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

    def answer_query(self, user_query, strategy):

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
            """You are a source-grounded university information assistant.

        Answer the user's question using ONLY the provided context.

        If the user states their own qualifications or scores, compare them explicitly against any requirement stated in the context — say clearly whether they meet, exceed, or fall short. If a requirement isn't a hard number (like a degree classification) or isn't stated in the context, say so honestly instead of guessing. Never state whether the user will be admitted — only whether they meet what is explicitly written.

        Rules:
        1. Use information that is explicitly stated or clearly supported by the context.
        2. Look through the entire context, including headings, tables, notes, lists, and programme-specific sections.
        3. Prefer information that is most directly relevant to the user's question and the programme or university being discussed.
        4. If multiple relevant pieces of information are present, combine them into a clear answer.
        5. If requirements or facts differ between programmes, categories, or circumstances, explain the distinction rather than choosing arbitrarily.
        6. Do not invent, assume, or use outside knowledge.
        7. If the context does not contain enough information to answer the question, clearly say that the available documents do not provide enough information.
        8. Do not say information is unavailable merely because it is not stated in one particular section; consider the entire provided context.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
        )

        answer_chain = prompt | self.llm | StrOutputParser()

        try:
            answer = answer_chain.invoke({
                "context": context,
                "question": user_query
            })

        except ChatGoogleGenerativeAIError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                return (
                    "Gemini API quota has been exhausted. Please try again later.",
                    retrieved_docs
                )
            raise

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