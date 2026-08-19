import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


class Query:

    def __init__(self):
        load_dotenv()

        # LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("langsmith_apikey")
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

        # Base vector retriever
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {
                    "college": "Dublin City University - DCU"
                }
            }
        )

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            api_key=os.getenv("api_key"),
            temperature=0
        )

    # ---------- RETRIEVAL STRATEGY 1 ----------

    def similarity_search(self, user_query):
        return self.retriever.invoke(user_query)

    # ---------- RETRIEVAL STRATEGY 2 ----------

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

    # ---------- HELPER ----------

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
            doc.page_content
            for doc in docs
        )

    # ---------- ANSWERING ----------

    def answer_query(self, user_query, strategy="similarity"):

        if strategy == "similarity":
            retrieved_docs = self.similarity_search(user_query)

        elif strategy == "multi_query":
            retrieved_docs = self.multi_query_search(user_query)

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

            # Choose retrieval strategy here
            strategy = "similarity"

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