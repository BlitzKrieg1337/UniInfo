import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

class Query:
    def __init__(self):
        # 1. Initialize Embeddings and DB
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model loaded.")
        
        self.db = Chroma(persist_directory="chroma_db", embedding_function=self.embedding_model)
        print("Chroma database initialized.")
        
        self.retriever = self.db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "filter": {
                    "college": "University College Dublin - UCD"
                }
            }
        )

        # 2. Initialize LLM 
        load_dotenv()
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", api_key = os.getenv("api_key"), temperature=0)
        
        # 3. Setup the retrieval chain immediately so it's ready for queries
        self._setup_retrieval_chain()

    def _setup_retrieval_chain(self):
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
            | (lambda x: [q for q in x.splitlines() if q.strip()])
        )

        self.retrieval_chain = generate_queries | self.retriever.map() | self.get_unique_union

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
        return "\n\n".join(doc.page_content for doc in docs)
    
    def answer_query(self, user_query):
        retrieved_docs = self.retrieval_chain.invoke({
            "question": user_query
        })

        context = self.format_docs(retrieved_docs)

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
            Answer the question using only the provided context.

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

    def query(self):
        while True:
            user_query = input("\nEnter your query (or type 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break

            answer, documents = self.answer_query(user_query)

            print("\nAnswer:\n")
            print(answer)

            print("\nSources:")
            sources = set()

            for doc in documents:
                sources.add(doc.metadata.get("source", "Unknown"))

            for source in sources:
                print("-", source)


if __name__ == "__main__":
    obj = Query()
    obj.query()