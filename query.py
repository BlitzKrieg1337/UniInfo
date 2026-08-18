from operator import itemgetter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.load import dumps, loads
from langchain_google_genai import ChatGoogleGenerativeAI


class Query:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model loaded.")
        self.db = Chroma( persist_directory = "chroma_db", embedding_function = self.embedding_model)
        print("Chroma database initialized.")
        self.retriever = self.db.as_retriever(
            search_type = "similarity",
            search_kwargs={
                "k": 5,
                "filter": {
                    "college": "University College Dublin - UCD"
                }
            }
        )
        self.llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")


    def get_unique_union(self, documents: list[list]):

        flattened_doc = [dumps(doc) for sublist in documents for doc in sublist]
        unique_doc = list(set(flattened_doc))

        return [loads(doc) for doc in unique_doc]

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self):

        while True:
            user_query = input("Enter your query (or type 'exit' to quit): ")
            if user_query.lower() == 'exit':
                break

            rewritten_queries = self.query_rewriting(user_query)



            # try:
            #     results = self.retriever.invoke(user_query)
            #     print(f"Top {len(results)} results:")
            #     for i, result in enumerate(results):
            #         print(f"\nDocument {i + 1}. \n{result.page_content}... \nSource: {result.metadata.get('source', 'Unknown')}")

            # except Exception as e:
            #     print(f"Query failed: {e}")


    def query_rewriting(self, user_query):
        
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

        self.retrieval_chain = generate_queries | self.retriever.map() | self.get_unique_union()

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant.
        Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {question}
        """)

        final_rag_chain = (
            {"context" : self.retrieval_chain | self.format_docs,
            "question" : itemgetter("question")}
            | prompt
            | self.llm
            | StrOutputParser()
)

        return final_rag_chain.invoke({"question": user_query})


    def get_retrieved_documents(self, rewritten_queries):
        pass


if __name__ == "__main__":
    obj = Query()
    obj.query()