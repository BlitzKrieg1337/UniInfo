import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings


class Embedding:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        print("Model loaded.")


obj = Embedding()
obj.embedded()