import os
from pathlib import Path 
from docling.document_converter import DocumentConverter

os.environ["TORCHDYNAMO_DISABLE"] = "1"

class DocumentExtractor:

    def extract(self, source:str, file_name:str, folder:str):
        converter = DocumentConverter()
        try: 
            source = source.strip(' "\'')
            output_dir = Path(__file__).parent / "data" / folder
            output_dir.mkdir(parents=True, exist_ok=True)

            result = converter.convert(source).document
            doc = result.export_to_markdown()

            with open(f"data/{folder}/{file_name}.md", "w", encoding = "utf-8") as f:
                f.write(doc)
            print(f"Saved: data/{folder}/{file_name}.md")
        except Exception as e:
            print(f"Extraction failed: {e}")

#Add Prgoram URLs with file names(TCD, UCD, DCU)
obj = DocumentExtractor()

source = input("Please enter source: ")
file_name = input("Please enter file name: ")
folder_name = input("Enter folder name: ")
obj.extract(source, file_name, folder_name)