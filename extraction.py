from docling.document_converter import DocumentConverter

class DocumentExtractor:

    def get_url_text(self, url:str, file_name:str):
        converter = DocumentConverter()
        result = converter.convert(url).document
        doc = result.export_to_markdown()

        with open(f"data/URL/{file_name}.md", "w", encoding = "utf-8") as f:
            f.write(doc)
        print("Saved URL.")

c = DocumentExtractor()
url = input("Please enter URl: ")
filename = input("Please enter file name: ")
c.get_url_text(url, filename)