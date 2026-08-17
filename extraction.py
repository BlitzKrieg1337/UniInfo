from docling.document_converter import DocumentConverter

class DocumentExtractor:

    def get_url_text(self, url:str, file_name:str):
        converter = DocumentConverter()
        try: 
            result = converter.convert(url).document
            doc = result.export_to_markdown()

            with open(f"data/URL/{file_name}.md", "w", encoding = "utf-8") as f:
                f.write(doc)
            print("Saved URL.")
        except Exception as e:
            print(f"Extraction failed: {e}")

#Add Prgoram URLs with file names(TCD, UCD, DCU)
obj = DocumentExtractor()
url = input("Please enter URl: ")
filename = input("Please enter file name: ")
obj.get_url_text(url, filename)