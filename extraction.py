import os
from pathlib import Path
from docling.document_converter import DocumentConverter

os.environ["TORCHDYNAMO_DISABLE"] = "1"

class DocumentExtractor:

    def __init__(self):
        self.converter = DocumentConverter()

    def extract(self, source: str, file_name: str, folder: str):
        try:
            source = source.strip(' "\'')
            output_dir = Path(__file__).parent / "data" / folder
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{file_name}.md"

            result = self.converter.convert(source).document
            doc = result.export_to_markdown()

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(doc)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Extraction failed for {source}: {e}")


if __name__ == "__main__":
    obj = DocumentExtractor()

    source = input("Please enter source: ")
    file_name = input("Please enter file name: ")
    folder_name = input("Enter folder name: ")
    obj.extract(source, file_name, folder_name)