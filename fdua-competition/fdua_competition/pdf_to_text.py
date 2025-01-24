from pathlib import Path
import pypdfium2 as pdfium
from tqdm import tqdm
from fdua_competition.utils import get_interim_dir


def get_documents_dir() -> Path:
    return Path().home() / ".fdua-competition/downloads/documents"


def pdf_to_text(filename: str) -> None:
    input_path = get_documents_dir() / filename
    pdf = pdfium.PdfDocument(input_path)

    for index, page in tqdm(enumerate(pdf)):
        output_path = get_interim_dir() / f"{input_path.stem}_{index}.txt"
        text = page.get_textpage().get_text_bounded().replace("\n", "")
        output_path.write_text(text)

    print("[pdf_to_text] done")
    print(f"source: {input_path}\noutput: {output_path}")


def main():
    for pdf_file in tqdm(get_documents_dir().glob("*.pdf")):
        pdf_to_text(pdf_file.name)


if __name__ == "__main__":
    main()
