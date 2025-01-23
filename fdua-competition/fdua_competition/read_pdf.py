from pathlib import Path
import pypdfium2 as pdfium
from tqdm import tqdm


def get_documents_dir():
    return Path().cwd().parent / "downloads/documents"


def get_interim_dir():
    return Path().cwd().parent / "interim"


def read_pdf(filename: str):
    input_path = get_documents_dir() / filename
    pdf = pdfium.PdfDocument(input_path)
    get_interim_dir().mkdir(exist_ok=True)

    for index, page in tqdm(enumerate(pdf)):
        text = page.get_textpage().get_text_bounded().replace("\n", "")
        (get_interim_dir() / f"{input_path.stem}_{index}.txt").write_text(text)

    print(f"[read_pdf] done reading {filename}\n")


def main():
    for pdf_file in tqdm(get_documents_dir().glob("*.pdf")):
        read_pdf(pdf_file.name)
        

if __name__ == "__main__":
    main()