from pathlib import Path
import pypdfium2 as pdfium
from tqdm import tqdm


def get_documents_dir() -> Path:
    return Path().cwd().parent / "downloads/documents"


def get_interim_dir() -> Path:
    dir = Path().cwd().parent / "interim"
    dir.mkdir(exist_ok=True)
    return dir


def pdf_to_text(filename: str) -> None:
    input_path = get_documents_dir() / filename
    pdf = pdfium.PdfDocument(input_path)

    for index, page in tqdm(enumerate(pdf)):
        text = page.get_textpage().get_text_bounded().replace("\n", "")
        (get_interim_dir() / f"{input_path.stem}_{index}.txt").write_text(text)

    print(f"[pdf_to_text] done: {input_path}")


def main():
    for pdf_file in tqdm(get_documents_dir().glob("*.pdf")):
        pdf_to_text(pdf_file.name)


if __name__ == "__main__":
    main()
