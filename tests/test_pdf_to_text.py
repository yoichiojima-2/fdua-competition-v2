from fdua_competition.pdf_to_text import pdf_to_text
from fdua_competition.utils import get_interim_dir


def test_pdf_to_text():
    target = get_interim_dir() / "1_0.txt"
    target.unlink(missing_ok=True)
    pdf_to_text("1.pdf")
    assert target.exists()
