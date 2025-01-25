from fdua_competition.vectorize import vectorize_text


def test_vectorize_text():
    result = vectorize_text("test")
    assert all(isinstance(x, float) for x in result)
