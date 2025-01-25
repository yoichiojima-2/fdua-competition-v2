from fdua_competition.vectorize import vectorize_text


def test_vectorize_text():
    vector = vectorize_text("test")
    assert all(isinstance(x, float) for x in vector)
