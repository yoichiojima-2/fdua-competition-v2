from fdua_competition.vectorize import vectorize_text


def test_vectorize_text():
    text = "Hello, world!"
    result = vectorize_text(text)
    assert result
    assert all(isinstance(x, float) for x in result)