from fdua_competition.parser import get_output_parser


def test_get_output_parser_valid_json():
    parser = get_output_parser()

    valid_json = """{
        "query": "some question",
        "response": "some answer",
        "reason": "some reasoning",
        "organization_name": "test org"
    }"""

    result = parser.parse(valid_json)
    assert isinstance(result, dict)
    assert result["query"] == "some question"
    assert result["response"] == "some answer"
    assert result["reason"] == "some reasoning"
    assert result["organization_name"] == "test org"
