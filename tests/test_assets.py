from pathlib import Path


def test_assets():
    assets_dir = Path(__file__).parent.parent / "assets"
    existing_items = sorted([i for i in assets_dir.iterdir()])
    expected_items = sorted(
        [
            assets_dir / i
            for i in [
                "query.csv",
                "validation.zip",
                "documents.zip",
                "readme.md",
                "sample_submit.zip",
                "evaluation.zip",
                ".success",
            ]
        ]
    )
    assert existing_items == expected_items
