from pathlib import Path


def test_assets():
    secrets_dir = Path(__file__).parent.parent / "secrets"
    existing_items = sorted([i for i in secrets_dir.iterdir()])
    expected_items = sorted([secrets_dir / i for i in [".env", "google-application-credentials.json"]])
    assert existing_items == expected_items
