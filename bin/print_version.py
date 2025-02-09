import tomllib


def get_version():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


if __name__ == "__main__":
    print(get_version())
