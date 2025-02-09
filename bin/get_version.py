import tomllib


def main():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    version = data["project"]["version"]
    print(version)


main()
