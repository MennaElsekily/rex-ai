
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".pptx",
    ".xlsx",
    ".json",
    ".html",
    ".htm",
}

# Leading bytes each binary format must start with. Text formats have no
# signature; they are checked for readable text when parsed instead.
_SIGNATURES = {
    ".pdf": b"%PDF",
    ".docx": b"PK",
    ".pptx": b"PK",
    ".xlsx": b"PK",
}


def get_file_type(file_path: str | Path) -> str:
    """Return the normalized extension of a supported file."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{extension}' for file: {path.name}"
        )

    return extension


def check_signature(file_path: str | Path) -> None:
    """
    Raise ValueError when a binary file's content does not match its
    extension (a renamed or corrupted file), so it fails clearly instead
    of producing garbage text.
    """
    path = Path(file_path)
    expected = _SIGNATURES.get(path.suffix.lower())

    if expected is None:
        return

    with path.open("rb") as file:
        head = file.read(len(expected))

    if head != expected:
        raise ValueError(
            f"{path.name} is not a valid {path.suffix.lower()} file"
        )


def discover_files(directory: str | Path) -> list[Path]:
    """Discover supported files recursively in a directory."""
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory}"
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {directory}"
        )

    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        # Office lock files ("~$catalogue.xlsx") and hidden files are
        # not documents.
        and not path.name.startswith(("~$", "."))
    )
