
from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".csv",
    ".pptx",
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
    )
