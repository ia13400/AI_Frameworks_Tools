import json
from io import BytesIO
from pathlib import Path


def write_bytes_if_changed(path: str | Path, content: bytes) -> bool:
    """Write bytes only when the target file content changes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.read_bytes() == content:
        print(f"Unchanged file: {path.name}")
        return False

    path.write_bytes(content)
    print(f"Saved file: {path.name}")
    return True


def write_text_if_changed(path: str | Path, content: str, encoding: str = "utf-8") -> bool:
    """Write text only when the target file content changes."""
    return write_bytes_if_changed(path, content.encode(encoding))


def write_json_if_changed(path: str | Path, payload) -> bool:
    """Write formatted JSON only when the target file content changes."""
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return write_text_if_changed(path, content, encoding="utf-8")


def save_figure_if_changed(fig, path: str | Path, **savefig_kwargs) -> bool:
    """Save a matplotlib figure only when the rendered PNG bytes change."""
    buffer = BytesIO()
    fig.savefig(buffer, **savefig_kwargs)
    return write_bytes_if_changed(path, buffer.getvalue())
