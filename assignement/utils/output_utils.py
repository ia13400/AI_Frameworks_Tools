import json
from io import BytesIO
from pathlib import Path


def normalize_text_content(content: str) -> str:
    """Normalize line endings before comparing generated text files."""
    return content.replace("\r\n", "\n").replace("\r", "\n")


def normalize_json_value(value, float_digits: int = 10):
    """Normalize JSON payloads so tiny float noise does not rewrite files."""
    if isinstance(value, dict):
        return {
            key: normalize_json_value(item, float_digits=float_digits)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            normalize_json_value(item, float_digits=float_digits)
            for item in value
        ]
    if isinstance(value, float):
        return round(value, float_digits)
    return value


def normalize_png_bytes(content: bytes) -> bytes:
    """Keep only semantically relevant PNG chunks for stable comparisons."""
    png_signature = b"\x89PNG\r\n\x1a\n"
    if not content.startswith(png_signature):
        return content

    offset = len(png_signature)
    chunks = [png_signature]
    keep_chunks = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS"}

    while offset + 12 <= len(content):
        chunk_start = offset
        chunk_length = int.from_bytes(content[offset : offset + 4], "big")
        chunk_type = content[offset + 4 : offset + 8]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(content):
            return content

        if chunk_type in keep_chunks:
            chunks.append(content[chunk_start:chunk_end])

        offset = chunk_end
        if chunk_type == b"IEND":
            break

    return b"".join(chunks)


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = path.read_text(encoding=encoding)
        if normalize_text_content(existing) == normalize_text_content(content):
            print(f"Unchanged file: {path.name}")
            return False

    return write_bytes_if_changed(path, content.encode(encoding))


def write_json_if_changed(path: str | Path, payload) -> bool:
    """Write formatted JSON only when the target file content changes."""
    path = Path(path)
    normalized_payload = normalize_json_value(payload)

    if path.exists():
        try:
            existing_payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_payload = None
        if normalize_json_value(existing_payload) == normalized_payload:
            print(f"Unchanged file: {path.name}")
            return False

    content = json.dumps(normalized_payload, indent=2, ensure_ascii=False) + "\n"
    return write_text_if_changed(path, content, encoding="utf-8")


def save_figure_if_changed(fig, path: str | Path, **savefig_kwargs) -> bool:
    """Save a matplotlib figure only when the rendered PNG bytes change."""
    path = Path(path)
    buffer = BytesIO()
    savefig_kwargs.setdefault("metadata", {})
    fig.savefig(buffer, **savefig_kwargs)
    content = buffer.getvalue()

    if path.exists():
        existing = path.read_bytes()
        if normalize_png_bytes(existing) == normalize_png_bytes(content):
            print(f"Unchanged file: {path.name}")
            return False

    return write_bytes_if_changed(path, content)
