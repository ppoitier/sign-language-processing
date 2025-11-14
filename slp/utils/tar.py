import orjson


def load_tar_index(tar_index_path: str) -> dict[str, tuple[int, int]]:
    with open(tar_index_path, "rb") as f:
        return orjson.loads(f.read())


def load_bytes_from_tar(tar_path: str, offset: int, size: int) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    return data