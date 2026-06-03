from __future__ import annotations

import io
import os
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

from torch.distributed.checkpoint.filesystem import (
    FileSystemBase,
    FileSystemReader,
    FileSystemWriter,
    SerializationFormat,
)
from torch.distributed.checkpoint._extension import StreamTransformExtension

from reporterv2.storage import store_delete, store_get, store_list, store_put, store_size


def _normalize_key(path: str | os.PathLike) -> str:
    return os.fspath(path).replace(os.sep, "/").strip("/")


def _prefix_key(path: str | os.PathLike) -> str:
    key = _normalize_key(path)
    return f"{key}/" if key else ""


def reporterv2_folder_exists(path: str | os.PathLike) -> bool:
    return bool(store_list(_prefix_key(path)))


def reporterv2_checkpoint_exists(path: str | os.PathLike) -> bool:
    return store_size(f"{_normalize_key(path)}/.metadata") is not None


def reporterv2_delete_checkpoint(path: str | os.PathLike) -> None:
    store_delete(_prefix_key(path), recursive=True)


def reporterv2_list_checkpoint_steps(
    path: str | os.PathLike, *, require_metadata: bool
) -> list[int]:
    prefix = _prefix_key(path)
    steps: set[int] = set()
    for key in store_list(prefix):
        relative_key = key[len(prefix) :] if key.startswith(prefix) else key
        parts = relative_key.split("/", 2)
        if len(parts) < 2 or not parts[0].startswith("step-"):
            continue
        try:
            step = int(parts[0].removeprefix("step-"))
        except ValueError:
            continue
        if require_metadata and parts[1] not in {
            ".metadata",
            "model.safetensors.index.json",
        }:
            continue
        steps.add(step)
    return sorted(steps)


class _ReporterV2WriteStream(io.BytesIO):
    def __init__(self, key: str, timeout: float) -> None:
        super().__init__()
        self._key = key
        self._timeout = timeout
        self._uploaded = False

    def close(self) -> None:
        if not self._uploaded:
            store_put(self._key, self.getvalue(), timeout=self._timeout)
            self._uploaded = True
        super().close()


class ReporterV2FileSystem(FileSystemBase):
    """DCP FileSystem adapter backed by reporterv2.storage."""

    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = timeout

    @contextmanager
    def create_stream(
        self, path: str | os.PathLike, mode: str
    ) -> Generator[io.IOBase, None, None]:
        key = _normalize_key(path)
        if "r" in mode:
            data = store_get(key)
            if data is None:
                raise FileNotFoundError(key)
            stream = io.BytesIO(data)
        elif "w" in mode:
            stream = _ReporterV2WriteStream(key, timeout=self.timeout)
        else:
            raise ValueError(f"Unsupported ReporterV2 checkpoint stream mode: {mode}")

        try:
            yield stream
        finally:
            stream.close()

    def concat_path(self, path: str | os.PathLike, suffix: str) -> str:
        prefix = _normalize_key(path)
        suffix = suffix.lstrip("/")
        return f"{prefix}/{suffix}" if prefix else suffix

    def rename(self, path: str | os.PathLike, new_path: str | os.PathLike) -> None:
        old_key = _normalize_key(path)
        data = store_get(old_key)
        if data is None:
            raise FileNotFoundError(old_key)
        store_put(_normalize_key(new_path), data, timeout=self.timeout)
        store_delete(old_key)

    def init_path(self, path: str | os.PathLike) -> str:
        return _normalize_key(path)

    def mkdir(self, path: str | os.PathLike) -> None:
        # ReporterV2/mkv stores keys, not directories. Parent prefixes are implicit.
        del path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike) -> bool:
        return bool(_normalize_key(checkpoint_id))

    def exists(self, path: str | os.PathLike) -> bool:
        key = _normalize_key(path)
        return store_size(key) is not None or bool(store_list(_prefix_key(key)))

    def rm_file(self, path: str | os.PathLike) -> None:
        store_delete(_normalize_key(path))

    def ls(self, path: str | os.PathLike) -> list[str]:
        return store_list(_prefix_key(path))


class ReporterV2StorageWriter(FileSystemWriter):
    """DCP StorageWriter that writes checkpoint files through ReporterV2."""

    def __init__(
        self,
        path: str | os.PathLike = "",
        single_file_per_rank: bool = True,
        sync_files: bool = False,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
        _extensions: Sequence[StreamTransformExtension] | None = None,
        serialization_format: SerializationFormat = SerializationFormat.TORCH_SAVE,
    ) -> None:
        super().__init__(
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            cache_staged_state_dict=cache_staged_state_dict,
            overwrite=overwrite,
            _extensions=_extensions,
            serialization_format=serialization_format,
        )
        self.fs = ReporterV2FileSystem()
        self.path = self.fs.init_path(path)


class ReporterV2StorageReader(FileSystemReader):
    """DCP StorageReader that reads checkpoint files through ReporterV2."""

    def __init__(
        self,
        path: str | os.PathLike = "",
        _extension_registry: Any | None = None,
    ) -> None:
        super().__init__(path=path, _extension_registry=_extension_registry)
        self.fs = ReporterV2FileSystem()
        self.path = self.fs.init_path(path)
