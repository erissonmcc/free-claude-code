"""Temporary, session-scoped files supplied to coding turns."""

import asyncio
import uuid
from pathlib import Path

from free_claude_code.application.code_sessions.models import CodeValidationError

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_SESSION_SIZE = 50 * 1024 * 1024
IMAGE_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}
ATTACHMENT_EXTENSIONS = (".bin", *IMAGE_EXTENSIONS.values())


class AttachmentSizeError(Exception):
    pass


class SessionAttachments:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.home() / ".fcc" / "tmp" / "code-attachments"
        self._locks: dict[str, asyncio.Lock] = {}

    def _session_directory(self, session_id: str) -> Path:
        try:
            parsed = uuid.UUID(session_id)
            if str(parsed) != session_id or parsed.version != 4:
                raise ValueError
        except ValueError:
            raise CodeValidationError("Invalid session ID.") from None
        directory = self.root / session_id
        if directory.is_symlink():
            raise CodeValidationError("Invalid session attachment directory.")
        return directory

    async def save(self, session_id: str, upload) -> dict[str, str | int]:
        directory = self._session_directory(session_id)
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            existing = tuple(path for path in directory.iterdir() if path.suffix in ATTACHMENT_EXTENSIONS and path.is_file())
            total = sum(path.stat().st_size for path in existing)
            if total >= MAX_SESSION_SIZE:
                raise CodeValidationError(
                    "The session attachment limit has been reached."
                )
            attachment_id = str(uuid.uuid4())
            suffix = IMAGE_EXTENSIONS.get((upload.content_type or "").lower(), ".bin")
            target = directory / f"{attachment_id}{suffix}"
            size = 0
            try:
                with target.open("xb") as output:
                    while chunk := await upload.read(64 * 1024):
                        size += len(chunk)
                        if size > MAX_FILE_SIZE or total + size > MAX_SESSION_SIZE:
                            raise AttachmentSizeError(
                                "The attachment exceeds its size limit."
                            )
                        output.write(chunk)
                if size == 0:
                    raise CodeValidationError("Empty attachments are not supported.")
            except BaseException:
                target.unlink(missing_ok=True)
                if not any(directory.iterdir()):
                    directory.rmdir()
                raise
            name = (upload.filename or "attachment").replace("\\", "/").split("/")[-1]
            name = "".join(char for char in name if char.isprintable()).strip() or "attachment"
            return {
                "attachment_id": attachment_id,
                "name": name,
                "size": size,
                "content_type": upload.content_type or "application/octet-stream",
            }

    async def resolve(
        self, session_id: str, attachment_ids: list[str]
    ) -> tuple[Path, ...]:
        if len(attachment_ids) > 5 or len(set(attachment_ids)) != len(attachment_ids):
            raise CodeValidationError("Choose at most 5 distinct attachments per turn.")
        directory = self._session_directory(session_id).resolve()
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            paths = []
            for attachment_id in attachment_ids:
                try:
                    if str(uuid.UUID(attachment_id)) != attachment_id:
                        raise ValueError
                except ValueError:
                    raise CodeValidationError(
                        "An attachment is invalid or unavailable."
                    ) from None
                candidates = [directory / f"{attachment_id}{suffix}" for suffix in ATTACHMENT_EXTENSIONS]
                matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
                if len(matches) != 1 or not matches[0].is_relative_to(directory):
                    raise CodeValidationError(
                        "An attachment is invalid or unavailable."
                    )
                paths.append(matches[0])
            return tuple(paths)

    async def delete_session(self, session_id: str) -> None:
        directory = self._session_directory(session_id)
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            if directory.is_dir():
                for path in directory.iterdir():
                    if path.is_file() and path.suffix in ATTACHMENT_EXTENSIONS:
                        path.unlink()
                directory.rmdir()
