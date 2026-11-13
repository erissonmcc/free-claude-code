import asyncio
import uuid

import httpx
import pytest
import pytest_asyncio

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.application.code_sessions.attachments import SessionAttachments
from free_claude_code.application.errors import ApplicationUnavailableError
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from tests.api.support import create_test_app
from tests.code_sessions_support import CodexPackets, FakeHarness, close_code_database


@pytest_asyncio.fixture
async def code_api(database_factory, tmp_path):
    harness = FakeHarness()
    code = CodeService(
        SQLiteCodeStore(database_factory(tmp_path / "code.db", tmp_path / "code.lock")),
        harness,
        attachments=SessionAttachments(tmp_path / "attachments"),
    )
    await code.start()
    app = create_test_app(code=code)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            yield client, code, harness, tmp_path, app
    finally:
        await code.close()
        await close_code_database(code)
        await app.state.services.admin.close()


async def create_session(code_api):
    client, _, _, folder, _ = code_api
    response = await client.post(
        "/admin/api/code/sessions",
        json={"session_id": str(uuid.uuid4()), "cwd": str(folder), "harness": "codex"},
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.asyncio
async def test_running_claude_message_is_idempotent_after_run_finishes(code_api):
    client, code, harness, folder, _ = code_api
    code._harnesses["claude"] = harness
    session_id, run_id, operation_id = (str(uuid.uuid4()) for _ in range(3))
    created = await client.post("/admin/api/code/sessions", json={"session_id": session_id, "cwd": str(folder), "harness": "claude"})
    assert created.status_code == 201
    await code.send(session_id, run_id, created.json()["revision"], "initial", expected_epoch=code.epoch)
    await asyncio.wait_for(harness.started.wait(), 2)
    connection = harness.connections[0]
    messages = []

    async def send_message(message_id, text):
        messages.append((message_id, text))

    connection.send_message = send_message
    path = f"/admin/api/code/sessions/{session_id}/runs/{run_id}/messages"
    payload = {"operation_id": operation_id, "text": "follow-up", "expected_epoch": code.epoch}
    response = await client.post(path, json=payload)
    assert response.status_code == 200
    assert response.json() == {"operation_id": operation_id, "run_id": run_id, "status": "forwarded"}
    await connection.finish("turn-1")
    duplicate = await client.post(path, json={**payload, "expected_epoch": "old"})
    assert duplicate.json() == response.json()
    assert messages == [(operation_id, "follow-up")]
    conflict = await client.post(path, json={**payload, "text": "different"})
    assert conflict.status_code == 409
    detail = await client.get(f"/admin/api/code/sessions/{session_id}")
    assert [(item["kind"], item["text"]) for item in detail.json()["items"]] == [("user", "initial"), ("user", "follow-up")]
    assert detail.json()["items"][-1]["detail"] == "forwarded"


@pytest.mark.asyncio
async def test_message_ambiguous_result_is_durable_and_never_retried(code_api):
    client, code, harness, folder, _ = code_api
    code._harnesses["claude"] = harness
    session_id, run_id, operation_id = (str(uuid.uuid4()) for _ in range(3))
    session = await code.create_session(session_id, str(folder), "claude")
    await code.send(session_id, run_id, session.revision, "initial", expected_epoch=code.epoch)
    await asyncio.wait_for(harness.started.wait(), 2)
    calls = []
    async def ambiguous(message_id, text):
        calls.append((message_id, text))
        raise OSError("outcome unknown")
    harness.connections[0].send_message = ambiguous
    path = f"/admin/api/code/sessions/{session_id}/runs/{run_id}/messages"
    payload = {"operation_id": operation_id, "text": "hello", "expected_epoch": code.epoch}
    result = await client.post(path, json=payload)
    assert result.json() == {"operation_id": operation_id, "run_id": run_id, "status": "uncertain"}
    await harness.connections[0].finish("turn-1")
    await code.close()
    restarted = CodeService(SQLiteCodeStore(code._store.database), {"claude": harness})
    await restarted.start()
    try:
        receipt = await restarted.send_message(session_id, run_id, operation_id, "hello", expected_epoch="old")
        assert receipt == result.json()
        detail = await restarted.get_detail(session_id)
        assert detail.items[-1].text == "hello"
        assert detail.items[-1].detail == "uncertain"
        assert calls == [(operation_id, "hello")]
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_message_serializes_with_stop(code_api):
    _, code, harness, folder, _ = code_api
    code._harnesses["claude"] = harness
    session_id, run_id, operation_id = (str(uuid.uuid4()) for _ in range(3))
    session = await code.create_session(session_id, str(folder), "claude")
    await code.send(session_id, run_id, session.revision, "initial", expected_epoch=code.epoch)
    await asyncio.wait_for(harness.started.wait(), 2)
    entered, release = asyncio.Event(), asyncio.Event()
    async def blocked(*args):
        entered.set()
        await release.wait()
    harness.connections[0].send_message = blocked
    sending = asyncio.create_task(code.send_message(session_id, run_id, operation_id, "hello", expected_epoch=code.epoch))
    await asyncio.wait_for(entered.wait(), 2)
    pending = await code._store.get_item(session_id, operation_id)
    assert pending is not None and pending.text == "hello" and pending.detail == "uncertain"
    assert (await asyncio.wait_for(code.stop(session_id, run_id), 1)).status == "stopping"
    release.set()
    assert (await sending)["status"] == "uncertain"


@pytest.mark.asyncio
async def test_multiple_messages_reconcile_independently(code_api):
    _, code, harness, folder, _ = code_api
    code._harnesses['claude'] = harness
    session_id, run_id, first_id, second_id = (str(uuid.uuid4()) for _ in range(4))
    session = await code.create_session(session_id, str(folder), 'claude')
    await code.send(session_id, run_id, session.revision, 'initial', expected_epoch=code.epoch)
    await asyncio.wait_for(harness.started.wait(), 2)
    entered = asyncio.Event()
    release = asyncio.Event()
    async def deliver(message_id, text):
        if message_id == first_id:
            entered.set()
            await release.wait()
            raise OSError('unconfirmed')
    harness.connections[0].send_message = deliver
    first = asyncio.create_task(code.send_message(session_id, run_id, first_id, 'one', expected_epoch=code.epoch))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = await asyncio.wait_for(code.send_message(session_id, run_id, second_id, 'two', expected_epoch=code.epoch), 2)
        assert second['status'] == 'forwarded'
    finally:
        release.set()
    assert (await first)['status'] == 'uncertain'
    detail = await code.get_detail(session_id)
    assert [(item.text, item.detail) for item in detail.items[-2:]] == [('one', 'uncertain'), ('two', 'forwarded')]


@pytest.mark.asyncio
async def test_consecutive_assistant_messages_persist_and_return_as_separate_items(
    code_api,
):
    client, code, harness, folder, _ = code_api
    code._harnesses["claude"] = harness
    session = await code.create_session(str(uuid.uuid4()), str(folder), "claude")
    run_id = str(uuid.uuid4())
    await code.send(
        session.id,
        run_id,
        session.revision,
        "prompt",
        expected_epoch=code.epoch,
    )
    await asyncio.wait_for(harness.started.wait(), 2)
    connection = harness.connections[0]
    await connection.text(
        "turn-1", "text:message-one:0", "Intermediate", complete=True
    )
    await connection.text(
        "turn-1", "text:message-two:0", "Final answer", complete=True
    )

    stored = await code._store.run_items(session.id, run_id)
    assistant_items = [item for item in stored if item.kind == "text"]
    assert [item.text for item in assistant_items] == ["Intermediate", "Final answer"]
    assert len({item.native_item_id for item in assistant_items}) == 2

    response = await client.get(f"/admin/api/code/sessions/{session.id}")
    assert response.status_code == 200
    returned_items = [
        item for item in response.json()["items"] if item["kind"] == "text"
    ]
    assert [item["text"] for item in returned_items] == [
        "Intermediate",
        "Final answer",
    ]


@pytest.mark.asyncio
async def test_message_rejects_missing_bridge_and_non_claude(code_api):
    client, code, harness, folder, _ = code_api
    session = await create_session(code_api)
    run_id = str(uuid.uuid4())
    await code.send(session["id"], run_id, session["revision"], "initial", expected_epoch=code.epoch)
    await asyncio.wait_for(harness.started.wait(), 2)
    path = f"/admin/api/code/sessions/{session['id']}/runs/{run_id}/messages"
    payload = {"operation_id": str(uuid.uuid4()), "text": "hello", "expected_epoch": code.epoch}
    assert (await client.post(path, json=payload)).status_code == 409
    assert len((await code.get_detail(session["id"])).items) == 1
    assert harness.turn_count == 1


@pytest.mark.asyncio
async def test_message_serializes_with_completion_and_concurrent_duplicate(code_api):
    _, code, harness, folder, _ = code_api
    code._harnesses["claude"] = harness
    session_id, run_id, operation_id = (str(uuid.uuid4()) for _ in range(3))
    session = await code.create_session(session_id, str(folder), "claude")
    await code.send(session_id, run_id, session.revision, "initial", expected_epoch=code.epoch)
    await asyncio.wait_for(harness.started.wait(), 2)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def blocked(message_id, text):
        calls.append((message_id, text))
        entered.set()
        await release.wait()
    connection = harness.connections[0]
    connection.send_message = blocked
    async def send():
        return await code.send_message(session_id, run_id, operation_id, "hello", expected_epoch=code.epoch)
    first = asyncio.create_task(send())
    await asyncio.wait_for(entered.wait(), 2)
    duplicate = asyncio.create_task(send())
    finish = asyncio.create_task(connection.finish("turn-1"))
    await asyncio.wait_for(finish, 2)
    assert await duplicate == {"operation_id": operation_id, "run_id": run_id, "status": "uncertain"}
    release.set()
    assert await first == {"operation_id": operation_id, "run_id": run_id, "status": "uncertain"}
    assert calls == [(operation_id, "hello")]


@pytest.mark.asyncio
async def test_claude_message_rejects_missing_bridge_stale_epoch_and_finished_run(code_api):
    client, code, harness, folder, _ = code_api
    code._harnesses["claude"] = harness
    session_id, run_id = str(uuid.uuid4()), str(uuid.uuid4())
    session = await code.create_session(session_id, str(folder), "claude")
    await code.send(session_id, run_id, session.revision, "initial", expected_epoch=code.epoch)
    await asyncio.wait_for(harness.started.wait(), 2)
    connection = harness.connections[0]
    path = f"/admin/api/code/sessions/{session_id}/runs/{run_id}/messages"
    payload = {"operation_id": str(uuid.uuid4()), "text": "hello", "expected_epoch": code.epoch}
    assert (await client.post(path, json=payload)).status_code == 409
    sent = []
    async def send_message(text):
        sent.append(text)
    connection.send_message = send_message
    assert (await client.post(path, json={**payload, "expected_epoch": "old"})).status_code == 409
    await connection.finish("turn-1")
    assert (await client.post(path, json=payload)).status_code == 409
    assert sent == []
    assert len((await code.get_detail(session_id)).items) == 1
    assert harness.turn_count == 1


@pytest.mark.asyncio
async def test_attachment_upload_returns_session_scoped_id_without_storage_path(code_api):
    client, _, _, _, _ = code_api
    session = await create_session(code_api)

    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/attachments",
        files={"file": ("../../private note.txt", b"read me", "text/plain")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "private note.txt"
    assert payload["size"] == len(b"read me")
    assert payload["content_type"] == "text/plain"
    assert uuid.UUID(payload["attachment_id"]).version == 4
    assert not any("path" in key for key in payload)


@pytest.mark.asyncio
async def test_attachment_upload_rejects_empty_file(code_api):
    client, _, _, _, _ = code_api
    session = await create_session(code_api)

    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/attachments",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_attachment_upload_rejects_file_over_limit(code_api):
    client, _, _, _, _ = code_api
    session = await create_session(code_api)

    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/attachments",
        files={"file": ("large.txt", b"x" * (10 * 1024 * 1024 + 1), "text/plain")},
    )

    assert response.status_code == 413
    assert not list((code_api[3] / "attachments" / session["id"]).glob("*.bin"))


@pytest.mark.asyncio
async def test_attachment_rejects_large_stream_before_consuming_entire_body(code_api):
    client, _, _, _, _ = code_api
    session = await create_session(code_api)
    chunks_read = 0

    async def body():
        nonlocal chunks_read
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="large"\r\n\r\n'
        for _ in range(200):
            chunks_read += 1
            yield b"x" * (128 * 1024)
        yield b"\r\n--boundary--\r\n"

    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/attachments",
        headers={"Content-Type": "multipart/form-data; boundary=boundary"},
        content=body(),
    )
    assert response.status_code == 413
    assert chunks_read < 200


@pytest.mark.asyncio
async def test_attachment_upload_enforces_session_total_without_partial_file(code_api, monkeypatch):
    from free_claude_code.application.code_sessions import attachments

    monkeypatch.setattr(attachments, "MAX_SESSION_SIZE", 8)
    client, _, _, folder, _ = code_api
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}/attachments"
    first = await client.post(path, files={"file": ("first.txt", b"12345", "text/plain")})
    second = await client.post(path, files={"file": ("second.txt", b"6789", "text/plain")})

    assert first.status_code == 201
    assert second.status_code == 413
    files = list((folder / "attachments" / session["id"]).glob("*.bin"))
    assert len(files) == 1
    assert files[0].read_bytes() == b"12345"


@pytest.mark.asyncio
async def test_attachment_upload_rejects_symlinked_session_directory(code_api):
    client, _, _, folder, _ = code_api
    session = await create_session(code_api)
    root = folder / "attachments"
    root.mkdir()
    outside = folder / "outside"
    outside.mkdir()
    (root / session["id"]).symlink_to(outside, target_is_directory=True)

    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/attachments",
        files={"file": ("note.txt", b"private", "text/plain")},
    )

    assert response.status_code == 400
    assert list(outside.iterdir()) == []


@pytest.mark.asyncio
async def test_turn_rejects_more_than_five_attachments(code_api):
    client, code, harness, _, _ = code_api
    session = await create_session(code_api)
    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/turns",
        json={"operation_id": str(uuid.uuid4()), "expected_revision": session["revision"],
              "expected_epoch": code.epoch, "text": "Read", "attachment_ids": [str(uuid.uuid4()) for _ in range(6)]},
    )
    assert response.status_code == 422
    assert harness.connections == []


@pytest.mark.asyncio
async def test_turn_rejects_attachment_uploaded_to_another_session(code_api):
    client, _, harness, _, _ = code_api
    source = await create_session(code_api)
    target = await create_session(code_api)
    uploaded = await client.post(
        f"/admin/api/code/sessions/{source['id']}/attachments",
        files={"file": ("note.txt", b"private", "text/plain")},
    )

    response = await client.post(
        f"/admin/api/code/sessions/{target['id']}/turns",
        json={
            "operation_id": str(uuid.uuid4()),
            "expected_revision": target["revision"],
            "expected_epoch": code_api[1].epoch,
            "text": "Read this",
            "attachment_ids": [uploaded.json()["attachment_id"]],
        },
    )

    assert response.status_code == 400
    assert harness.connections == []


@pytest.mark.asyncio
async def test_turn_submits_only_server_resolved_attachment_paths(code_api):
    client, _, harness, folder, _ = code_api
    session = await create_session(code_api)
    uploaded = await client.post(
        f"/admin/api/code/sessions/{session['id']}/attachments",
        files={"file": ("note.txt", b"private", "text/plain")},
    )
    attachment_id = uploaded.json()["attachment_id"]

    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/turns",
        json={
            "operation_id": str(uuid.uuid4()),
            "expected_revision": session["revision"],
            "expected_epoch": code_api[1].epoch,
            "text": "Read this",
            "attachment_ids": [attachment_id],
        },
    )

    assert response.status_code == 202
    await harness.started.wait()
    prompt = harness.connections[0].inputs[0][1]
    assert prompt.startswith("Read this\n\nAttachments:\n")
    assert f"{folder}/attachments/{session['id']}/{attachment_id}.bin" in prompt
    assert "note.txt" not in prompt
    detail = await client.get(f"/admin/api/code/sessions/{session['id']}")
    user_items = [item for item in detail.json()["items"] if item["kind"] == "user"]
    assert user_items[0]["text"] == "Read this"


@pytest.mark.asyncio
@pytest.mark.parametrize(("mime", "suffix"), [("image/jpeg", ".jpg"), ("image/png", ".png")])
async def test_image_attachment_is_submitted_with_readable_extension(code_api, mime, suffix):
    client, code, harness, folder, _ = code_api
    session = await create_session(code_api)
    uploaded = await client.post(
        f"/admin/api/code/sessions/{session['id']}/attachments",
        files={"file": ("../../image", b"image contents", mime)},
    )
    attachment_id = uploaded.json()["attachment_id"]
    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/turns",
        json={"operation_id": str(uuid.uuid4()), "expected_revision": session["revision"],
              "expected_epoch": code.epoch, "text": "Read this image", "attachment_ids": [attachment_id]},
    )
    assert response.status_code == 202
    await harness.started.wait()
    path = folder / "attachments" / session["id"] / f"{attachment_id}{suffix}"
    assert str(path) in harness.connections[0].inputs[0][1]
    assert path.read_bytes() == b"image contents"


@pytest.mark.asyncio
async def test_image_attachments_count_toward_quota_and_are_deleted(code_api, monkeypatch):
    from free_claude_code.application.code_sessions import attachments

    monkeypatch.setattr(attachments, "MAX_SESSION_SIZE", 8)
    client, code, _, folder, _ = code_api
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}/attachments"
    first = await client.post(path, files={"file": ("photo.jpg", b"12345", "image/jpeg")})
    second = await client.post(path, files={"file": ("photo.png", b"6789", "image/png")})
    assert first.status_code == 201
    assert second.status_code == 413
    assert len(list((folder / "attachments" / session["id"]).iterdir())) == 1
    response = await client.delete(
        f"/admin/api/code/sessions/{session['id']}", params={"expected_revision": session["revision"]},
    )
    await code.wait_idle(session["id"])
    assert response.status_code == 202
    assert not (folder / "attachments" / session["id"]).exists()


@pytest.mark.asyncio
async def test_session_deletion_removes_its_attachments(code_api):
    client, code, _, folder, _ = code_api
    session = await create_session(code_api)
    await client.post(
        f"/admin/api/code/sessions/{session['id']}/attachments",
        files={"file": ("note.txt", b"private", "text/plain")},
    )

    response = await client.delete(
        f"/admin/api/code/sessions/{session['id']}",
        params={"expected_revision": session["revision"]},
    )
    await code.wait_idle(session["id"])

    assert response.status_code == 202
    assert not (folder / "attachments" / session["id"]).exists()


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_undo_confirmed_session_deletion(code_api, monkeypatch):
    client, code, _, _, _ = code_api
    session = await create_session(code_api)
    original = code._attachments.delete_session

    async def fail_cleanup(session_id):
        raise OSError("cleanup failed")

    monkeypatch.setattr(code._attachments, "delete_session", fail_cleanup)
    response = await client.delete(
        f"/admin/api/code/sessions/{session['id']}",
        params={"expected_revision": session["revision"]},
    )
    await code.wait_idle(session["id"])

    assert response.status_code == 202
    assert await code._store.is_deleted(session["id"])
    assert session["id"] not in code._owners
    monkeypatch.setattr(code._attachments, "delete_session", original)
    retry = await client.delete(
        f"/admin/api/code/sessions/{session['id']}",
        params={"expected_revision": session["revision"]},
    )
    assert retry.status_code in (200, 202, 204)


@pytest.mark.asyncio
async def test_detail_exposes_child_review_liveness_without_native_source(code_api):
    client, code, harness, _, _ = code_api
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}"
    await code.send(
        session["id"],
        str(uuid.uuid4()),
        session["revision"],
        "Delegate",
        expected_epoch=code.epoch,
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    connection = harness.connections[0]
    packets = CodexPackets(connection)
    await packets.spawn()
    await packets.review()
    await connection.finish("turn-1")
    detail = (await client.get(path)).json()
    review = detail["items"][-1]
    assert detail.get("active_review_ids") == [review["id"]]
    assert review["kind"] == "subagent_auto_review"
    assert not {"raw", "native_turn_id", "native_item_id", "generation"} & review.keys()
    await connection.close()
    closed = (await client.get(path)).json()
    assert closed["active_review_ids"] == []
    assert closed["items"] == detail["items"]


@pytest.mark.asyncio
async def test_mode_is_saved_but_native_defaults_stay_private(code_api):
    client, code, harness, _, _ = code_api
    harness.configurations["other/vendor/model"] = "other"
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}"
    changed = await client.patch(
        path,
        json={
            "expected_revision": session["revision"],
            "mode": "auto_review",
            "model": "other/vendor/model",
        },
    )
    assert changed.status_code == 200
    assert changed.json()["mode"] == "auto_review"
    assert changed.json()["provider_id"] == "other"
    assert changed.json()["model_name"] == "vendor/model"
    for mode in (None, "plan", "never"):
        rejected = await client.patch(
            path, json={"expected_revision": changed.json()["revision"], "mode": mode}
        )
        assert rejected.status_code == 422
    await code.send(
        session["id"],
        str(uuid.uuid4()),
        changed.json()["revision"],
        "hello",
        expected_epoch=code.epoch,
    )
    await asyncio.wait_for(harness.started.wait(), 3)
    detail = (await client.get(path)).json()
    assert detail["run"]["mode"] == "auto_review"
    assert "native_permission_defaults" not in detail["session"]
    assert (await code.get_detail(session["id"])).session.native_permission_defaults
    rejected = await client.patch(
        path,
        json={
            "expected_revision": detail["session"]["revision"],
            "native_permission_defaults": {},
        },
    )
    assert rejected.status_code == 422


@pytest.mark.asyncio
async def test_code_library_lists_harnesses_and_has_no_native_work_on_open(code_api):
    client, _, harness, _, _ = code_api
    bootstrap = await client.get("/admin/api/code/bootstrap")
    assert bootstrap.json()["harnesses"] == [
        {"id": "codex", "name": "Codex"},
        {"id": "claude", "name": "Claude Code"},
    ]
    session = await create_session(code_api)
    detail = await client.get(f"/admin/api/code/sessions/{session['id']}")
    assert detail.status_code == 200
    assert detail.headers["cache-control"] == "no-store"
    assert detail.json()["items"] == []
    assert harness.connections == []
    page = await client.get(f"/admin/code/{session['id']}")
    assert page.status_code == 200


@pytest.mark.asyncio
async def test_context_usage_and_raw_provider_capacity_are_public_session_state(
    code_api,
):
    client, code, harness, _, _ = code_api
    harness.context_windows[harness.model] = 100_000
    harness.configurations["unknown/model"] = "unknown"
    bootstrap = (await client.get("/admin/api/code/bootstrap")).json()
    models = {model["id"]: model for model in bootstrap["models"]}
    assert models[harness.model]["context_window_tokens"] == 100_000
    assert models["unknown/model"]["context_window_tokens"] is None

    session = await create_session(code_api)
    assert session["context_used_tokens"] is None
    await code.send(
        session["id"],
        str(uuid.uuid4()),
        session["revision"],
        "hello",
        expected_epoch=code.epoch,
    )
    await harness.started.wait()
    await harness.connections[0].context_usage("turn-1", 12_438)

    detail = (await client.get(f"/admin/api/code/sessions/{session['id']}")).json()
    assert detail["session"]["context_used_tokens"] == 12_438
    library = (await client.get("/admin/api/code/sessions")).json()
    assert library["sessions"][0]["context_used_tokens"] == 12_438


@pytest.mark.asyncio
async def test_http_send_competition_and_safe_item_projection(code_api):
    client, code, harness, _, _ = code_api
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}"
    responses = await asyncio.gather(
        *(
            client.post(
                path + "/turns",
                json={
                    "operation_id": str(uuid.uuid4()),
                    "expected_revision": session["revision"],
                    "expected_epoch": code.epoch,
                    "text": text,
                },
            )
            for text in ("one", "two")
        )
    )
    assert sorted(response.status_code for response in responses) == [202, 409]
    await harness.started.wait()
    await harness.connections[0].text(
        "turn-1", "answer", "**safe** <script>bad()</script>", complete=True
    )
    await harness.connections[0].finish("turn-1")
    detail = (await client.get(path)).json()
    item = detail["items"][-1]
    assert "<strong>safe</strong>" in item["html"]
    assert "<script>" not in item["html"]
    assert "raw" not in item
    assert detail["run"]["status"] == "completed"


@pytest.mark.asyncio
async def test_code_commands_reject_unplanned_fields_and_other_harnesses(code_api):
    client, code, _, folder, _ = code_api
    response = await client.post(
        "/admin/api/code/sessions",
        json={"session_id": str(uuid.uuid4()), "cwd": str(folder), "harness": "cline"},
    )
    assert response.status_code == 422
    session = await create_session(code_api)
    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/turns",
        json={
            "operation_id": str(uuid.uuid4()),
            "expected_revision": session["revision"],
            "expected_epoch": code.epoch,
            "text": "hi",
            "model": "override",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers", [{"host": "remote.example"}, {"origin": "https://remote.example"}]
)
async def test_code_routes_share_admin_access_boundary(code_api, headers):
    client, _, harness, _, _ = code_api
    for path in (
        "/admin/code",
        "/admin/api/code/bootstrap",
        "/admin/api/code/sessions",
        "/admin/api/code/events",
    ):
        response = await client.get(path, headers=headers)
        assert response.status_code == 403
    assert harness.connections == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_folder_selection_only_returns_to_requesting_form(
    code_api, monkeypatch, cancelled
):
    client, _, harness, folder, app = code_api
    selected = None if cancelled else str(folder)
    hints = []

    async def pick(initial_path):
        hints.append(initial_path)
        return selected

    monkeypatch.setattr(app.state.services.admin, "pick_folder", pick)
    response = await client.post(
        "/admin/api/code/folder-picker", json={"initial_path": str(folder)}
    )
    assert response.status_code == 200
    assert response.json() == {"path": selected}
    assert response.headers["cache-control"] == "no-store"
    assert hints == [str(folder)]
    assert (await client.get("/admin/api/code/sessions")).json()["sessions"] == []
    assert harness.connections == []


@pytest.mark.asyncio
async def test_folder_picker_errors_use_admin_detail(code_api, monkeypatch):
    client, _, _, _, app = code_api

    async def pick(_initial_path):
        raise ApplicationUnavailableError("A folder picker is already open")

    monkeypatch.setattr(app.state.services.admin, "pick_folder", pick)
    response = await client.post("/admin/api/code/folder-picker", json={})
    assert response.status_code == 503
    assert response.json() == {"detail": "A folder picker is already open"}


@pytest.mark.asyncio
async def test_folder_picker_only_opens_on_an_authorized_explicit_post(
    code_api, monkeypatch
):
    client, _, _, _, app = code_api

    async def unexpected(_initial_path):
        pytest.fail("Unauthorized or passive request opened a dialog")

    monkeypatch.setattr(app.state.services.admin, "pick_folder", unexpected)
    for headers in (
        {"host": "remote.example"},
        {"origin": "https://remote.example"},
    ):
        response = await client.post(
            "/admin/api/code/folder-picker", json={}, headers=headers
        )
        assert response.status_code == 403
    for payload in ({"initial_path": "x" * 4097}, {"other": True}):
        response = await client.post("/admin/api/code/folder-picker", json=payload)
        assert response.status_code == 422
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 1234)),
        base_url="http://127.0.0.1",
    ) as remote:
        assert (
            await remote.post("/admin/api/code/folder-picker", json={})
        ).status_code == 403
    assert (await client.get("/admin/api/code/folder-picker")).status_code == 405
    assert (await client.get("/admin/code")).status_code == 200
    assert (await client.get("/admin/api/code/bootstrap")).status_code == 200


@pytest.mark.asyncio
async def test_fork_route_returns_new_session_and_validates_revision(code_api, monkeypatch):
    from unittest.mock import AsyncMock
    from free_claude_code.application.code_sessions.models import CodeSession
    client, code, _, folder, _ = code_api
    result = CodeSession(id=str(uuid.uuid4()), cwd=str(folder), model="p/m", harness="claude")
    fork = AsyncMock(return_value=result)
    monkeypatch.setattr(code, "fork_session", fork)
    payload = {"session_id":result.id, "item_id":"answer", "expected_revision":2}
    response = await client.post('/admin/api/code/sessions/source/fork', json=payload)
    assert response.status_code == 201
    assert response.json()['id'] == result.id
    fork.assert_awaited_once_with('source', result.id, 'answer', 2)
    response = await client.post('/admin/api/code/sessions/source/fork', json={**payload, 'expected_revision':0})
    assert response.status_code == 422
    assert fork.await_count == 1
