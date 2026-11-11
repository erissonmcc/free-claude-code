"""Deterministic before/after estimate for one long conversation fixture."""

import asyncio
import json

from free_claude_code.application.context_runtime import RequestContextRuntime
from free_claude_code.config.settings import Settings
from free_claude_code.core.token_estimation import estimate_text_tokens


SUMMARY = json.dumps(
    {
        "current_task": "Continue the sample maintenance task",
        "requirements": ["Keep existing behavior"],
        "preferences": [],
        "constraints": [],
        "decisions": ["Use the existing adapter"],
        "files": ["src/example.py"],
        "changes": [],
        "errors_and_fixes": [],
        "pending_work": ["Finish validation"],
        "tool_result_references": [],
    }
)


class SummaryProvider:
    def stream_messages(self, request, **_kwargs):
        async def stream():
            escaped = json.dumps(SUMMARY)
            yield (
                'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n'
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":'
                + escaped
                + '}}\n\n'
                'data: {"type":"content_block_stop","index":0}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":32}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            )

        return stream()


async def main() -> None:
    async def resolver(_provider):
        return SummaryProvider()

    body = {
        "model": "benchmark-model",
        "messages": [
            {"role": "user", "content": f"Old decision {index}: " + ("context detail " * 180)}
            for index in range(24)
        ]
        + [
            {"role": "user", "content": f"Recent turn {index}: continue."}
            for index in range(6)
        ],
    }
    runtime = RequestContextRuntime(
        Settings(
            context_max_input_tokens=800,
            context_min_recent_messages=6,
            context_summary_target_tokens=256,
        ),
        resolver,
        request_id="context-benchmark",
        request_headers={"authorization": "Bearer benchmark"},
        model_info_lookup=lambda *_args: None,
    )
    optimized = await runtime.prepare("chat", body, "benchmark")
    before = estimate_text_tokens(json.dumps(body, ensure_ascii=False))
    after = estimate_text_tokens(json.dumps(optimized, ensure_ascii=False))
    saved = before - after
    print(f"estimated_input_before={before}")
    print(f"estimated_input_after={after}")
    print(f"estimated_input_saved={saved}")
    print(f"estimated_input_reduction_percent={saved / before * 100:.1f}")
    print(f"messages_before={len(body['messages'])}")
    print(f"messages_after={len(optimized['messages'])}")
    if saved <= 0:
        raise SystemExit("context benchmark did not reduce the estimate")


if __name__ == "__main__":
    asyncio.run(main())
