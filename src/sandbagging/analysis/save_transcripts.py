# ABOUTME: Extracts and saves agent run transcripts from Inspect AI .eval files.
# ABOUTME: Run with: python save_transcripts.py [logs_dir] [output_dir]

import json
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log


def format_content(content):
    """Convert message content to a serializable format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        result = []
        for item in content:
            if isinstance(item, str):
                result.append(item)
            elif hasattr(item, "type"):
                # Handle ContentReasoning, ContentText, etc.
                item_dict = {"type": item.type}
                if hasattr(item, "reasoning"):
                    item_dict["reasoning"] = item.reasoning
                if hasattr(item, "text"):
                    item_dict["text"] = item.text
                result.append(item_dict)
            else:
                result.append(str(item))
        return result
    return str(content)


def format_message(msg) -> dict:
    """Convert a message to a serializable dict."""
    result = {
        "role": msg.role,
        "content": format_content(msg.content),
    }

    # Include tool calls for assistant messages
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "function": tc.function,
                "arguments": tc.arguments,
            }
            for tc in msg.tool_calls
        ]

    # Include tool_call_id for tool responses
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        result["tool_call_id"] = msg.tool_call_id

    return result


def save_transcripts(logs_dir: str = "logs/", output_dir: str = "transcripts/"):
    """Extract and save transcripts from Inspect AI eval files."""
    logs_dir = Path(logs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs = list(list_eval_logs(str(logs_dir)))

    if not logs:
        print("No eval logs found.")
        return

    for log_info in logs:
        log = read_eval_log(log_info.name)

        if not log.samples:
            continue

        # Create a filename from the log
        run_id = log.eval.run_id
        task = log.eval.task
        model = log.eval.model.replace("/", "_")

        for i, sample in enumerate(log.samples):
            transcript = {
                "run_id": run_id,
                "task": task,
                "model": model,
                "status": log.status,
                "sample_id": sample.id,
                "input": sample.input,
                "target": sample.target,
                "task_args": log.eval.task_args,
                "messages": [format_message(m) for m in sample.messages],
            }

            # Add scores if available
            if sample.scores:
                transcript["scores"] = {
                    name: score.value for name, score in sample.scores.items()
                }

            # Add output/completion
            if sample.output:
                transcript["output"] = sample.output.completion

            filename = f"{task}_{model}_{run_id[:8]}_sample{i}.json"
            filepath = output_dir / filename

            with open(filepath, "w") as f:
                json.dump(transcript, f, indent=2)

            print(f"Saved: {filepath}")


if __name__ == "__main__":
    logs_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "transcripts/"
    save_transcripts(logs_dir, output_dir)
