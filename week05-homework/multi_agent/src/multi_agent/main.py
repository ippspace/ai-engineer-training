#!/usr/bin/env python
import os
import sys

from .crew import ResearchCrew


def _get_topic_from_args() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    return "AI Agent 的发展与应用"


def _append_error_logs(report_path: str) -> None:
    try:
        import json

        with open("output/logs.json", "r", encoding="utf-8") as f:
            logs = json.load(f)
        errors: list[str] = []
        for item in logs if isinstance(logs, list) else []:
            msg = str(item)
            if (
                "Guardrail Failed" in msg
                or "Task Failed" in msg
                or "Crew Failure" in msg
            ):
                errors.append(msg)
        if errors:
            with open(report_path, "a", encoding="utf-8") as f:
                _ = f.write("\n\n## 异常处理日志\n")
                _ = f.write("\n- 一级重试：同一代理在守卫校验失败时自动重试（最多2次）")
                _ = f.write("\n- 二级重试：切换至备用代理执行审核任务（如适用）")
                _ = f.write("\n- 三级重试：若持续失败，向用户请求补充信息\n")
                for e in errors:
                    _ = f.write(f"\n- {e}")
    except Exception:
        pass


def run():
    os.makedirs("output", exist_ok=True)
    topic = _get_topic_from_args()
    inputs = {"topic": topic}

    result = ResearchCrew().crew().kickoff(inputs=inputs)

    print("\n=== 协作完成，生成终稿 ===\n")
    print(result.raw)


if __name__ == "__main__":
    run()
