from typing import Any

from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool


def _validate_json_output(result: TaskOutput) -> tuple[bool, Any]:
    try:
        import json
        import re

        text = str(result)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            data = json.loads(match.group(0))
            return True, data
        data = json.loads(text)
        return True, data
    except Exception as e:
        return False, f"Invalid JSON output: {e}"


def _step_logger(step: Any) -> None:
    print(f"[STEP] {step}")


def _task_logger(task_result: Any) -> None:
    try:
        print(f"[TASK] {task_result}")
    except Exception:
        print("[TASK] <unprintable>")


@CrewBase
class ResearchCrew:
    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["research_agent"],  # type: ignore[index]
            verbose=True,
            tools=[SerperDevTool()],
        )

    @agent
    def writing_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["writing_agent"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def review_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["review_agent"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def senior_review_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["senior_review_agent"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def polishing_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["polishing_agent"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
            guardrail=_validate_json_output,
            guardrail_max_retries=1,
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["writing_task"],  # type: ignore[index]
            guardrail=_validate_json_output,
            guardrail_max_retries=1,
        )

    @task
    def review_task(self) -> Task:
        return Task(
            config=self.tasks_config["review_task"],  # type: ignore[index]
            guardrail=_validate_json_output,
            guardrail_max_retries=2,
        )

    @task
    def fallback_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["fallback_review_task"],  # type: ignore[index]
            guardrail=_validate_json_output,
            guardrail_max_retries=2,
        )

    @task
    def polishing_task(self) -> Task:
        return Task(
            config=self.tasks_config["polishing_task"],  # type: ignore[index]
            output_file="result.md",
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            step_callback=_step_logger,
            task_callback=_task_logger,
            output_log_file="output/logs.json",
        )
