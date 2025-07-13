from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import PDFSearchTool
from pathlib import Path
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent
pdf_path = str(SCRIPT_DIR / "agentops.pdf")

# Use the resolved path for the PDFSearchTool
tool = PDFSearchTool(
    pdf=pdf_path,
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="mistral",  # or "mixtral", "llama3", etc.
                # temperature=0.5,
                # top_p=1,
                # stream=True,
            ),
        ),
        embedder=dict(
            provider="ollama",
            config=dict(
                model="nomic-embed-text",  # or another embedding model supported by Ollama
                # task_type can be omitted unless required by the tool
            ),
        ),
     
    )
)

@CrewBase
class PdfRag():
    """PdfRag crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def pdf_rag_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['pdf_rag_agent'], # type: ignore[index]
            tools=[tool],
            verbose=True
        )

    @agent
    def pdf_summary_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['pdf_summary_agent'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def pdf_rag_task(self) -> Task:
        return Task(
            config=self.tasks_config['pdf_rag_task'], # type: ignore[index]
        )

    @task
    def pdf_summary_task(self) -> Task:
        return Task(
            config=self.tasks_config['pdf_summary_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PdfRag crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
