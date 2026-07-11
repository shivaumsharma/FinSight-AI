from app.planner import Planner
from app.tools.tool_registry import ToolRegistry


class ResearchAgent:
    """
    Orchestrates the execution of tools.

    It does NOT contain financial logic.
    It simply asks the planner what to do,
    executes the tools in order,
    and returns all results.
    """

    def __init__(self):

        self.planner = Planner()
        self.registry = ToolRegistry()

    def run(self, question: str, **kwargs):
        kwargs["question"]=question
        plan = self.planner.create_plan(question)
        print("QUESTION:",question)
        print("PLAN:",plan)
        results = {}
        for tool_name in plan:
            tool = self.registry.get(tool_name)
            if tool is None:
                continue
            # results[tool_name] = tool.run(**kwargs)
            print("=" * 60)
            print("Running:", tool_name)
            print(kwargs)
            print("=" * 60)
            results[tool_name] = tool.run(**kwargs)
        return results