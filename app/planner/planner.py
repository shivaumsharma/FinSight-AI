from .planner_rules import TOOL_RULES


class Planner:
    """
    Rule-based planner.

    Takes a user question and determines which tools
    are required to answer it.
    """

    def __init__(self):
        self.rules = TOOL_RULES

    def create_plan(self, question: str) -> list[str]:
        """
        Returns an ordered list of tool names required
        to answer the user's question.
        """

        question = question.lower()
        selected_tools = []

        for keywords, tools in self.rules:
            if any(keyword in question for keyword in keywords):
                for tool in tools:
                    if tool not in selected_tools:
                        selected_tools.append(tool)

        return selected_tools

