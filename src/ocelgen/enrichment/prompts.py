"""Meta-prompt templates for LLM-based trace enrichment."""

from __future__ import annotations


def build_enrichment_prompt(
    domain_description: str,
    pattern_description: str,
    agent_role: str,
    agent_persona: str,
    user_query: str,
    tool_names: list[str],
    tool_descriptions: dict[str, str],
    expected_llm_calls: int,
    expected_tool_calls: int,
    previous_output: str | None,
    deviation_context: str | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts for a single enrichment call.

    Returns (system_prompt, user_prompt).
    """
    system_prompt = (
        f"You are simulating an AI agent in a multi-agent workflow.\n"
        f"Domain: {domain_description}\n"
        f"Workflow pattern: {pattern_description}\n\n"
        f"Generate realistic, detailed content that would appear in a real agent trace. "
        f"Include specific data, names, numbers, and technical details — not generic placeholders. "
        f"Respond with valid JSON only."
    )

    tools_section = ""
    if tool_names:
        tool_lines = []
        for name in tool_names:
            desc = tool_descriptions.get(name, name)
            tool_lines.append(f"  - {name}: {desc}")
        tools_section = "Available tools:\n" + "\n".join(tool_lines)
    else:
        tools_section = "Available tools: none (this agent uses only LLM reasoning)"

    previous_section = ""
    if previous_output:
        previous_section = f"Previous agent output:\n{previous_output}\n"

    deviation_section = ""
    if deviation_context:
        deviation_section = f"\n{deviation_context}\n"

    user_prompt = (
        f"You are acting as the **{agent_role}** agent.\n"
        f"Persona: {agent_persona}\n\n"
        f"User query: {user_query}\n\n"
        f"{previous_section}"
        f"{deviation_section}"
        f"{tools_section}\n\n"
        f"Generate exactly {expected_llm_calls} LLM call(s) and {expected_tool_calls} tool call(s).\n\n"
        f"Respond as JSON with this exact structure:\n"
        f"{{\n"
        f'  "reasoning": "Your chain-of-thought reasoning (2-4 sentences)",\n'
        f'  "llm_calls": [\n'
        f'    {{"prompt": "The prompt sent to the LLM", "completion": "The LLM response"}}\n'
        f"  ],\n"
        f'  "tool_calls": [\n'
        f'    {{"input": {{"arg": "value"}}, "output": {{"result": "value"}}}}\n'
        f"  ],\n"
        f'  "output_to_next_agent": "Summary output passed to the next agent in the chain"\n'
        f"}}"
    )

    return system_prompt, user_prompt
