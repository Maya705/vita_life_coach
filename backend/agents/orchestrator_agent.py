"""Orchestrator Agent (Head Coach): autonomous ReAct reasoning loop."""
import logging
import re
from typing import Any

from backend import llm_client
from backend.agents.specialists import run_specialist
from backend.rag.retrieval import get_nutrition_context, get_research_context

logger = logging.getLogger(__name__)
MODULE_NAME = "Orchestrator Agent"
MAX_ITERATIONS = 5

# Canonical specialist names for fuzzy matching
_SPECIALIST_ALIASES: dict[str, str] = {
    "nutrition expert": "Nutrition Expert",
    "nutritionexpert": "Nutrition Expert",
    "nutrition": "Nutrition Expert",
    "science researcher": "Science Researcher",
    "scienceresearcher": "Science Researcher",
    "science": "Science Researcher",
    "researcher": "Science Researcher",
    "wellness coach": "Wellness Coach",
    "wellnesscoach": "Wellness Coach",
    "wellness": "Wellness Coach",
    "coach": "Wellness Coach",
}

REACT_SYSTEM_PROMPT = """You are the Head Coach of Vita, an AI wellness and nutrition coach.

You solve the user's request by reasoning step-by-step. Each turn you MUST output exactly one block in this format:

Thought: <your reasoning about what to do next>
Action: <one of the available actions>
Action Input: <input for the action>

Available actions:
- call_specialist(Nutrition Expert, <task>) — diet, food, nutrients, meals, ingredients
- call_specialist(Science Researcher, <task>) — evidence, research, clinical trials, medical facts
- call_specialist(Wellness Coach, <task>) — stress, exercise, sleep, mindfulness, habits
- search_nutrition(<query>) — direct RAG lookup for nutrition data
- search_research(<query>) — direct RAG lookup for research/pubmed data
- finish(<response>) — return the final answer to the user

Rules:
- Always start with a Thought.
- Call specialists or search tools to gather info before finishing.
- You may call multiple specialists across turns if needed.
- When you have enough information, use finish() with your complete, friendly final response.
- Do NOT output anything after "Action Input:".
"""


def _normalize_specialist(name: str) -> str:
    """Normalize a specialist name to its canonical form."""
    key = name.strip().lower()
    if key in _SPECIALIST_ALIASES:
        return _SPECIALIST_ALIASES[key]
    # Fallback: return cleaned-up original
    return name.strip()


def _parse_react_output(text: str) -> dict[str, str]:
    """Parse Thought / Action / Action Input from LLM output.

    Returns dict with keys: thought, action, action_input.
    """
    thought = ""
    action = ""
    action_input = ""

    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    action_match = re.search(r"Action:\s*(.+?)(?=\nAction Input:|\Z)", text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()

    input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    if input_match:
        action_input = input_match.group(1).strip()

    return {"thought": thought, "action": action, "action_input": action_input}


def _execute_action(action: str, action_input: str, steps: list[dict[str, Any]]) -> str:
    """Dispatch an action to the appropriate specialist or RAG function.

    Returns the observation string.
    """
    # call_specialist(Specialist Name, task)
    cs_match = re.match(r"call_specialist\(\s*(.+?)\s*,\s*(.+?)\s*\)$", action, re.DOTALL)
    if cs_match:
        specialist_name = _normalize_specialist(cs_match.group(1))
        task = cs_match.group(2).strip()
    elif action.lower().startswith("call_specialist"):
        # Fallback: action_input contains "Specialist Name, task"
        parts = action_input.split(",", 1)
        specialist_name = _normalize_specialist(parts[0])
        task = parts[1].strip() if len(parts) > 1 else action_input
    else:
        specialist_name = None
        task = None

    if specialist_name and task:
        # Inject RAG context for relevant specialists
        context = ""
        if specialist_name == "Nutrition Expert":
            context = get_nutrition_context(task)
        elif specialist_name == "Science Researcher":
            context = get_research_context(task)
        try:
            response_text, step_dict = run_specialist(specialist_name, task, context=context)
            steps.append(step_dict)
            return response_text
        except Exception as e:
            logger.exception("Specialist %s failed", specialist_name)
            steps.append({"module": specialist_name, "prompt": {"task": task}, "response": {"error": str(e)}})
            return f"Error calling {specialist_name}: {e}"

    # search_nutrition(query)
    sn_match = re.match(r"search_nutrition\(\s*(.+?)\s*\)$", action, re.DOTALL)
    if sn_match:
        query = sn_match.group(1)
        result = get_nutrition_context(query)
        return result or "No nutrition data found."

    # search_research(query)
    sr_match = re.match(r"search_research\(\s*(.+?)\s*\)$", action, re.DOTALL)
    if sr_match:
        query = sr_match.group(1)
        result = get_research_context(query)
        return result or "No research data found."

    # finish(response) — handled in the main loop, but just in case
    if action.lower().startswith("finish"):
        return action_input

    return f"Unknown action: {action}"


def _force_finish(prompt: str, scratchpad: str, steps: list[dict[str, Any]]) -> str:
    """Force a final synthesis when max iterations are reached."""
    messages = [
        {"role": "system", "content": "You are the Head Coach of Vita, an AI wellness and nutrition coach. Synthesize everything gathered so far into one concise, friendly response."},
        {"role": "user", "content": f"User request: {prompt}\n\nResearch so far:\n{scratchpad}\n\nPlease provide your final answer now."},
    ]
    final_content, raw_response = llm_client.chat_with_raw_response(messages)
    steps.append({"module": MODULE_NAME, "prompt": {"messages": messages}, "response": raw_response})
    return final_content


def run(prompt: str) -> tuple[str, list[dict[str, Any]]]:
    """Run the orchestrator ReAct loop. Returns (final_response, steps)."""
    steps: list[dict[str, Any]] = []
    scratchpad = ""

    for iteration in range(MAX_ITERATIONS):
        # Build messages for this iteration
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"User request: {prompt}\n\n{scratchpad}".strip()},
        ]

        # Call LLM
        llm_output, raw_response = llm_client.chat_with_raw_response(messages)
        steps.append({"module": MODULE_NAME, "prompt": {"messages": messages}, "response": raw_response})

        # Parse Thought / Action / Action Input
        parsed = _parse_react_output(llm_output)
        thought = parsed["thought"]
        action = parsed["action"]
        action_input = parsed["action_input"]

        logger.info("Iteration %d — Thought: %s | Action: %s", iteration + 1, thought[:80], action[:80] if action else "")

        # Check for finish action
        finish_match = re.match(r"finish\(\s*(.+?)\s*\)$", action, re.DOTALL | re.IGNORECASE)
        if finish_match:
            return finish_match.group(1), steps
        if action.lower() == "finish":
            return action_input, steps

        # Execute the action
        observation = _execute_action(action, action_input, steps)

        # Append to scratchpad
        scratchpad += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"

    # Max iterations reached — force finish
    logger.warning("Max iterations (%d) reached, forcing synthesis.", MAX_ITERATIONS)
    return _force_finish(prompt, scratchpad, steps), steps
