"""OpenAI-compatible LLM client for trace enrichment."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from openai import OpenAI

DEFAULT_MODEL = "google/gemini-2.0-flash-001"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3


@dataclass
class EnrichmentResponse:
    """Parsed response from the enrichment LLM."""

    reasoning: str
    llm_calls: list[dict[str, str]]
    tool_calls: list[dict[str, object]]
    output_to_next_agent: str

    @classmethod
    def from_dict(
        cls,
        raw: dict,
        expected_llm_calls: int | None = None,
        expected_tool_calls: int | None = None,
    ) -> EnrichmentResponse:
        llm_calls = raw.get("llm_calls", [])
        tool_calls = raw.get("tool_calls", [])

        if expected_llm_calls is not None:
            llm_calls = llm_calls[:expected_llm_calls]
        if expected_tool_calls is not None:
            tool_calls = tool_calls[:expected_tool_calls]

        # Coerce fields to expected types (LLM may return dicts instead of strings)
        reasoning = raw.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = json.dumps(reasoning)
        output = raw.get("output_to_next_agent", "")
        if not isinstance(output, str):
            output = json.dumps(output)

        return cls(
            reasoning=reasoning,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            output_to_next_agent=output,
        )


class LLMClient:
    """Thin wrapper around any OpenAI-compatible API (OpenRouter, llama.cpp, vLLM, etc.)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        is_local = base_url.startswith(("http://localhost", "http://127.0.0.1"))
        if not api_key and not is_local:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for remote endpoints. "
                "Set it to your API key (e.g. OpenRouter). "
                "Local servers (localhost) do not require a key."
            )
        self.model = model
        self._client = OpenAI(api_key=api_key or "local", base_url=base_url)

    def generate_queries(
        self, seed_queries: list[str], domain_description: str, count: int
    ) -> list[str]:
        """Generate diverse user queries by expanding from seed examples.

        Calls the LLM once with all seed queries and asks it to produce
        *count* unique, diverse queries in the same style.
        """
        system_prompt = (
            "You are a query generator. Given a domain description and example queries, "
            "generate new, unique, diverse queries in the same style. "
            'Respond with valid JSON only: {"queries": ["query1", "query2", ...]}'
        )
        seed_list = "\n".join(f"- {q}" for q in seed_queries)
        user_prompt = (
            f"Domain: {domain_description}\n\n"
            f"Example queries:\n{seed_list}\n\n"
            f"Generate exactly {count} unique, diverse queries for this domain. "
            f"Each query should be different from the examples and from each other."
        )
        result = self.generate(system_prompt, user_prompt)
        queries = result.get("queries", [])
        # Ensure we return exactly count queries, padding with seeds if needed
        if len(queries) < count:
            for i in range(count - len(queries)):
                queries.append(seed_queries[i % len(seed_queries)])
        return queries[:count]

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """Call the LLM and parse the JSON response.

        Retries up to MAX_RETRIES times on transient failures.
        """
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2**attempt)

        raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")
