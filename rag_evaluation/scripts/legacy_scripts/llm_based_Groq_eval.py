"""
NOTE: This code needs a high performing model as for example the:
    - llama-3.1-8b-instant is unable to generate the response in json and so RAGAs fails

"""

import os
import asyncio
import json
import re

from langchain_groq import ChatGroq
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AspectCritic


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")  # set via environment variable


# --- Custom Wrapper that sanitizes invalid JSON outputs ---
class SafeLangchainLLMWrapper(LangchainLLMWrapper):
    async def ainvoke(self, prompt, **kwargs):
        result = await super().ainvoke(prompt, **kwargs)
        if isinstance(result, str):
            return self._safe_json(result)
        return result

    def _safe_json(self, text: str):
        # Try to extract JSON-like content
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            # fallback fake JSON
            return {"score": 0, "explanation": text.strip()[:200]}


async def main():
    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

    evaluator_llm = SafeLangchainLLMWrapper(llm)

    test_data = {
        "user_input": (
            "summarise given text\n"
            "The company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. "
            "Sales in this region have significantly contributed to the overall growth. "
            "Analysts attribute this success to strategic marketing and product localization. "
            "The positive trend in the Asian market is expected to continue into the next quarter."
        ),
        "response": (
            "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and "
            "product adaptation, with expectations of continued growth in the coming quarter."
        ),
    }

    metric = AspectCritic(
        name="summary_accuracy",
        llm=evaluator_llm,
        definition="Verify if the summary accurately reflects the original text."
    )

    test_sample = SingleTurnSample(**test_data)

    try:
        result = await asyncio.wait_for(metric.single_turn_ascore(test_sample), timeout=60)
        print("\n Aspect Score:", result)
    except asyncio.TimeoutError:
        print("Timeout: Groq model took too long to respond.")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
