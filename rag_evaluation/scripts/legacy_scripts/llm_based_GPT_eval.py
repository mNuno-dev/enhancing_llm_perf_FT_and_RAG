
import os
import asyncio

# pip install langchain-openai
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import AspectCritic

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")  # set via environment variable

async def main():

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    # evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


    test_data = {
        "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
        "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
    }


    metric = AspectCritic(name="summary_accuracy", llm=evaluator_llm, definition="Verify if the summary is accurate.")
    test_sample = SingleTurnSample(**test_data)

    result = await metric.single_turn_ascore(test_sample)
    print("Aspect Score:", result)

    print(result)


asyncio.run(main())

