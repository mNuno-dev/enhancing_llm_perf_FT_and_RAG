from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.prompt_values import StringPromptValue

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric

import re
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class AnswerAccuracy_PT(MetricWithLLM, SingleTurnMetric):
    """
    Measures answer accuracy compared to ground truth (reference) given a user_input.
    This metric averages two distinct LLM-as-ajudge prompts to evaluate.

    Top10, Zero-shoot LLM-as-a-Judge Leaderboard:
    1)- mistralai/mixtral-8x22b-instruct-v0.1
    2)- mistralai/mixtral-8x7b-instruct-v0.1
    3)- meta/llama-3.1-70b-instruct
    4)- meta/llama-3.3-70b-instruct
    5)- meta/llama-3.1-405b-instruct
    6)- mistralai/mistral-nemo-12b-instruct
    7)- nvidia/llama-3.1-nemotron-70b-instruct
    8)- meta/llama-3.1-8b-instruct
    9)- google/gemma-2-2b-it
    10)- nvidia/nemotron-mini-4b-instruct
    The top1 LB model have high correlation with human judges (~0.90).

    Attributes
    ----------
    name: string
        The name of the metrics

    answer_accuracy:
        The AnswerAccuracy object
    """

    name: str = field(default="nv_accuracy", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
                "reference",
            },
        }
    )
    # template_accuracy1 = (
    #     "Instruction: You are a world class state of the art assistant for rating "
    #     "a User Answer given a Question. The Question is completely answered by the Reference Answer.\n"
    #     "Say 4, if User Answer is full contained and equivalent to Reference Answer"
    #     "in all terms, topics, numbers, metrics, dates and units.\n"
    #     "Say 2, if User Answer is partially contained and almost equivalent to Reference Answer"
    #     "in all terms, topics, numbers, metrics, dates and units.\n"
    #     "Say 0, if User Answer is not contained in Reference Answer or not accurate in all terms, topics,"
    #     "numbers, metrics, dates and units or the User Answer do not answer the question.\n"
    #     "Do not explain or justify your rating. Your rating must be only 4, 2 or 0 according to the instructions above.\n"
    #     "### Question: {query}\n"
    #     "### {answer0}: {sentence_inference}\n"
    #     "### {answer1}: {sentence_true}\n"
    #     "The rating is:\n"
    # )
    # Prompt 1 (pergunta + resposta do utilizador + referência)
    template_accuracy1: str = (
        "Instrução: És um assistente especialista em avaliar uma Resposta do Utilizador "
        "dada uma Pergunta. A Pergunta está completamente respondida pela Resposta de Referência.\n\n"
        "Responde com:\n"
        "4 — se a Resposta do Utilizador estiver totalmente contida e for equivalente à Resposta de Referência "
        "em todos os termos, tópicos, números, métricas, datas e unidades.\n"
        "2 — se a Resposta do Utilizador estiver parcialmente contida e for quase equivalente à Resposta de Referência, "
        "com pequenas discrepâncias, em termos, tópicos, números, métricas, datas e unidades.\n"
        "0 — se a Resposta do Utilizador não estiver contida na Resposta de Referência, ou estiver incorreta "
        "em termos, tópicos, números, métricas, datas e unidades, ou não responder à Pergunta.\n\n"
        "Não expliques nem justifiques. A tua saída tem de ser apenas um número: 0, 2 ou 4.\n\n"
        "### Pergunta: {query}\n"
        "### {answer0}: {sentence_inference}\n"
        "### {answer1}: {sentence_true}\n"
        "Classificação:\n"
    )
    
    
    # template_accuracy2 = (
    #     "I will rate the User Answer in comparison to the Reference Answer for a given Question.\n"
    #     "A rating of 4 indicates that the User Answer is entirely consistent with the Reference Answer, covering all aspects, topics, numbers, metrics, dates, and units.\n"
    #     "A rating of 2 signifies that the User Answer is mostly aligned with the Reference Answer, with minor discrepancies in some areas.\n"
    #     "A rating of 0 means that the User Answer is either inaccurate, incomplete, or unrelated to the Reference Answer, or it fails to address the Question.\n"
    #     "I will provide the rating without any explanation or justification, adhering to the following scale: 0 (no match), 2 (partial match), 4 (exact match).\n"
    #     "Do not explain or justify my rating. My rating must be only 4, 2 or 0 only.\n\n"
    #     "Question: {query}\n\n"
    #     "{answer0}: {sentence_inference}\n\n"
    #     "{answer1}: {sentence_true}\n\n"
    #     "Rating: "
    # )
    # Prompt 2 (inverte a ordem, tal como no teu original)
    template_accuracy2: str = (
        "Vou classificar a Resposta do Utilizador em comparação com a Resposta de Referência para uma dada Pergunta.\n"
        "Uma classificação de 4 indica que a Resposta do Utilizador é totalmente consistente com a Resposta de Referência, cobrindo todos os aspetos, tópicos, números, métricas, datas e unidades.\n"
        "Uma classificação de 2 significa que a Resposta do Utilizador está maioritariamente alinhada com a Resposta de Referência, com discrepâncias menores.\n"
        "Uma classificação de 0 significa que a Resposta do Utilizador é incorreta, incompleta, não relacionada, ou não responde à Pergunta.\n\n"
        "Devo devolver apenas a classificação sem qualquer explicação. A minha classificação tem de ser apenas 0, 2 ou 4.\n\n"
        ""
        "Pergunta: {query}\n\n"
        "{answer0}: {sentence_inference}\n\n"
        "{answer1}: {sentence_true}\n\n"
        "Classificação: "
    )
    retry = 5  # Number of retries if rating is not in the first 8 tokens.

    # def process_score(self, response):
    #     for i in range(5):
    #         if str(i) in response[:]:
    #             return i / 4
    #     return np.nan
    
    # """
    # my process_score # 1
    # """
    # def process_score(self, response_text: str) -> float:
    #     """
    #     Extrai 0/2/4 do texto do juiz e devolve score normalizado (0.0, 0.5, 1.0).
    #     Mais robusto do que procurar 'i' em range(5).
    #     """
    #     if not response_text:
    #         return np.nan

    #     # Preferir o primeiro rótulo válido que apareça no output
    #     # (muitos LLMs devolvem "4" ou "Rating: 4", etc.)
    #     for label in ("4", "2", "0"):
    #         if label in response_text:
    #             return int(label) / 4.0
    #     return np.nan

    """
    my process_score # 2
    """
    def process_score(self, response_text: str) -> float:
        if not response_text:
            return np.nan

        text = response_text.strip()

        # Strict: if the model obeyed and returned exactly one of these
        if text in ("0", "2", "4"):
            return int(text) / 4.0

        # Otherwise: find the first standalone 0/2/4 token (avoids "0/2/4" issues better than `in`)
        m = re.search(r'(^|\s)(0|2|4)(\s|$)', text)
        if m:
            return int(m.group(2)) / 4.0

        return np.nan

    def average_scores(self, score0, score1):
        score = np.nan
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        else:
            score = max(score0, score1)
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.user_input is not None, "User input is not set"
        assert sample.response is not None, "Response is not set"
        assert sample.reference is not None, "Reference is not set"

        try:
            score_ref_gen = score_gen_ref = np.nan
            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_accuracy1.format(
                        query=sample.user_input,
                        answer0="User Answer",
                        answer1="Reference Answer",
                        sentence_inference=sample.response,
                        sentence_true=sample.reference,
                    )
                )
                req0 = self.llm.agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.10,
                )
                resp0 = await req0
                score_ref_gen = resp0.generations[0][0].text
                score_ref_gen = self.process_score(score_ref_gen)
                if score_ref_gen == score_ref_gen:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_accuracy2.format(
                        query=sample.user_input,
                        answer0="Reference Answer",
                        answer1="User Answer",
                        sentence_inference=sample.reference,
                        sentence_true=sample.response,
                    )
                )
                req1 = self.llm.agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.10,
                )
                resp1 = await req1
                score_gen_ref = resp1.generations[0][0].text
                score_gen_ref = self.process_score(score_gen_ref)
                if score_gen_ref == score_gen_ref:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            score = self.average_scores(score_ref_gen, score_gen_ref)

        except Exception as e:
            logger.warning(
                f"An error occurred: {e}. Skipping a sample by assigning it nan score."
            )
            score = np.nan

        return score


@dataclass
class ContextRelevance_PT(MetricWithLLM, SingleTurnMetric):
    """Parameters:
    Score the relevance of the retrieved contexts be based on the user input.

    Input:
        data: list of Dicts with keys: user_input, retrieved_contexts
    Output:
        0.0: retrieved_contexts is not relevant for the user_input
        0.5: retrieved_contexts is partially relevant for the user_input
        1.0: retrieved_contexts is fully relevant for the user_input
    """

    name: str = field(default="nv_context_relevance", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "retrieved_contexts",
            },
        }
    )
    # template_relevance1 = (
    #     "### Instructions\n\n"
    #     "You are a world class expert designed to evaluate the relevance score of a Context"
    #     " in order to answer the Question.\n"
    #     "Your task is to determine if the Context contains proper information to answer the Question.\n"
    #     "Do not rely on your previous knowledge about the Question.\n"
    #     "Use only what is written in the Context and in the Question.\n"
    #     "Follow the instructions below:\n"
    #     "0. If the context does not contains any relevant information to answer the question, say 0.\n"
    #     "1. If the context partially contains relevant information to answer the question, say 1.\n"
    #     "2. If the context contains any relevant information to answer the question, say 2.\n"
    #     "You must provide the relevance score of 0, 1, or 2, nothing else.\nDo not explain.\n"
    #     "### Question: {query}\n\n"
    #     "### Context: {context}\n\n"
    #     "Do not try to explain.\n"
    #     "Analyzing Context and Question, the Relevance score is "
    # )
    # template_relevance2 = (
    #     "As a specially designed expert to assess the relevance score of a given Context in relation to a Question, "
    #     "my task is to determine the extent to which the Context provides information necessary to answer the Question. "
    #     "I will rely solely on the information provided in the Context and Question, and not on any prior knowledge.\n\n"
    #     "Here are the instructions I will follow:\n"
    #     "* If the Context does not contain any relevant information to answer the Question, I will respond with a relevance score of 0.\n"
    #     "* If the Context partially contains relevant information to answer the Question, I will respond with a relevance score of 1.\n"
    #     "* If the Context contains any relevant information to answer the Question, I will respond with a relevance score of 2.\n\n"
    #     "### Question: {query}\n\n"
    #     "### Context: {context}\n\n"
    #     "Do not try to explain.\n"
    #     "Based on the provided Question and Context, the Relevance score is  ["
    # )

    template_relevance1 = (
        "### Instruções\n\n"
        "És um grande perito na tarefa de avaliar o grau de relevância de um Contexto "
        "para responder a uma Pergunta.\n"
        "A tua tarefa é determinar se o Contexto contém informação adequada para responder à Pergunta.\n"
        "Não utilizes como base o teu conhecimento prévio sobre a Pergunta.\n"
        "Usa apenas o que está escrito no Contexto e na Pergunta.\n"
        "Segue as instruções abaixo:\n"
        "0. Se o contexto não contém nenhuma informação relevante para responder à pergunta, diz 0.\n"
        "1. Se o contexto contém alguma informação parcialmente relevante para responder à pergunta, diz 1.\n"
        "2. Se o contexto contém alguma informação relevante para responder à pergunta, diz 2.\n"
        "Deves fornecer o grau de relevância 0, 1 ou 2, nada mais.\nNão expliques.\n"
        "### Pergunta: {query}\n\n"
        "### Contexto: {context}\n\n"
        "Não tentes explicar.\n"
        "Analisando o Contexto e a Pergunta, o grau de relevância é "
    )

    template_relevance2 = (
        "Como um perito especialmente designado para avaliar o grau de relevância de um dado Contexto em relação a uma Pergunta, "
        "a minha tarefa é determinar em que medida o Contexto fornece a informação necessária para responder à Pergunta. "
        "Vou basear-me unicamente na informação fornecida no Contexto e na Pergunta, e não em qualquer conhecimento prévio.\n\n"
        "Aqui estão as instruções que vou seguir:\n"
        "* Se o Contexto não contiver nenhuma informação relevante para responder à Pergunta, responderei com o grau de relevância 0.\n"
        "* Se o Contexto contiver alguma informação parcialmente relevante para responder à Pergunta, responderei com o grau de relevância 1.\n"
        "* Se o Contexto contiver alguma informação relevante para responder à Pergunta, responderei com o grau de relevância 2.\n\n"
        "### Pergunta: {query}\n\n"
        "### Contexto: {context}\n\n"
        "Não tentes explicar.\n"
        "Com base na Pergunta e Contexto fornecidos, o grau de relevância é  ["
    )

    retry = 5  # Number of retries if rating is not in the first 8 tokens.

    def process_score(self, response):
        for i in [2, 1, 0]:
            if str(i) in response:
                return i / 2
        return np.nan

    def average_scores(self, score0, score1):
        score = np.nan
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        else:
            score = max(score0, score1)
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.user_input is not None, "User input is not set"
        assert sample.retrieved_contexts is not None, "Retrieved Context is not set"

        if (sample.user_input.strip() == "") or (
            "\n".join(sample.retrieved_contexts).strip() == ""
        ):
            return 0.0
        if sample.user_input.strip() == "\n".join(sample.retrieved_contexts).strip():
            return 0.0
        if "\n".join(sample.retrieved_contexts).strip() in sample.user_input.strip():
            return 0.0

        try:
            score0 = score1 = np.nan
            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_relevance1.format(
                        query=sample.user_input,
                        context="\n".join(sample.retrieved_contexts)[:7000],
                    )
                )
                req = self.llm.agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.1,
                )
                resp = await req
                score0 = self.process_score(resp.generations[0][0].text)
                if score0 == score0:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_relevance1.format(
                        query=sample.user_input,
                        context="\n".join(sample.retrieved_contexts)[:7000],
                    )
                )
                req = self.llm.agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.1,
                )
                resp = await req
                score1 = self.process_score(resp.generations[0][0].text)
                if score1 == score1:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            score = self.average_scores(score0, score1)

        except Exception as e:
            print(
                f"An error occurred: {e}. Skipping a sample by assigning it nan score."
            )
            score = np.nan

        return score


@dataclass
class ResponseGroundedness(MetricWithLLM, SingleTurnMetric):
    """Parameters:
    Score the groundedness of the response based on the retrieved contexts.

    Input:
        data: list of Dicts with keys: response, retrieved contexts
    Output:
        0.0: response is not grounded in the retrieved contexts
        0.5: response is partially grounded in the retrieved contexts
        1.0: response is fully grounded in the retrieved contexts
    """

    name: str = field(default="nv_response_groundedness", repr=True)  # type: ignore
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "response",
                "retrieved_contexts",
            },
        }
    )
    template_groundedness1 = (
        "### Instruction\n\n"
        "You are a world class expert designed to evaluate the groundedness of an assertion.\n"
        "You will be provided with an assertion and a context.\n"
        "Your task is to determine if the assertion is supported by the context.\n"
        "Follow the instructions below:\n"
        "A. If there is no context or no assertion or context is empty or assertion is empty, say 0.\n"
        "B. If the assertion is not supported by the context, say 0.\n"
        "C. If the assertion is partially supported by the context, say 1.\n"
        "D. If the assertion is fully supported by the context, say 2.\n"
        "You must provide a rating of 0, 1, or 2, nothing else.\n\n"
        "### Context:\n"
        "<{context}>\n\n"
        "### Assertion:\n"
        "<{response}>\n\n"
        "Analyzing Context and Response, the Groundedness score is "
    )
    template_groundedness2 = (
        "As a specialist in assessing the strength of connections between statements and their given contexts, "
        "I will evaluate the level of support an assertion receives from the provided context. Follow these guidelines:\n\n"
        "* If the assertion is not supported or context is empty or assertion is empty, assign a score of 0.\n"
        "* If the assertion is partially supported, assign a score of 1.\n"
        "* If the assertion is fully supported, assign a score of 2.\n\n"
        "I will provide a rating of 0, 1, or 2, without any additional information.\n\n"
        "---\n**Context:**\n[{context}]\n\n"
        "**Assertion:**\n[{response}]\n\n"
        "Do not explain."
        "Based on the provided context and response, the Groundedness score is:"
    )
    retry = 5  # Number of retries if rating is not in the first 8 tokens.

    def process_score(self, response):
        for i in [2, 1, 0]:
            if str(i) in response:
                return i / 2
        return np.nan

    def average_scores(self, score0, score1):
        score = np.nan
        if score0 >= 0 and score1 >= 0:
            score = (score0 + score1) / 2
        else:
            score = max(score0, score1)
        return score

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        assert self.llm is not None, "LLM is not set"
        assert sample.response is not None, "Response is not set"
        assert sample.retrieved_contexts is not None, "Retrieved Context is not set"

        if (sample.response.strip() == "") or (
            "\n".join(sample.retrieved_contexts).strip().strip() == ""
        ):
            return 0.0
        if sample.response.strip() == "\n".join(sample.retrieved_contexts).strip():
            return 1.0
        if sample.response.strip() in "\n".join(sample.retrieved_contexts).strip():
            return 1.0

        try:
            score0 = score1 = np.nan
            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_groundedness1.format(
                        context="\n".join(sample.retrieved_contexts)[:7000],
                        response=sample.response,
                    )
                )
                req = self.llm.agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.1,
                )
                resp = await req
                score0 = self.process_score(resp.generations[0][0].text)
                if score0 == score0:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            for retry in range(self.retry):
                formatted_prompt = StringPromptValue(
                    text=self.template_groundedness2.format(
                        context="\n".join(sample.retrieved_contexts)[:7000],
                        response=sample.response,
                    )
                )
                req = self.llm.agenerate_text(
                    formatted_prompt,
                    n=1,
                    temperature=0.1,
                )
                resp = await req
                score1 = self.process_score(resp.generations[0][0].text)
                if score1 == score1:
                    break
                else:
                    logger.warning(f"Retry: {retry}")

            score = self.average_scores(score0, score1)

        except Exception as e:
            print(
                f"An error occurred: {e}. Skipping a sample by assigning it nan score."
            )
            score = np.nan

        return score
