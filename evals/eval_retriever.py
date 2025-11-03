import sys
import os
from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.rag.retrieval_generation import rag_pipeline

from langsmith import Client
from qdrant_client import QdrantClient

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ls_client = Client()
qdrant_client = QdrantClient(
    url=f"http://localhost:6333"
)

from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))


async def ragas_faithfulness(run, example):

    sample = SingleTurnSample(
            user_input=run.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = Faithfulness(llm=ragas_llm)

    return await scorer.single_turn_ascore(sample)


async def ragas_response_relevancy(run, example):

    sample = SingleTurnSample(
            user_input=run.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    return await scorer.single_turn_ascore(sample)


async def ragas_context_precision(run, example):

    sample = SingleTurnSample(
            user_input=run.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = ContextPrecision(llm=ragas_llm)

    return await scorer.single_turn_ascore(sample)


async def ragas_context_recall(run, example):

    sample = SingleTurnSample(
            user_input=run.inputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"],
            ground_truth=example.outputs["ground_truth"]
        )
    scorer = ContextRecall(llm=ragas_llm)

    return await scorer.single_turn_ascore(sample)


results = ls_client.evaluate(
    lambda x: rag_pipeline(x["inputs"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision,
        ragas_context_recall
    ],
    experiment_prefix="retriever"
)