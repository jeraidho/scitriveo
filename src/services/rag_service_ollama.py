import requests
import json


class RAGServiceOllama:
    """
    service providing generative capabilities using structured research data
    """

    def __init__(self, model_name: str = "mistral"):
        """
        initialise the service to communicate with the local inference engine

        :param model_name: the identifier of the model pulled in ollama
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def generate_answer(self, question: str, documents: list[str]) -> str:
        """
        execute research synthesis using enriched context and user inquiry

        :param question: the specific research inquiry
        :param documents: list of structured metadata and abstract strings
        :returns: generated response string from the model
        """
        if not documents:
            return "no documentation provided for analysis"

        # combine structured records into a comprehensive context block
        context_block = "\n\n".join(documents)

        # design a prompt that encourages the model to use metadata for accuracy
        prompt = (
            f"instructions: use the provided research papers and their metadata to answer the question.\n"
            f"give a professional and concise scientific response.\n\n"
            f"context:\n{context_block}\n\n"
            f"question: {question}\n\n"
            f"scientific answer:"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 512,  # increased token limit for more detailed answers
                "temperature": 0.2  # lower temperature for higher factual precision
            }
        }

        try:
            # execute the request to the local api
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "Failed to extract model output")
        except Exception as error:
            return f"Generative engine connection failure: check backend service"