from transformers import AutoModel
from pathlib import Path
import torch


class RAGService:
    """
    Service to provide answering using apple clara architecture
    Based on: https://huggingface.co/apple/CLaRa-7B-Instruct
    """

    def __init__(self, model_path: Path, device: str = "cuda"):
        """
        Initialise the generative service with a specific model path
        :param model_path: local path or huggingface id for the model
        :param device: target hardware device
        """
        self.model_path = model_path
        self.device = device
        self._model = None

    @property
    def model(self):
        """
        Lazy loader for the transformer model to save memory
        :returns: initialised model instance
        """
        if self._model is None:
            dtype = torch.float16 if self.device == "mps" else torch.float32

            try:
                # explicit device loading without auto mapping for stability
                self._model = AutoModel.from_pretrained(
                    str(self.model_path),
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    low_cpu_mem_usage = True
                ).to(self.device)
            except Exception as error:
                print(f"model loading failed during initialization: {error}")
                raise error

        return self._model

    def generate_answer(self, question: str, context_documents: list[str], max_tokens: int = 256) -> str:
        """
        Genetrate a response based on a question and a list of context documents
        :param question: user query
        :param context_documents: list of strings containing paper data
        :param max_tokens: limit for the generated response
        :returns: answer string
        """
        # clara expects questions as a list and documents as a list of lists
        # length of outer lists must match exactly
        questions = [question]
        nested_docs = [context_documents]

        # trigger inference via the internal generation method
        output = self.model.generate_from_text(
            questions=questions,
            documents=nested_docs,
            max_new_tokens=max_tokens
        )

        # extract the first result from the returned list
        return output[0] if output else "Sorry, no answer generated"
