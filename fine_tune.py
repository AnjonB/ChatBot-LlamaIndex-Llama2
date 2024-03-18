import os, sys
from datetime import datetime
from gradientai import Gradient
from samples import samples
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientModelAdapterLLM
from llama_index import ServiceContext
from llama_index import set_global_service_context

NUM_EPOCHS = 5

class FineTuner:
    def __init__(self, model_name, num_epochs):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.gradient = None
        self.model_adapter = None

    def initialize_gradient(self):
        self.gradient = Gradient()

    def create_model_adapter(self):
        # Create model adapter with the specified name
        base_model = self.gradient.get_base_model(base_model_slug="llama2-7b-chat")
        model_adapter = base_model.create_model_adapter(name=self.model_name)
        return model_adapter

    def fine_tune_model(self, samples):
        count = 0
        while count < NUM_EPOCHS:
            print(f"Fine-tuning the model with iteration {count + 1}")
            self.model_adapter.fine_tune(samples=samples)
            count = count + 1

    def fine_tune(self):
        try:
            # Initialize logging
          
            # Initialize Gradient AI Cloud
            self.initialize_gradient()

            # Create model adapter
            self.model_adapter = self.create_model_adapter()
            print(f"Created model adapter with id {self.model_adapter.id}")

            # Fine-tune the model
            self.fine_tune_model(samples)
            
            llm = GradientModelAdapterLLM(model_adapter_id = self.model_adapter.id, max_tokens=200)

            # Initialize Gradient AI Cloud with credentials
            embed_model = GradientEmbedding(
                                gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
                                gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
                                gradient_model_slug="bge-large")

            service_context = ServiceContext.from_defaults(
                        llm = llm,
                        embed_model = embed_model,
                        chunk_size=256)

            set_global_service_context(service_context)
            
            return service_context

        except Exception as e:
        # Handle exceptions using custom exception class and logging
            raise Llama2Exception(e, sys)
            

class Llama2Exception(Exception):
    """
    Custom exception class for handling money laundering-related errors.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Method Name : __init__
        Description : Initialize the MoneyLaunderingException exception.
        Return : None
        Args  :
            error_message (str): The main error message.
            error_detail (sys): Additional details about the error.
        """
        super().__init__(error_message)
        self.error_message_detail = error_detail

    def __str__(self):
        """
        Method Name : __str__
        Description : Return a string representation of the MoneyLaundering exception.
        Return : str
        Args  : None
        """
        return str(self.error_message_detail)