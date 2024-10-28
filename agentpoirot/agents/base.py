import os
from openai import OpenAI


class BaseAgent:
    def __init__(
        self,
        tables,
        savedir=None,
        gen_engine=None,
        meta_dict=None,
        verbose=False,
        use_pandas_ai=False,
        prompt_question="basic",
        prompt_code="basic",
        prompt_interpret="basic",
        prompt_summarize="basic",
        prompt_followup="basic",
        prompt_action="basic",
        prompt_meta="basic",
        prompt_dataset_description="basic",
        signature=None,  # Adding signature attribute
    ):
        self.verbose = verbose
        self.meta_dict = meta_dict
        self.savedir = savedir or tempfile.mkdtemp()
        self.prompt_question = prompt_question
        self.prompt_code = prompt_code
        self.prompt_interpret = prompt_interpret
        self.prompt_summarize = prompt_summarize
        self.prompt_followup = prompt_followup
        self.prompt_action = prompt_action
        self.prompt_meta = prompt_meta
        self.prompt_dataset_description = prompt_dataset_description
        self.gen_engine = gen_engine
        self.signature = signature
        self.tables = tables
        self.schemas = [self.get_schema(df) for df in self.tables]

        if self.meta_dict is None:
            self.meta_dict = self.predict_meta()


def get_chat_model(model_name, temperature=0):
    """
    Gets the chat model based on the name of the model

    Args:
    model_name (str): Name of the model. Examples: "gpt-4o-mini", "gpt-4o-turbo"
    """
    assert "gpt" in model_name, f"Model {model_name} is not supported"

    llm_dict = get_llm(model_name)

    client = OpenAI(api_key=llm_dict["OPENAI_API_KEY"])
    llm = (
        lambda content: client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[{"role": "user", "content": content}],
        )
        .choices[0]
        .message.content
    )

    return llm


def get_llm(name):
    if "gpt" in name:
        return {
            "model_name": name,
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "temperature": 0,
        }
    else:
        raise ValueError(f"Model {name} is not supported")
