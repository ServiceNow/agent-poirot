import json
import os
import tempfile
import time
import textwrap

import re

from functools import partial
from agentpoirot import prompts
from agentpoirot.agents.base import BaseAgent, get_chat_model
from agentpoirot import utils as ut


class Poirot(BaseAgent):

    def __init__(
        self,
        table=None,
        savedir=None,
        model_name=None,
        meta_dict=None,
        verbose=False,
        prompt_question="basic",
        prompt_code="basic",
        prompt_interpret="basic",
        prompt_summarize="basic",
        prompt_followup="basic",
        prompt_action="basic",
        prompt_meta="basic",
        prompt_dataset_description="basic",
    ):
        self.verbose = verbose
        if meta_dict is None:
            meta_dict = {
                "role": "data scientist",
                "goal": "I want to find interesting trends in this dataset",
                "indicator_list": [
                    "anomaly detection",
                    "correlation analysis",
                    "outlier detection",
                    "trend analysis",
                ],
            }
        self.meta_dict = meta_dict
        if savedir is None:
            savedir = tempfile.mkdtemp()
        self.savedir = savedir

        # DEFINE PROMPTS
        self.prompt_question = prompt_question
        self.prompt_code = prompt_code
        self.prompt_interpret = prompt_interpret
        self.prompt_summarize = prompt_summarize
        self.prompt_followup = prompt_followup
        self.prompt_action = prompt_action
        self.prompt_meta = prompt_meta
        self.prompt_dataset_description = prompt_dataset_description
        self.model_name = model_name
        self.output_dict = None

        # instantiate llm model
        self.chat_model = get_chat_model(self.model_name)

        self.insights_history = []
        self.verbose = verbose

        # TABLE META INFO
        table = ut.clean_table(table)
        self.dataset_csv_path = os.path.join(self.savedir, "dataset.csv")
        table.to_csv(self.dataset_csv_path, index=False)
        self.table = table
        self.schemas = ut.get_schema(table)
        self.meta = ut.dict_to_string(self.meta_dict)

    def forward(self, prompt):
        return self.chat_model(prompt)

    def get_dataset_description(self):
        # print the description of the dataset
        if self.verbose:
            print(f"Generating dataset description...")

        prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../prompts/stage_0_dataset_description.txt",
        )
        prompt_template = ut.load_txt(prompt_path)
        prompt = prompt_template.format(meta=self.meta, schemas=self.schemas)

        def _validate_tasks(out):
            try:
                description = ut.extract_html_tags(out, ["description"])["description"]
            except Exception as e:
                return (
                    out,
                    False,
                    f"Error: {str(e)}",
                )

            return (description, out), True, ""

        data_description, message, _ = _validate_tasks(self.chat_model(prompt))
        return data_description[0]

    def clean_tables(self, tables, inplace=True):
        """
        tables is a df or a list of dfs
        """
        self.tables = self.tables

    def answer_question(
        self,
        question,
        n_retries=3,
        output_folder=None,
        add_timestamp=True,
        with_pdf=False,
    ):
        """
        should save the insight_dict in the output_folder
        with keys
                {
            "answer": "The primary reasons for escalations in these cases are predominantly due to 'Multiple Escalations', which occurred in 70% of the cases analyzed, with an average escalation count of 2.6, peaking at 7 escalations in some instances.",
            "followup": "What are the common characteristics of cases with multiple escalations?",
            "header": "High frequency of multiple escalations observed",
            "indicator": "Escalation_and_reassignment_patterns; the high frequency of multiple escalations suggests a need for review of case assignments.",
            "insight": "This finding highlights that 70% of the cases had 'Multiple Escalations', suggesting a systemic issue in case management, as the average escalation count was 2.6, with some cases reaching as high as 7, which could indicate that initial assignments may not be adequately addressing customer needs, raising concerns about service efficiency.",
            "output_folder": "static/users/g/csm/insight_card_40",
            "plot_html": "plot.html",
            "plot_image": "plot.jpg",
            "question": "What are the specific reasons for escalations in these cases?",
            "score": "85",
            "severity": "high"
        }

        """
        # generate code
        code_output = self.generate_code(
            question,
            n_retries=n_retries,
            output_folder=output_folder,
            add_timestamp=add_timestamp,
        )

        # get insight
        insight_dict = self.generate_insight(
            code_output, n_retries=n_retries, with_pdf=with_pdf
        )

        return insight_dict

    def generate_insights(self, n_insights=5, as_str=False):
        """
        Generate insights for a set of questions

        Parameters:
        -----------
        n_questions: int
            The number of questions to generate insights for

        Returns:
        --------
        insights: list
            A list of insights for each question
        """
        questions = self.generate_questions(n_questions=n_insights)
        insight_list = self.get_insights(questions)

        if as_str:
            return self.display_insights(insight_list)
        return insight_list

    def generate_insight(
        self, code_output: dict, n_retries=2, return_prompt=False, with_pdf=False
    ) -> str:
        """
        Produce insights for a task based on a code_output output by a model

        Parameters:
        -----------
        code_output: dict
            The output of the code generation function
        answer_template: dict
            A template for the answer that the human should provide. This template should contain a "results" tag
            that contains a list of expected results in the form of dictionaries. Each dictionary should contain
            the following keys: "name", "description", and "value". The model will be asked to fill in the values.
        model: str
            The name of the model to use (default: gpt-4)
        n_retries: int
            The number of times to retry the interpretation if it fails

        Returns:
        --------
        code_output_path: str
            The path to the input code_output file, which has been updated with the interpretation

        """
        itime = time.time()
        output_folder = code_output["output_folder"]

        prompt_path = get_prompt_path("stage_4_generate_insight.txt")
        prompt_template = ut.load_txt(prompt_path)
        # create prompt
        prompt = prompt_template.format(
            meta=self.meta,
            question=code_output["question"],
            stat=str(code_output["stat"]),
            sample=str(self.table.iloc[:, :5].head().to_dict(orient="records")),
            code_used=str(code_output["code"]),
        )

        # Get human readable answer
        # insight_dict, completions = ut.retry_on_parsing_error(
        #     self.chat_model,
        #     prompt,
        #     parser=ut.parse_insight,
        #     n_retries=n_retries,
        # )
        from agentpoirot.agents.llms import prompt_llm

        print(code_output["plot"]["name"])
        completions = []
        completions.append(
            {
                "code": prompt_llm(
                    prompt, model="gpt-4o", image=code_output["plot"]["name"]
                ),
                "prompt": prompt,
            }
        )
        insight_dict, _, _ = ut.parse_insight(completions[-1]["code"])
        # print()
        # code_output["interpretation"] = out

        insight_dict["question"] = code_output["question"]
        insight_dict["output_folder"] = code_output["output_folder"]
        if os.path.exists(f"{code_output['output_folder']}/plot.jpg"):
            insight_dict["plot_image"] = f"{code_output['output_folder']}/plot.jpg"
        # chec if plot_html exists
        if os.path.exists(f"{code_output['output_folder']}/plot.html"):
            insight_dict["plot_html"] = f"{code_output['output_folder']}/plot.html"

        # save prompt
        prompt_insight = completions[-1]["prompt"]
        with open(os.path.join(output_folder, "prompt_insight.txt"), "w") as f:
            f.write(prompt_insight)

        if self.verbose:
            print("Saved insight_dict.json in ", output_folder)

        insight_dict["time_elapsed"] = {
            "code_gen": code_output["code_time"],
            "insight_gen": time.time() - itime,
        }

        ut.save_json(
            os.path.join(output_folder, "insight_dict.json"),
            insight_dict,
        )
        # save pdf and html
        if with_pdf:
            ut.insight2pdf(output_folder)

        if return_prompt:
            return insight_dict, completions[-1]["prompt"]

        return insight_dict

    def verify_question(self, question):
        """
        The function verifies the question by asking the user to verify the question

        Returns
        -------
        isValid: bool
            True if the question is valid, False otherwise
        explanation: str
            The explanation for the verification
        """
        prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../prompts/_verify.txt",
        )
        prompt_template = ut.load_txt(prompt_path)
        prompt = prompt_template.format(question=question, schema=self.schemas)
        response = self.chat_model(prompt)

        def is_valid_verification(verification_text):
            pattern = r"<verification>\s*result:\s*(RELATED|UNRELATED)\s*explanation:\s*(.+?)\s*score:\s*(.+?)\s*</verification>"
            match = re.search(pattern, verification_text, re.DOTALL | re.IGNORECASE)

            if match:
                result = match.group(1).upper()
                explanation = match.group(2).strip()
                score = match.group(3).strip()
                if len(explanation) == 0:
                    raise ValueError("Explanation is empty")
                return result == "RELATED", explanation, score
            else:
                raise ValueError("No valid verification format found in the text")

        isValid, explanation, score = is_valid_verification(response)

        return isValid, explanation, score

    def question_breakdown_extract_questions_and_explanations(self, response_text):
        """
        Extracts questions and their explanations from the LLM response text.

        This function parses the output of the question analysis prompt, extracting
        each question and its corresponding explanation.

        Parameters:
        -----------
        response_text : str
            The full text response from the LLM, containing the questions and explanations.

        Returns:
        --------
        list of dict
            A list where each element is a dictionary containing:
            {
                'question': str,
                'explanation': str
            }

        Example:
        --------
        >>> response = '''
        ... <questions>
        ... <question>
        ... What is the monthly trend of total incidents?
        ... <explanation>This question focuses on time-based trends.</explanation>
        ... </question>
        ... <question>
        ... How does incident severity vary by location?
        ... <explanation>This explores geographical patterns in severity.</explanation>
        ... </question>
        ... </questions>
        ... '''
        >>> questions = extract_questions_and_explanations(response)
        >>> print(questions)
        [
            {
                'question': 'What is the monthly trend of total incidents?',
                'explanation': 'This question focuses on time-based trends.'
            },
            {
                'question': 'How does incident severity vary by location?',
                'explanation': 'This explores geographical patterns in severity.'
            }
        ]
        """
        # Extract questions and explanations
        pattern = r"<question>\s*(.*?)\s*<explanation>(.*?)</explanation>\s*</question>"
        matches = re.findall(pattern, response_text, re.DOTALL)

        if not matches:
            raise ValueError("No valid question format found in the text")

        questions = []
        explanations = []
        for question, explanation in matches:
            question = question.strip()
            explanation = explanation.strip()
            if not explanation:
                raise ValueError(f"No explanation found for question: {question}")
            questions.append(question)
            explanations.append(explanation)

        return questions, explanations

    def generate_questions(
        self,
        n_questions=3,
        return_prompt=False,
        return_question_only=True,
        past_questions=None,
    ):
        # print the description of the dataset
        if self.verbose:
            print(f"Generating {n_questions} Questions using {self.model_name}...")
        if past_questions is None:
            past_questions = ["Give me an overview of the dataset"]

        prompt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../prompts/stage_2_get_questions.txt",
        )
        prompt_template = ut.load_txt(prompt_path)
        prompt = prompt_template.format(
            meta=self.meta,
            schemas=self.schemas,
            past_questions=past_questions,
            max_questions=n_questions,
        )

        def _validate_tasks(out):
            questions = ut.parse_questions(out)
            if "question" not in questions[0]:
                return (
                    out,
                    False,
                    f"Error: you did not generate questions within the <question></question> tags",
                )
            # Check that there are at most max_questions questions
            if len(questions) > n_questions:
                return (
                    out,
                    False,
                    f"Error: you can only ask at most {n_questions} questions, but you asked {len(questions)}.",
                )

            return questions, out, True, ""

        questions, _, _, _ = _validate_tasks(self.chat_model(prompt))
        if return_question_only:
            questions = [q["question"] for q in questions]

        if return_prompt:
            return questions, prompt

        # save prompts in savedir
        with open(os.path.join(self.savedir, "prompt_get_questions.txt"), "w") as f:
            f.write(prompt)
            print(f"Prompt saved in {self.savedir}/prompt_get_questions.txt")

        return questions

    def refine_question(self, question, max_questions=3, return_explanation=False):
        """
        Breaks down a complex question into simpler, more specific questions using a language model.
        Returns two lists: one containing questions and another containing explanations.
        """
        metadata = ut.dict_to_string(self.meta_dict)
        prompt_path = get_prompt_path("_refine.txt")
        prompt_template = ut.load_txt(prompt_path)

        prompt = prompt_template.format(
            question=question,
            schema=self.schemas,
            metadata=metadata,
            max_questions=1,
        )
        response = self.chat_model(prompt)
        questions, explanations = (
            self.question_breakdown_extract_questions_and_explanations(response)
        )
        if return_explanation:
            return questions[0], explanations[0]
        return questions[0]

    def save_state_dict(self, fname):
        with open(fname, "w") as f:
            json.dump(self.insights_history, f, indent=4)

    def load_state_dict(self, fname):
        with open(fname, "r") as f:
            self.insights_history = json.load(f)

    def generate_code(
        self,
        question,
        n_retries=3,
        return_prompt=False,
        output_folder=None,
        add_timestamp=True,
    ):
        """
        Solve a task using the naive single step approach

        See main function docstring for more details

        """
        gentime = time.time()
        assert isinstance(question, str), "Question should be a string"

        # Prompt 3: Generate Code
        if output_folder is None:
            output_folder = os.path.join(self.savedir, f"question_{int(time.time())}")
        else:
            if add_timestamp:
                output_folder = os.path.join(
                    output_folder, f"question_{int(time.time())}"
                )
        if self.verbose:
            print(f"Generating Code at {output_folder}... for Question {question}")
        os.makedirs(output_folder, exist_ok=True)

        import_name = "from agentpoirot.tools import func_tools"
        function_docs = ut.get_available_functions(question=question)
        prompt_path = get_prompt_path("stage_3_generate_code.txt")
        prompt_template = ut.load_txt(prompt_path)
        initial_prompt = prompt_template.format(
            meta=self.meta,
            schemas=self.schemas,
            question=question,
            import_name=import_name,
            database_path=os.path.abspath(self.dataset_csv_path),
            function_docs=function_docs,
            # header=get_header(self.tables[0]),
        )

        # Run the retry on error function
        output, completions = ut.retry_on_parsing_error(
            llm=self.chat_model,
            initial_prompt=initial_prompt,
            parser=partial(ut._code_parser, output_folder=output_folder),
            n_retries=n_retries,
            exception_on_max_retries=False,
        )

        # Create the output dict
        # Then, iterate over all generated plots and add them to the output dict
        output_dict = {
            "code": completions[-1]["code"],
            "prompt": str(completions[-1]["prompt"]),
            "code_output": output,
            "message": output,
            "n_retries": len(completions) - 1,
            "meta_dict": self.meta_dict,
            "question": question,
            "output_folder": output_folder,
        }

        error = ""
        # write code to a file
        with open(f"{output_folder}/code.py", "w") as file:
            # use regex to capture the python code block
            code = completions[-1]["code"]
            try:
                code = re.findall(r"```python(.*?)```", code, re.DOTALL)[0]
                file.write(code.strip())
            except Exception as e:
                error += str(f"Failed to write code", e)
                file.write(code.strip())

        # write question to a file
        with open(f"{output_folder}/question.txt", "w") as file:
            file.write(str(question))

        # write the prompt to a file
        with open(f"{output_folder}/prompt_code.txt", "w") as file:
            file.write(str(completions[-1]["prompt"]))

        # Try to load each json file in the directory
        for file in os.listdir(output_folder):
            if file.endswith(".json"):
                try:
                    json.load(open(f"{output_folder}/{file}", "r"))
                except Exception as e:
                    error += str(f"Failed to load {output_folder}/{file}", e)

        if error != "":
            print(error)
            # save error as error.txt
            with open(f"{output_folder}/error.txt", "w") as file:
                file.write(str(error))

        # Add the plot to the final output dict
        plot_path = f"{output_folder}/plot.jpg"
        # load stat.json
        if not os.path.exists(f"{output_folder}/stat.json"):
            print(f"stat.json not found in {output_folder}")
            stat = {}
        else:
            with open(f"{output_folder}/stat.json", "r") as file:
                stat = json.load(file)
        plot_dict = {"name": plot_path, "type": "plot"}
        # TODO: maybe remove this
        for k in stat.keys():
            # sample
            if stat[k] is not None and "values" in stat[k]:
                stat[k]["values"] = ut.equally_spaced(stat[k]["values"])

        output_dict["stat"] = stat
        output_dict["plot"] = plot_dict

        if self.verbose:
            print(f"Code Output for Question {question} saved in ", output_folder)

        output_dict["code_time"] = time.time() - gentime

        ut.save_json(
            os.path.join(output_folder, "code_output.json"),
            output_dict,
        )

        if return_prompt:
            return output_dict, completions[-1]["prompt"]

        return output_dict

    def verify_code_validity(self, code, verbose=False):
        """
        Verify the validity of the given code against a template.

        Args:
            code (str): The code to be verified.
            template_code (str): The template code to compare against.
            verbose (bool, optional): If True, print verbose output. Defaults to False.

        Returns:
            dict: A dictionary containing the validity results with keys:
                - 'valid' (bool): Whether the code is valid.
                - 'function_name' (str): The name of the function if valid.
                - 'explanation' (str): An explanation of the validity result.
        """
        prompt_template = ut.load_txt(get_prompt_path("agentada/validate.txt"))
        prompt = prompt_template.format(code=code)
        response = self.forward(prompt)

        response_dict = ut.extract_html_tags(response)
        return response_dict
        # Prepare the result dictionary
        # result = {
        #     "validation_done": True,
        #     "valid": False,
        #     "function_name": "",
        #     "explanation": "",
        # }

        # # Update the result based on the response
        # if "valid" in response_dict:
        #     result["valid"] = response_dict["valid"].lower() == "yes"
        # if "function_name" in response_dict:
        #     result["function_name"] = response_dict["function_name"]
        # if "explanation" in response_dict:
        #     result["explanation"] = response_dict["explanation"]

        if verbose:
            print(f"Code validity: {result['valid']}")
            if result["valid"]:
                print(f"Function name: {result['function_name']}")
            print(f"Explanation: {result['explanation']}")

        return result

    def check_cell_validity(
        self, meta, meta_fname="agentpoirot/tools/learnt/meta.json", verbose=False
    ):
        """
        Check the validity of code cells in the meta dictionary and update their status.

        Args:
            meta (dict): Dictionary containing cell information.
            meta_fname (str, optional): Path to the meta.json file.
                Defaults to "agentpoirot/tools/learnt/meta.json".
            verbose (bool, optional): If True, print verbose output. Defaults to False.

        Returns:
            dict: Updated meta dictionary with validity information.
        """
        for code_hash, cell_info in meta.items():
            notebook_fname = cell_info["notebook_fname"]
            cell_id = cell_info["cell_id"]
            cell_validation_done = cell_info["validation_done"]

            code = self.get_code_from_notebook(notebook_fname, cell_id)
            if (not cell_validation_done) and (code is not None):
                validity_result = self.verify_code_validity(code)

                cell_info["validation_done"] = validity_result.get(
                    "validation_done", True
                )
                cell_info["valid"] = validity_result.get("valid", False)
                cell_info["function_name"] = validity_result.get("function_name", "")
                cell_info["explanation"] = validity_result.get("explanation", "")

                if verbose:
                    print(
                        f"Checked cell {cell_id} in {notebook_fname}: Valid={cell_info['valid']}"
                    )

        with open(meta_fname, "w") as f:
            json.dump(meta, f, indent=2)

        return meta

    def get_code_from_notebook(self, notebook_fname, cell_id, verbose=False):
        """
        Retrieve the code from a specific cell in a Jupyter notebook.

        Args:
            notebook_fname (str): Path to the Jupyter notebook file.
            cell_id (int): ID of the cell to retrieve.
            verbose (bool, optional): If True, print verbose output. Defaults to False.

        Returns:
            str or None: The code from the specified cell if found, None otherwise.
        """
        try:
            with open(notebook_fname, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            if 0 <= cell_id < len(nb.cells):
                cell = nb.cells[cell_id]
                if cell.cell_type == "code":
                    if verbose:
                        print(
                            f"Successfully retrieved code from cell {cell_id} in {notebook_fname}"
                        )
                    return "".join(cell.source)
                elif verbose:
                    print(f"Cell {cell_id} in {notebook_fname} is not a code cell")
            else:
                if verbose:
                    print(f"Invalid cell_id: {cell_id} for notebook: {notebook_fname}")
        except Exception as e:
            if verbose:
                print(f"Error reading notebook {notebook_fname}: {str(e)}")

        return None

    def index_all_cells_notebooks(
        self,
        notebook_fname,
        meta_fname="agentpoirot/tools/learnt/meta.json",
        verbose=False,
    ):
        """
        Index all code cells in a Jupyter notebook and update the meta.json file.

        Args:
            notebook_fname (str): Path to the Jupyter notebook file.
            meta_fname (str, optional): Path to the meta.json file.
                Defaults to "agentpoirot/tools/learnt/meta.json".
            verbose (bool, optional): If True, print verbose output. Defaults to False.

        Returns:
            dict: Updated meta dictionary containing indexed cell information.
        """
        try:
            with open(meta_fname, "r") as f:
                meta = json.load(f)
        except FileNotFoundError:
            meta = {}

        notebook = ut.read_notebook(notebook_fname)
        for cell_id, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                code = cell["source"]
                code_hash = ut.hash_str(code)

                if code_hash not in meta:
                    meta[code_hash] = {
                        "cell_id": cell_id,
                        "notebook_fname": notebook_fname,
                        "valid": False,
                        "validation_done": False,
                        "code": code,
                    }
                    if verbose:
                        print(f"New cell indexed: {code_hash}")
                elif verbose:
                    print(f"Cell already indexed: {code_hash}")

        with open(meta_fname, "w") as f:
            json.dump(meta, f, indent=2)

        notebook_dict = meta
        return notebook_dict

    def display_insights(self, insight_list):
        insight_str = ""
        for i, insight_dict in enumerate(insight_list):
            formatted_insight = self.format_insight_dict(insight_dict)
            insight_str += f"\n# Insight {i+1}\n{formatted_insight}\n"
            insight_str += "\n" + "=" * 50 + "\n"  # Separator between insights
        return insight_str

    def format_insight_dict(self, insight_dict):
        """
        Formats an insight dictionary into a readable string format.

        Args:
            insight_dict (dict): Dictionary containing insight information

        Returns:
            str: Formatted string representation of the insight
        """
        template = """## Header
-----------
{header}

## Question
-----------
{question}

## Answer
-----------
{answer}

## Insight
-----------
{insight}

## Severity
-----------
{severity}

## Follow up
-----------
{followup}"""

        return template.format(
            header=insight_dict.get("header", "N/A"),
            question=insight_dict.get("question", "N/A"),
            answer=insight_dict.get("answer", "N/A"),
            insight=insight_dict.get("insight", "N/A"),
            indicator=insight_dict.get("indicator", "N/A"),
            severity=insight_dict.get("severity", "N/A"),
            followup=insight_dict.get("followup", "N/A"),
        )

    def get_insights(self, questions, with_pdf=False):
        insights = []
        for q in questions:
            insight_dict = self.answer_question(q, with_pdf=with_pdf)
            insights.append(insight_dict)

        if self.verbose:
            print("\nPredicted Insights:")
            print("=" * 20)
            print(self.display_insights(insights))
            print("=" * 20)
        return insights

    def compute_score(self, pred_insights, golden_insights):
        prompt_template = ut.load_txt(get_prompt_path("evaluate.txt"))
        prompt = prompt_template.format(
            pred_insights=pred_insights, golden_insights=golden_insights, meta=self.meta
        )
        response = self.forward(prompt)
        response_dict = ut.extract_html_tags(response)

        if self.verbose:
            for i, (p, g) in enumerate(zip(pred_insights, golden_insights)):
                print("\nInsights:")
                print("=" * 20)
                print("Predicted:")
                print(textwrap_str(p["insight"]))
                print()
                print("Golden:")
                print(textwrap_str(g))
                print("=" * 20)

        return str(response_dict["rating"])


def textwrap_str(text):
    return "\n".join(textwrap.wrap(text, 50))


def get_prompt_path(prompt_fname):
    fname = os.path.join(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0],
        f"prompts",
        prompt_fname,
    )
    assert os.path.exists(fname), f"Prompt file {fname} not found"
    return fname


def get_first_function_name(file_path: str) -> str:
    """
    Extracts the name of the first function defined in a Python script.

    Args:
        file_path (str): Path to the Python file to be parsed.

    Returns:
        str: Name of the first function found in the file, or None if no function is found.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name

    return None


def rename_file_to_function_name(file_path: str) -> None:
    """
    Renames the given file to the name of the first function defined in it.

    Args:
        file_path (str): Path to the file to be renamed.
    """
    function_name = get_first_function_name(file_path)
    if function_name:
        new_file_path = f"agentpoirot/tools/learnt/{function_name}.py"
        os.rename(file_path, new_file_path)
        print(f"File renamed to: {new_file_path}")
    else:
        print("No function definition found in the file.")
    return function_name
