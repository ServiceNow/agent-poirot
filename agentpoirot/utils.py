import hashlib, os, json, pprint
import os
import numpy as np, pandas as pd, time, re, json, os, shutil, inspect

import contextlib
import io, ast
import os
import re
import subprocess
import traceback
import sys

from io import StringIO
from pathlib import Path
from copy import deepcopy
from typing import Dict

from dateutil.parser import parse
from warnings import warn


from pathlib import Path
import matplotlib.pyplot as plt
import textwrap
import nbformat


JSON_MAX_TOKENS = 5000
JSON_MAX_CHARS = 99999999999


def get_available_functions(question=None, top_k=5):
    """
    return top k functions that are most relevant to the question
    """
    from agentpoirot.tools import func_tools as tools

    available_functions = [
        func_name
        for func_name, obj in inspect.getmembers(tools)
        if inspect.isfunction(obj)
    ]
    # question_embedding = create_embedding(text=question)

    def get_similarity(function_doc):
        """
        Return the similarity between the question and the function_doc
        """
        return 1.0
        function_embedding = create_embedding(text=function_doc)
        return cosine_similarity([question_embedding], [function_embedding])

        # return cosine_similarity([question], [function_doc])

    function_docs = []
    for func_name in available_functions:
        function_doc = (
            f"{func_name}{inspect.signature(getattr(tools, func_name))}:\n{inspect.getdoc(getattr(tools, func_name))}\n"
            + "=" * 20
            + "\n"
        )
        score = get_similarity(function_doc)
        function_docs += [{"doc": function_doc, "score": score}]
    # sort based on highest score
    function_docs = sorted(function_docs, key=lambda x: x["score"], reverse=True)

    # Get only top k
    function_docs_topk = function_docs[:top_k]

    # Get only the docs and join them
    function_docs_list = [doc["doc"] for doc in function_docs_topk]
    function_docs_str = "\n".join(function_docs_list)

    return function_docs_str


def dict_to_string(d):
    return "\n".join(f"<{k}>{v}</{k}>" for k, v in d.items())


def list_to_string(key, values):
    return "".join(
        f"<{key} {i + 1}>\n<{value}>\n</{key} {i + 1}>\n"
        for i, value in enumerate(values)
    )


def extract_python_code_blocks(text):
    """
    Extract and merge Python code blocks from a given text string.

    The function identifies code blocks that start with ``` or ```python and end with ```.
    After extracting the code blocks, it removes the start and end delimiters (```, ```python),
    and merges the code blocks together into a single string.

    Parameters
    ----------
    text : str
        The input string from which Python code blocks need to be extracted.

    Returns
    -------
    str
        A string containing the merged Python code blocks stripped of leading and trailing whitespaces.
        Code blocks are separated by a newline character.

    """

    code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
    return "\n".join(block.strip() for block in code_blocks)

    def generate_report(self, filename="report.pdf"):
        if self.insights_history is None:
            return "No insights generated yet. Please generate insights first."

        # generate a table preview
        ut.df_to_image(
            self.tables[0].head(10), os.path.join(self.savedir, "table_preview.png")
        )

        # Create instance of FPDF class
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Add a page
        pdf.add_page()

        # Set title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Data Analysis Report", ln=True, align="C")
        pdf.ln(10)  # Line break

        # Set font for content
        pdf.set_font("Arial", size=12)
        # Add content from the insights dictionary
        for title in ["dataset_description", "insights_summary", "recommended_action"]:
            content = self.output_dict[title]
            # capitalize the title
            title = " ".join([w.capitalize() for w in title.split("_")])
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, txt=title, ln=True, align="L")
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, txt=content)
            if title == "Dataset Description":
                # compute x and y for the image
                image_width = (pdf.w - 20) * 1.5
                x_position = (pdf.w - image_width) / 2  # Center the image
                pdf.image(
                    os.path.join(self.savedir, "table_preview.png"),
                    x=x_position,
                    w=image_width,
                )
            pdf.ln(5)  # Line break after each insight

        # Start a new page
        pdf.add_page()
        # now add the insights to the report
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Detailed Insights", ln=True, align="C")
        pdf.ln(10)

        for insight in self.output_dict["insights"]:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(200, 10, txt=insight["question"], ln=True, align="L")
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, txt=insight["answer"])
            pdf.multi_cell(0, 5, txt="Justification: " + insight["justification"])

            # Add the plot below the text
            if insight["plot"] is not None:
                pdf.ln(5)  # Add a small gap before the image
                # Calculate x position to center the image
                image_width = (
                    pdf.w - 20
                ) * 0.5  # Width of the image considering page margins
                x_position = (pdf.w - image_width) / 2  # Center the image

                pdf.image(
                    insight["plot"],
                    x=x_position,  # Center the image
                    w=image_width,  # Width of the image considering page margins
                )

            pdf.ln(10)  # Add space after each insight

        # Save the PDF
        pdf.output(os.path.join(self.savedir, filename))
        print(f"Report saved as {os.path.join(self.savedir, filename)}")


class PythonREPL:
    """
    Simulates a standalone Python REPL.

    TODO add a way to pass a random seed to the REPL
    """

    def __init__(self):
        self.history = []

    def run(self, command: str, workdir: str = None) -> str:
        """Run command with own globals/locals and returns anything printed."""

        if workdir is not None:
            old_cwd = Path.cwd()
            os.chdir(workdir)

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                exec(command, locals())
                valid = True
                retry_message = ""
                self.history.append((command, workdir))
            except Exception as e:
                valid = False
                retry_message = traceback.format_exc() + "\n" + str(e)
            finally:
                if workdir is not None:
                    os.chdir(old_cwd)
        output = buffer.getvalue()

        return output, valid, retry_message

    def clone(self):
        """Clone the REPL from history.

        it is not possible to clone the REPL from the globals/locals because they
        may contain references to objects that cannot be pickled e.g. python modules.
        Instead, we clone the REPL by replaying the history.
        """
        new_repl = PythonREPL()
        # deepcopy of history
        new_repl.history = deepcopy(self.history)

        for command, workdir in self.history:
            new_repl.run(command, workdir=workdir)

        return new_repl


def _execute_command(args):
    """Execute a command and return the stdout, stderr and return code

    Parameters
    ----------
    args : list of str or str, directly passed to subprocess.Popen

    Returns
    -------
    stdout : str
        stdout of the command
    stderr : str
        stderr of the command
    returncode : int
        return code of the command
    """
    try:
        process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
        )
    except FileNotFoundError as e:
        return "", str(e), 1

    stdout, stderr = process.communicate()

    # decode bytes object to string
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    return stdout, stderr, process.returncode


def pip_install(requirements):
    """Install a list of requirements using pip

    Parameters
    ----------
    requirements : list of str
        List of requirements to install, should be in the format of pip install <requirement>

    Returns
    -------
    stdout : str
        stdout of the command
    valid : bool
        True if the installation was successful, False otherwise
    retry_message : str
        Error message if the installation was not successful
    """

    if isinstance(requirements, str):
        requirements = [requirements]
    retry_messages = []
    stdouts = []
    for req in requirements:
        stdout, stderr, code = _execute_command(["pip", "install", req])
        stdouts.append(stdout)
        if stdout.strip().startswith("Usage:"):
            retry_messages.append(
                f"Seems like there is an error on the pip commandline, it just prints usage. stderr:\n{stderr}. stdout:\n{stdout[-1000:]}"
            )
        if code != 0:
            retry_messages.append(
                f"Error code {code} when installing {req}. stderr:\n{stderr}. stdout:\n{stdout[-1000:]}"
            )

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return "\n".join(stdouts), valid, retry_message


# =============================================================================
# Code generation
# =============================================================================
def _code_parser(code, output_folder):
    """
    A parser that is used to parse the code generated by the LLM
    and determine whether it is acceptable or not

    """
    # Clean output folder
    output_folder = Path(output_folder)
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True)

    # Extract code blocks from the input code (might contain other text)
    code_block = extract_python_code_blocks(code)
    if len(code_block) == 0:
        # No code blocks detected so input is likely already raw code
        code_block = code

    with open(f"{output_folder}/code.py", "w") as file:
        file.write(code_block)

    # Run code and report any errors
    output, valid, retry_message = PythonREPL().run(code_block, workdir=output_folder)
    if not valid:
        return "", valid, retry_message

    return output, True, ""


def replace_large_lists(data, max_size=50):
    if isinstance(data, dict):
        # If the item is a dictionary, recurse into it
        for key, value in data.items():
            data[key] = replace_large_lists(value, max_size)
    elif isinstance(data, list):
        # If the item is a list and has more than max_size elements, return None
        if len(data) > max_size:
            return None
        else:
            # Otherwise, recurse into each element of the list
            return [replace_large_lists(item, max_size) for item in data]
    return data


def save_txt(filename, content):
    with open(filename, "w") as file:
        file.write(content)


def get_docstrings(directory):
    # Initialize an empty string to store all information
    all_docstrings = ""

    # Loop over each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            file_path = os.path.join(directory, filename)

            # Open and read the content of the file
            with open(file_path, "r") as file:
                file_content = file.read()

            # Parse the file content using the ast module
            tree = ast.parse(file_content)

            # Add filename header to the string
            all_docstrings += f"\nFunctions and docstrings in {filename}:\n{'-'*40}\n"

            # Loop over each node in the abstract syntax tree
            for node in ast.walk(tree):
                # Check if the node is a function definition
                if isinstance(node, ast.FunctionDef):
                    # Get the function name
                    func_name = node.name
                    # Get the docstring (if present)
                    docstring = ast.get_docstring(node)

                    # Append function name and docstring to the string
                    all_docstrings += f"Function: {func_name}\n"
                    if docstring:
                        all_docstrings += f"Docstring: {docstring}\n"
                    else:
                        all_docstrings += "Docstring: None\n"
                    all_docstrings += "-" * 40 + "\n"
    return all_docstrings


def retry_on_parsing_error(
    llm,
    initial_prompt,
    parser,
    n_retries,
    exception_on_max_retries=True,
):
    """
    Try querying a LLM until it returns a valid value with a maximum number of retries.

    Parameters:
    -----------
    llm : callable
        A langchain LLM model.
    initial_prompt : str
        The initial prompt to send to the LLM.
    parser : callable
        A function taking a message and returning a tuple (value, valid, retry_message),
        where retries will be made until valid is True.
    n_retries : int
        The maximum number of retries.
    exception_on_max_retries : bool
        If True, raise an exception if the maximum number of retries is reached.
        Otherwise, returns "".

    Returns:
    --------
    value : str
        The value returned by the LLM.
    completions : list
        The attempts made by the LLM.

    """
    prompt_name = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "prompts/_retry.txt",
    )
    retry_template = load_txt(prompt_name)
    prompt = initial_prompt

    completions = []
    for i in range(n_retries + 1):  # Add one since the initial prompt is not a retry
        # Try to get a valid completion
        completions.append({"code": llm(prompt), "prompt": prompt})
        output, valid, retry_message = parser(completions[-1]["code"])

        # If parser execution succeeds return the output
        if valid:
            return output, completions

        # If parser execution fails, produce a new prompt that includes the previous output and the error message
        warn(
            f"Retry {i+1}/{n_retries} - Query failed with error: {retry_message}",
            RuntimeWarning,
        )
        prompt = retry_template.format(
            initial_prompt=initial_prompt,
            prev_output=completions[-1]["code"],
            error=retry_message,
        )

    if exception_on_max_retries:
        return f"Could not parse a valid value after {n_retries} retries.", [
            "```python\nimport pandas as pd```",
            "```python\nimport numpy as np```",
        ]
    else:
        return retry_message, completions


def _extract_top_values(values, k=5, max_str_len=100):
    """
    Extracts the top k values from a pandas series

    Parameters
    ----------
    values : pandas.Series
        Series to extract top values from
    k : int, optional
        Number of top values to extract, by default 5
    max_str_len : int, optional
        Maximum length of string values (will be truncated), by default 100

    """
    top = values.value_counts().iloc[:k].index.values.tolist()
    top = [x if not isinstance(x, str) else x[:max_str_len] for x in top]
    return top


def get_schema(df):
    """
    Extracts schema from a pandas dataframe

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to extract schema from

    Returns
    -------
    list of dict
        Schema for each column in the dataframe

    """
    schema = []

    for col in df.columns:
        info = {
            "name": col,
            "type": df[col].dtype,
            "missing_count": df[col].isna().sum(),
            "unique_count": df[col].unique().shape[0],
        }

        # If the column is numeric, extract some stats
        if np.issubdtype(df[col].dtype, np.number):
            info["min"] = df[col].min()
            info["max"] = df[col].max()
            info["mean"] = df[col].mean()
            info["std"] = df[col].std()
        # If the column is a date, extract the min and max
        elif _is_date(df[col].iloc[0]):
            info["min"] = df[col].dropna().min()
            info["max"] = df[col].dropna().max()
        # If the column is something else, extract the top values
        else:
            info["top5_unique_values"] = _extract_top_values(df[col])

        schema.append(info)
    schema = schema_to_str(schema)
    df_small = df.head()[:2]

    # Initialize the result string
    result_str = ""

    # Build the string representation
    for column in df_small.columns:
        result_str += f"Column '{column}' examples\n"
        for example in df_small[column].unique():
            result_str += f"    {example}\n"
        result_str += "\n"

    schema = schema + "\n\n" + result_str
    return schema


def truncate_text(text, width=20):
    if isinstance(text, str):
        wrapped_lines = textwrap.wrap(text, width=width)
        if len(wrapped_lines) > 1:
            return f"{wrapped_lines[0]}..."
        return wrapped_lines[0]
    return text


def df_to_image(df, filename):
    # Wrap text for each cell in the DataFrame
    wrapped_df = df.map(lambda x: truncate_text(x, width=20))

    # Create a figure and a subplot
    fig, ax = plt.subplots(
        figsize=(len(df.columns) * 2, len(df) * 0.5)
    )  # Adjust size as needed
    ax.axis("tight")
    ax.axis("off")

    # Calculate column widths based on the maximum text length in each column
    col_widths = [
        max(len(str(x)) for x in wrapped_df[col]) * 1.2 for col in wrapped_df.columns
    ]

    # Create a table and add it to the plot
    table = ax.table(
        cellText=wrapped_df.values,
        colLabels=wrapped_df.columns,
        cellLoc="center",
        loc="center",
    )

    # Adjust the column widths
    for i, width in enumerate(col_widths):
        table.auto_set_column_width([i])
        cells = table.get_celld()
        for (row, col), cell in cells.items():
            if col == i:
                cell.set_width(width / fig.dpi)  # Convert width to figure size (inches)

    # Increase row height
    for (row, col), cell in table.get_celld().items():
        if row >= 0:  # Skip the header row
            cell.set_height(0.1)  # Set row height (adjust as needed)

    # Adjust font size and scaling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    # Save the figure as an image file
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)


def schema_to_str(schema) -> str:
    """Converts the list of dict to a promptable string.

    Parameters
    ----------
    schema : list of dict
        Schema for each column in the dataframe

    Returns
    -------
    str
        String representation of the schema
    """
    schema_str = ""
    for col in schema:
        schema_str += f"Column: {col['name']} ({col['type']})\n"
        for key, val in col.items():
            if key in ["name", "type"]:
                continue
            schema_str += f"  {key}: {val}\n"
    return schema_str


def _is_date(string):
    """
    Checks if a string is a date

    Parameters
    ----------
    string : str
        String to check

    Returns
    -------
    bool
        True if the string is a date, False otherwise

    """
    try:
        parse(str(string))
        return True
    except ValueError:
        return False


def read_notebook(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    return nb


def schema_to_str(schema) -> str:
    """Converts the list of dict to a promptable string.

    Parameters
    ----------
    schema : list of dict
        Schema for each column in the dataframe

    Returns
    -------
    str
        String representation of the schema
    """
    schema_str = ""
    for col in schema:
        schema_str += f"Column: {col['name']} ({col['type']})\n"
        for key, val in col.items():
            if key in ["name", "type"]:
                continue
            schema_str += f"  {key}: {val}\n"
    return schema_str


# Function to convert columns to float if they are numeric or boolean
def clean_table(df):
    for col in df.columns:
        original_col = df[col].copy()  # Store the original column
        if df[col].dtype == "bool":  # Specifically handle boolean columns
            df[col] = df[col].astype(float)
        elif df[col].dtype in ["int64", "float64", "object"]:
            try:
                # Attempt to convert the column to float
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
                # If conversion results in too many NaNs, revert to the original column
                if df[col].isna().sum() > 0.5 * len(df):
                    df[col] = original_col
            except ValueError:
                # If conversion fails, the column will remain unchanged
                pass
    # Remove duplicate rows
    # df = df.drop_duplicates()

    # Drop columns with more than 70% NaN values
    threshold = 0.7 * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    return df


def convert_messages_to_text(messages):
    """
    Convert a list of messages to a string

    Parameters
    ----------
    messages : list
        List of messages to convert

    Returns
    -------
    str
        String representation of the messages

    """
    return "\n".join(
        [
            (
                f"[INST]\n{m.content}\n[/INST]"
                if m.type in ["system", "agent"]
                else f"\n{m.content}\n"
            )
            for m in messages
        ]
    )


def parse_questions(data: str):
    # Regular expression to match each <container> block
    containers = re.findall(r"<container>(.*?)</container>", data, re.DOTALL)

    # List to hold the dictionaries
    container_dicts = []

    # Loop through each container and extract all tags using extract_html_tags
    for container in containers:
        content = extract_html_tags(container)
        container_dicts.append(content)

    return container_dicts


def extract_html_tags(text, keys=None):
    """Extract the content within HTML tags for a list of keys.

    If keys are None, extract all tags found within the text.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str, optional
        The HTML tags to extract the content from. If None, all tags are extracted.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.
    """
    content_dict = {}

    if keys is None:
        # Find all unique tags in the text
        keys = set(re.findall(r"<(.*?)>", text))

    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
            if len(content_dict[key]) == 1:
                content_dict[key] = content_dict[key][0]

    return content_dict


def parse_insight(output):
    """
    A parser that makes sure that the human readable insight is produced in the correct format

    """
    try:
        insight_dict = extract_html_tags(output, None)
    except ValueError as e:
        return (
            "",
            False,
            f"The following error occured while extracting the value for the <justification> tag: {str(e)}",
        )

    return insight_dict, True, ""


def _build_insight_prompt(solution) -> str:
    """
    Gather all plots and statistics produced by the model and format then nicely into text

    """
    insight_prompt = ""
    for i, var in enumerate(solution["vars"]):
        insight_prompt += f"<insight id='{i}'>"
        insight_prompt += f"    <stat>"
        insight_prompt += f"        <name>{var['stat'].get('name', 'n/a')}</name>"
        insight_prompt += f"        <description>{var['stat'].get('description', 'n/a')}</description>"
        stat_val = var["stat"].get("value", "n/a")
        stat_val = stat_val[:50] if isinstance(stat_val, list) else stat_val
        insight_prompt += f"        <value>{stat_val}</value>"
        insight_prompt += f"    </stat>"
        insight_prompt += f"    <plot filename='{var['plot']['name']}'>"
        insight_prompt += f"        <xaxis>"
        insight_prompt += f"            <description>{var['x_axis'].get('description', 'n/a')}</description>"
        x_val = var["x_axis"].get("value", "n/a")
        x_val = x_val[:50] if isinstance(x_val, list) else x_val
        insight_prompt += f"            <value>{x_val}</value>"
        insight_prompt += f"        </xaxis>"
        insight_prompt += f"        <yaxis>"
        insight_prompt += f"            <description>{var['y_axis'].get('description', 'n/a')}</description>"
        y_val = var["y_axis"].get("value", "n/a")
        y_val = y_val[:50] if isinstance(y_val, list) else y_val
        insight_prompt += f"            <value>{y_val}</value>"
        insight_prompt += f"        </yaxis>"
        insight_prompt += f"    </plot>"
        insight_prompt += f"</insight>"
    return insight_prompt


# def get_insights(
#     context,
#     goal,
#     messages=[],
#     schema=None,
#     max_questions=3,
#     model_name="gpt-4o",
#     temperature=0,
# ):

#     chat = get_chat_model(model_name, temperature)

#     prompt = prompts.GET_INSIGHTS_TEMPLATE
#     messages = [
#         SystemMessage(content=prompts.GET_INSIGHTS_SYSTEM_MESSAGE),
#         HumanMessage(
#             content=prompt.format(
#                 context=context, goal=goal, schema=schema, max_questions=max_questions
#             )
#         ),
#     ]

#     def _validate_tasks(out):
#         isights = extract_html_tags(out, ["insight"])

#         # Check that there are insights generated
#         if "insight" not in isights:
#             return (
#                 out,
#                 False,
#                 f"Error: you did not generate insights within the <insight></insight> tags.",
#             )
#         isights = isights["insight"]
#         print("The insights are:", isights)
#         print("Length:", len(isights), "   Max:", max_questions)
#         return (isights, out), True, ""

#     insights, message = chat_and_retry(
#         chat, messages, n_retry=3, parser=_validate_tasks
#     )

#     return insights


def get_questions(
    prompt_method,
    meta_data,
    messages=[],
    schema=None,
    max_questions=10,
    model_name="gpt-4o",
    temperature=0,
):
    if prompt_method is None:
        prompt_method = "basic"

    prompt, system = prompts.get_question_prompt(method=prompt_method)

    chat = get_chat_model(model_name, temperature)

    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=prompt.format(
                context=context, goal=goal, schema=schema, max_questions=max_questions
            )
        ),
    ]

    def _validate_tasks(out):
        questions = extract_html_tags(out, ["question"])
        if "question" not in questions:
            return (
                out,
                False,
                f"Error: you did not generate questions within the <question></question> tags",
            )
        questions = questions["question"]
        # Check that there are at most max_questions questions
        if len(questions) > max_questions:
            return (
                out,
                False,
                f"Error: you can only ask at most {max_questions} questions, but you asked {len(questions)}.",
            )

        return (questions, out), True, ""

    questions, message = chat_and_retry(
        chat, messages, n_retry=3, parser=_validate_tasks
    )

    return questions


# def get_dataset_description(
#     prompt,
#     system,
#     context,
#     goal,
#     messages=[],
#     schema=None,
#     model_name="gpt-4o",
#     temperature=0,
# ):

#     chat = get_chat_model(model_name, temperature)

#     messages = [
#         SystemMessage(content=system),
#         HumanMessage(content=prompt.format(context=context, goal=goal, schema=schema)),
#     ]

#     def _validate_tasks(out):
#         try:
#             questions = extract_html_tags(out, ["description"])["description"]
#         except Exception as e:
#             return (
#                 out,
#                 False,
#                 f"Error: {str(e)}",
#             )

#         return (questions, out), True, ""

#     data_description, message = chat_and_retry(
#         chat, messages, n_retry=2, parser=_validate_tasks
#     )

#     return data_description


def get_follow_up_questions(
    context,
    goal,
    question,
    answer,
    schema=None,
    max_questions=3,
    model_name="gpt-4o",
    prompt_method=None,
    question_type="descriptive",
    temperature=0,
):
    if prompt_method is None:
        prompt_method = "follow_up"

    prompt, system = prompts.get_question_prompt(method=prompt_method)
    chat = get_chat_model(model_name, temperature)

    if prompt_method == "follow_up_with_type":
        content = prompt.format(
            context=context,
            goal=goal,
            question=question,
            answer=answer,
            schema=schema,
            max_questions=max_questions,
            question_type=question_type,
        )

    else:
        content = prompt.format(
            context=context,
            goal=goal,
            question=question,
            answer=answer,
            schema=schema,
            max_questions=max_questions,
        )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=content),
    ]

    def _validate_tasks(out):
        questions = extract_html_tags(out, ["question"])["question"]
        # print("The questions are:", questions)
        # print("Length:", len(questions), "   Max:", max_questions)

        # Check that there are at most max_questions questions
        if len(questions) > max_questions:
            return (
                out,
                False,
                f"Error: you can only ask at most {max_questions} questions, but you asked {len(questions)}.",
            )

        return (questions, out), True, ""

    questions, message = chat_and_retry(
        chat, messages, n_retry=3, parser=_validate_tasks
    )

    return questions


# def select_a_question(
#     questions,
#     context,
#     goal,
#     prev_questions,
#     model_name="gpt-4o",
#     prompt_template=None,
#     system_template=None,
#     temperature=0,
# ):

#     chat = get_chat_model(model_name, temperature)

#     followup_questions_formatted = "\n".join(
#         [f"{i+1}. {q}\n" for i, q in enumerate(questions)]
#     )
#     if prev_questions:
#         prev_questions_formatted = "\n".join(
#             [f"{i+1}. {q}\n" for i, q in enumerate(prev_questions)]
#         )
#     else:
#         prev_questions_formatted = None

#     prompt = prompt_template
#     messages = [
#         SystemMessage(content=system_template),
#         HumanMessage(
#             content=prompt.format(
#                 context=context,
#                 goal=goal,
#                 prev_questions_formatted=prev_questions_formatted,
#                 followup_questions_formatted=followup_questions_formatted,
#             )
#         ),
#     ]

#     def _validate_tasks(out):
#         question_id = extract_html_tags(out, ["question_id"])["question_id"][0]
#         # Check that there are at most max_questions questions
#         if int(question_id) >= len(questions):
#             return (
#                 out,
#                 False,
#                 f"Error: selected question index should be between 0-{len(questions)-1}.",
#             )
#         return (int(question_id), out), True, ""

#     question_id, message = chat_and_retry(
#         chat, messages, n_retry=3, parser=_validate_tasks
#     )
#     return question_id


# def generate_code(
#     schema,
#     user_schema,
#     goal,
#     question,
#     database_path,
#     user_database_path,
#     output_folder,
#     n_retries,
#     prompt_method=None,
#     model_name="gpt-4o",
#     temperature=0,
# ):
#     """
#     Solve a task using the naive single step approach

#     See main function docstring for more details

#     """
#     prompt_template = prompts.get_code_prompt(method=prompt_method)

#     available_functions = [
#         func_name
#         for func_name, obj in inspect.getmembers(tools)
#         if inspect.isfunction(obj)
#     ]
#     function_docs = []
#     for func_name in available_functions:
#         function_docs.append(
#             f"{func_name}{inspect.signature(getattr(tools, func_name))}:\n{inspect.getdoc(getattr(tools, func_name))}\n"
#             + "=" * 20
#             + "\n"
#         )
#     function_docs = "\n".join(function_docs)

#     # instantiate llm model
#     llm = get_chat_model(model_name, temperature)

#     # create prompt
#     if user_schema is None:
#         prompt = PromptTemplate(
#             template=prompt_template,
#             input_variables=[
#                 "goal",
#                 "schema",
#                 "question",
#                 "database_path",
#                 "function_docs",
#             ],
#         )
#     else:
#         prompt = PromptTemplate(
#             template=prompt_template,
#             input_variables=[
#                 "goal",
#                 "schema",
#                 "question",
#                 "database_path",
#                 "function_docs",
#                 "user_schema",
#                 "user_database_path",
#             ],
#         )

#     # Run the retry on error function
#     if user_schema is None:
#         output, completions = retry_on_parsing_error(
#             llm=llm,
#             initial_prompt=prompt.format(
#                 goal=goal,
#                 schema=schema,
#                 question=question,
#                 database_path=database_path,
#                 function_docs=function_docs,
#             ),
#             parser=partial(_code_parser, output_folder=output_folder),
#             n_retries=n_retries,
#             exception_on_max_retries=False,
#         )
#     else:
#         output, completions = retry_on_parsing_error(
#             llm=llm,
#             initial_prompt=prompt.format(
#                 goal=goal,
#                 schema=schema,
#                 question=question,
#                 database_path=database_path,
#                 function_docs=function_docs,
#                 user_schema=user_schema,
#                 user_database_path=user_database_path,
#             ),
#             parser=partial(_code_parser, output_folder=output_folder),
#             n_retries=n_retries,
#             exception_on_max_retries=False,
#         )

#     # Create the output dict
#     # Then, iterate over all generated plots and add them to the output dict
#     output_dict = {
#         "code": completions[-1],
#         "prompt": str(prompt),
#         "code_output": output,
#         "message": output,
#         "n_retries": len(completions) - 1,
#         "goal": goal,
#         "question": question,
#         "vars": [],
#     }

#     # write code to a file
#     with open(f"{output_folder}/code.py", "w") as file:
#         # use regex to capture the python code block
#         code = completions[-1]
#         try:
#             code = re.findall(r"```python(.*?)```", code, re.DOTALL)[0]
#             file.write(code.strip())
#         except Exception as e:
#             print(f"Failed to write code", e)
#             file.write(code.strip())

#     # Try to load the model's output files
#     # TODO: We should detect errors in such files and trigger a retry
#     try:
#         stat = json.load(open(f"{output_folder}/stat.json", "r"))
#     except Exception as e:
#         print(f"Failed to load {output_folder}/stat.json", e)
#         stat = {}
#     try:
#         x_axis = json.load(open(f"{output_folder}/x_axis.json", "r"))
#     except Exception as e:
#         print(f"Failed to load {output_folder}/x_axis.json", e)
#         x_axis = {}
#     try:
#         y_axis = json.load(open(f"{output_folder}/y_axis.json", "r"))
#     except Exception as e:
#         print(f"Failed to load {output_folder}/y_axis.json", e)
#         y_axis = {}

#     # Add the plot to the final output dict
#     plot_path = f"{output_folder}/plot.jpg"
#     stat["type"] = "stat"
#     x_axis["type"] = "x_axis"
#     y_axis["type"] = "y_axis"
#     plot_dict = {"name": plot_path, "type": "plot"}
#     output_dict["vars"] += [
#         {
#             "stat": stat,
#             "x_axis": x_axis,
#             "y_axis": y_axis,
#             "plot": plot_dict,
#         }
#     ]

#     return output_dict


# def root_depth_to_prompt(
#     history: Dict,
#     root: int,
#     depth: int,
#     goal: str,
#     csv_path: str,
#     results_dir: str,
# ) -> Dict:
#     """Get question from a given depth in the history and convert it to the prompt format"""
#     context = "This is a dataset of ServiceNow incidents that contains different types of failure categories"
#     import prompts

#     node = f"{root}{depth}"
#     node_output = history[node]
#     # extract question being asked at this node
#     question = node_output["question"]
#     # extract the data_df
#     data_df = pd.read_csv(csv_path)
#     prompts_dict = {}
#     # now, reconstruct the prompts used in different stages at this node
#     if root == 0 and depth == 0:
#         get_questions_prompt = prompts.GET_QUESTIONS_TEMPLATE
#         get_questions_system = prompts.GET_QUESTIONS_SYSTEM_MESSAGE
#         messages = [
#             SystemMessage(content=get_questions_system),
#             HumanMessage(
#                 content=get_questions_prompt.format(
#                     context=context,
#                     goal=goal,
#                     schema=get_schema(data_df),
#                     max_questions=3,
#                 )
#             ),
#         ]
#         prompts_dict["get_questions"] = messages
#     else:
#         prompts_dict["get_questions"] = None

#     # prompt for generating code
#     code_prompt = prompts.GENERATE_CODE_TEMPLATE
#     available_functions = [
#         func_name
#         for func_name, obj in inspect.getmembers(tools)
#         if inspect.isfunction(obj)
#     ]
#     function_docs = []
#     for func_name in available_functions:
#         function_docs.append(
#             f"{func_name}{inspect.signature(getattr(tools, func_name))}:\n{inspect.getdoc(getattr(tools, func_name))}\n"
#             + "=" * 20
#             + "\n"
#         )
#     function_docs = "\n".join(function_docs)
#     template = prompts.GENERATE_CODE_TEMPLATE
#     code_prompt = PromptTemplate(
#         template=template,
#         input_variables=[
#             "goal",
#             "schema",
#             "question",
#             "database_path",
#             "function_docs",
#         ],
#     )
#     schema = get_schema(data_df)
#     prompts_dict["generate_code"] = code_prompt.format(
#         goal=goal,
#         schema=schema,
#         question=question,
#         database_path=csv_path,
#         function_docs=function_docs,
#     )

#     output_folder = os.path.join(results_dir, node)
#     solution_dict = {
#         "code": open(os.path.join(output_folder, "code.py")).read(),
#         "prompt": str(code_prompt),
#         "code_output": "N/A",
#         "message": "N/A",
#         "n_retries": 3,
#         "goal": goal,
#         "question": question,
#         "vars": [],
#     }

#     # Extract the IDs of all generated plots
#     plot_ids = set(
#         [
#             os.path.splitext(f)[0].split("_")[-1]
#             for f in os.listdir(output_folder)
#             if any(w in f for w in ["plot_", "stat_", "x_axis_", "y_axis_"])
#         ]
#     )

#     for pid in plot_ids:
#         # Try to load the model's output files
#         # TODO: We should detect errors in such files and trigger a retry
#         try:
#             stat = json.load(open(f"{output_folder}/stat_{pid}.json", "r"))
#         except Exception as e:
#             print(f"Failed to load {output_folder}/stat_{pid}.json", e)
#             stat = {}
#         try:
#             x_axis = json.load(open(f"{output_folder}/x_axis_{pid}.json", "r"))
#         except Exception as e:
#             print(f"Failed to load {output_folder}/x_axis_{pid}.json", e)
#             x_axis = {}
#         try:
#             y_axis = json.load(open(f"{output_folder}/y_axis_{pid}.json", "r"))
#         except Exception as e:
#             print(f"Failed to load {output_folder}/y_axis_{pid}.json", e)
#             y_axis = {}

#         # Add the plot to the final output dict
#         plot_path = f"{output_folder}/plot_{pid}.jpg"
#         stat["type"] = "stat"
#         x_axis["type"] = "x_axis"
#         y_axis["type"] = "y_axis"
#         plot_dict = {"name": plot_path, "type": "plot"}
#         solution_dict["vars"] += [
#             {
#                 "stat": stat,
#                 "x_axis": x_axis,
#                 "y_axis": y_axis,
#                 "plot": plot_dict,
#             }
#         ]

#     # prompt for interpreting soln
#     interpret_prompt = prompts.INTERPRET_SOLUTION
#     insight_prompt = _build_insight_prompt(solution_dict)
#     prompts_dict["interpret_solution_prompt"] = interpret_prompt.format(
#         goal=solution_dict["goal"],
#         question=solution_dict["question"],
#         message=solution_dict["message"],
#         insights=insight_prompt,
#     )

#     # build data analysis prompt
#     data_analysis_prompt = prompts.DATA_ANALYTICS_TEMPLATE
#     da_messages = [
#         SystemMessage(content=prompts.GET_DATA_ANALYTICS_SYSTEM_MESSAGE),
#         HumanMessage(
#             content=data_analysis_prompt.format(
#                 context=context,
#                 goal=goal,
#                 schema=schema,
#                 question=question,
#                 answer=node_output["answer"]["answer"],
#                 justification=node_output["answer"]["justification"],
#                 max_questions=3,
#             )
#         ),
#     ]

#     prompts_dict["data_analysis"] = da_messages

#     # get select a follow up prompt
#     select_follow_up_prompt = prompts.SELECT_A_QUESTION_TEMPLATE
#     followup_questions_formatted = "\n".join(
#         [f"{i+1}. {q}\n" for i, q in enumerate(node_output["follow_ups"])]
#     )
#     prev_questions_formatted = "\n".join(
#         [
#             f"{i+1}. {q}\n"
#             for i, q in enumerate(
#                 output["question"]
#                 for node, output in history.items()
#                 if (int(node[0]) <= int(root) and int(node[1]) <= depth)
#             )
#         ]
#     )
#     messages = [
#         SystemMessage(content=prompts.SELECT_A_QUESTION_SYSTEM_MESSAGE),
#         HumanMessage(
#             content=select_follow_up_prompt.format(
#                 context=context,
#                 goal=goal,
#                 prev_questions_formatted=prev_questions_formatted,
#                 followup_questions_formatted=followup_questions_formatted,
#             )
#         ),
#     ]

#     prompts_dict["select_question"] = messages

#     # build the G-Eval prompt
#     curr_answer = node_output["answer"]["answer"]
#     scores_dict = json.load(open(os.path.join(results_dir, "scores.json")))
#     all_gts = [n["gt"] for gt_id, n in scores_dict.items()]

#     geval_prompt = prompts.G_EVAL_TEMPLATE
#     prompts_dict["geval_prompts"] = [
#         prompts.G_EVAL_SYSTEM_MESSAGE,
#         [geval_prompt.format(answer=curr_answer, gt_answer=gt) for gt in all_gts],
#     ]

#     return prompts_dict


def analysis_nb_to_gt(fname_notebook, include_df_head=False) -> None:
    """
    Reads all ipynb files in data_dir and parses each cell and converts it into a ground truth file.
    The ipynb files are structured as follows: code (outputs plot), then a cell with an insight dict
    """

    def _extract_metadata(nb):
        # iterate through the cells
        metadata = {}
        # extract metadata

        # extract name of the dataset from the first cell
        dname = re.findall(r"## (.+) \(Flag \d+\)", nb.cells[0].source)[0].strip()
        metadata["dataset_name"] = dname
        # extract dataset description
        description = (
            re.findall(
                r"Dataset Description(.+)Your Task", nb.cells[0].source, re.DOTALL
            )[0]
            .replace("#", "")
            .strip()
        )
        metadata["dataset_description"] = description

        # extract goal and role
        metadata["goal"] = re.findall(r"Goal\**:(.+)", nb.cells[0].source)[0].strip()
        metadata["role"] = re.findall(r"Role\**:(.+)", nb.cells[0].source)[0].strip()

        metadata["difficulty"] = re.findall(
            r"Difficulty\**: (\d) out of \d", nb.cells[0].source
        )[0].strip()
        metadata["difficulty_description"] = (
            re.findall(r"Difficulty\**: \d out of \d(.+)", nb.cells[0].source)[0]
            .replace("*", "")
            .strip()
        )
        metadata["dataset_category"] = re.findall(
            r"Category\**: (.+)", nb.cells[0].source
        )[0].strip()

        # Get Dataset Info
        tag = r"^dataset_path =(.+)"

        dataset_csv_path = None
        for cell in nb.cells:
            if cell.cell_type == "code":
                if re.search(tag, cell.source):
                    dataset_csv_path = (
                        re.findall(tag, cell.source)[0]
                        .strip()
                        .replace("'", "")
                        .replace('"', "")
                    )
                    break
        assert dataset_csv_path is not None
        metadata["dataset_csv_path"] = dataset_csv_path

        if include_df_head:
            metadata["df_head"] = pd.read_html(
                StringIO(cell.outputs[0]["data"]["text/html"])
            )

        # Get Dataset Info
        tag = r"user_dataset_path =(.+)"

        user_dataset_csv_path = None
        for cell in nb.cells:
            if cell.cell_type == "code":
                if re.search(tag, cell.source):
                    user_dataset_csv_path = (
                        re.findall(tag, cell.source)[0]
                        .strip()
                        .replace("'", "")
                        .replace('"', "")
                    )
                    break
        metadata["user_dataset_csv_path"] = user_dataset_csv_path

        # Get Summary of Findings
        tag = r"Summary of Findings \(Flag \d+\)(.+)"

        flag = None
        for cell in reversed(nb.cells):
            if cell.cell_type == "markdown":
                if re.search(tag, cell.source, re.DOTALL | re.IGNORECASE):
                    flag = (
                        re.findall(tag, cell.source, re.DOTALL | re.IGNORECASE)[0]
                        .replace("#", "")
                        .replace("*", "")
                        .strip()
                    )
                    break
        assert flag is not None
        metadata["flag"] = flag

        return metadata

    def _parse_question(nb, cell_idx):
        qdict = {}
        qdict["question"] = (
            re.findall(
                r"Question( |-)(\d+).*:(.+)", nb.cells[cell_idx].source, re.IGNORECASE
            )[0][2]
            .replace("*", "")
            .strip()
        )

        if nb.cells[cell_idx + 2].cell_type == "code":
            # action to take to answer the question
            assert nb.cells[cell_idx + 1].cell_type == "markdown"
            qdict["q_action"] = nb.cells[cell_idx + 1].source.replace("#", "").strip()
            assert nb.cells[cell_idx + 2].cell_type == "code"
            qdict["code"] = nb.cells[cell_idx + 2].source
            # extract output plot. Note that this image data is in str,
            # will need to use base64 to load this data

            qdict["plot"] = nb.cells[cell_idx + 2].outputs
            # loop as there might be multiple outputs and some might be stderr
            for o in qdict["plot"]:
                if "data" in o and "image/png" in o["data"]:
                    qdict["plot"] = o["data"]["image/png"]
                    break

            # extract the insight
            qdict["insight_dict"] = json.loads(nb.cells[cell_idx + 4].source)
        else:
            print(f"Found prescriptive insight in {fname_notebook}")
            qdict["insight_dict"] = {
                "data_type": "prescriptive",
                "insight": nb.cells[cell_idx + 1].source.strip(),
                "question": qdict["question"],
            }
        return qdict

    def _parse_notebook(nb):
        gt_dict = _extract_metadata(nb)

        # extract questions, code, and outputs
        que_indices = [
            idx
            for idx, cell in enumerate(nb.cells)
            if cell.cell_type == "markdown"
            and re.search(r"Question( |-)\d+", cell.source, re.IGNORECASE)
        ]
        gt_dict["insights"] = []
        for que_idx in que_indices:
            gt_dict["insights"].append(_parse_question(nb, que_idx))
        return gt_dict

    # Convert the notebook to a ground truth file
    if not fname_notebook.endswith(".ipynb"):
        raise ValueError("The file must be an ipynb file")
    else:
        # extract dataset id from flag-analysis-i.ipynb using re
        fname_json = fname_notebook.replace(".ipynb", ".json")

        with open(fname_notebook, "r") as f:
            notebook = nbformat.read(f, as_version=4)
        gt_dict = _parse_notebook(notebook)

    return gt_dict


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def print(string):
    pprint.pprint(string)


def hash_str(string):
    """Create a hash for a string.

    Parameters
    ----------
    string : str
        A string

    Returns
    -------
    hash_id: str
        A unique id defining the string
    """
    hash_id = hashlib.md5(string.encode()).hexdigest()
    return hash_id


def hash_dict(exp_dict):
    """Create a hash for an experiment.

    Parameters
    ----------
    exp_dict : dict
        An experiment, which is a single set of hyper-parameters

    Returns
    -------
    hash_id: str
        A unique id defining the experiment
    """
    dict2hash = ""
    if not isinstance(exp_dict, dict):
        raise ValueError("exp_dict is not a dict")

    for k in sorted(exp_dict.keys()):
        if "." in k:
            raise ValueError(". has special purpose")
        elif isinstance(exp_dict[k], dict):
            v = hash_dict(exp_dict[k])
        elif isinstance(exp_dict[k], tuple):
            raise ValueError(
                f"{exp_dict[k]} tuples can't be hashed yet, consider converting tuples to lists"
            )
        elif (
            isinstance(exp_dict[k], list)
            and len(exp_dict[k])
            and isinstance(exp_dict[k][0], dict)
        ):
            v_str = ""
            for e in exp_dict[k]:
                if isinstance(e, dict):
                    v_str += hash_dict(e)
                else:
                    raise ValueError("all have to be dicts")
            v = v_str
        else:
            v = exp_dict[k]

        dict2hash += str(k) + "/" + str(v)
    hash_id = hashlib.md5(dict2hash.encode()).hexdigest()

    return hash_id


def save_json(fname, data):
    """Save data into a json file.

    Parameters
    ----------
    fname : str
        Name of the json file
    data : [type]
        Data to save into the json file
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    # turn fname to string in case it is a Path object
    dirname = os.path.dirname(fname)
    os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def load_txt(fname):
    with open(fname, "r") as f:
        data = f.read()
    return data


def load_json(fname, decode=None):  # TODO: decode???
    """Load a json file.

    Parameters
    ----------
    fname : str
        Name of the file
    decode : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        Content of the file
    """
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d


def verify_table(df: pd.DataFrame) -> bool:
    """
    Verify the integrity of a pandas DataFrame.

    This function checks if the DataFrame is valid by ensuring it's not empty
    and has consistent row counts across all columns. It also checks for
    parsing errors that might indicate a non-standard CSV format.

    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to verify.

    Returns:
    --------
    bool
        True if the DataFrame passes all checks.

    Raises:
    -------
    ValueError
        If the DataFrame is empty, contains inconsistent row counts,
        or if there are parsing errors.
    """
    try:
        # Check if the dataframe is empty
        if df.empty:
            raise ValueError("The DataFrame is empty.")

        # Check if all columns have the same number of rows
        if not df.index.is_unique or not all(
            len(df[col]) == len(df) for col in df.columns
        ):
            raise ValueError(
                "Inconsistent number of rows detected. Not a standard 2D table."
            )

        return True  # If we've made it this far, the DataFrame is valid

    except pd.errors.EmptyDataError:
        raise ValueError("The DataFrame is empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing data. Not a standard 2D table.")


def equally_spaced(array):
    # Get 50 equally spaced indices
    indices = np.linspace(0, len(array) - 1, num=50, dtype=int)

    # Extract the values at these indices
    equally_spaced_values = [
        float(f"{s:.2f}") for s in np.array(array)[indices].tolist()
    ]
    return equally_spaced_values


def save_stats_fig(df, extra_stats=None, fig=None):
    """Saves statistics about the DataFrame columns to a JSON file."""
    stats = {}
    columns_list = []
    # remove repeated columns
    df = df.loc[:, ~df.columns.duplicated()]

    def equally_spaced(array):
        # Get 50 equally spaced indices
        indices = np.linspace(0, len(array) - 1, num=50, dtype=int)

        # Extract the values at these indices
        equally_spaced_values = np.array(array)[indices].tolist()
        return equally_spaced_values

    for col in df.columns:
        col_dict = {
            "column_name": col,
            "column_values": equally_spaced(
                df[col].astype(str).tolist()
            ),  # Ensure all values are strings
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_dict["column_stats"] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                # quantiles
                "quantile_25": float(df[col].astype("float").quantile(0.25)),
                "quantile_50": float(df[col].astype("float").quantile(0.50)),
                "quantile_75": float(df[col].astype("float").quantile(0.75)),
            }
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(
            df[col]
        ):
            col_dict["column_stats"] = {
                "unique_values": int(df[col].nunique()),
                "most_frequent": (
                    str(df[col].mode().iloc[0]) if not df[col].mode().empty else None
                ),
            }

        columns_list.append(col_dict)
    stats = {"columns": columns_list, "extra_stats": extra_stats}
    with open("stat.json", "w") as f:
        json.dump(stats, f, indent=4)
    fig.savefig("plot.jpg")
    # if fig is not None:
    #     fig.write_html("plot.html")
    #     fig.write_image("plot.jpg")


def get_all_class_docstrings_from_module():
    # Get all members of the module
    members = inspect.getmembers(module, predicate=inspect.isclass)

    # Dictionary to hold class names and their docstrings
    class_docstrings = {}

    for class_name, class_obj in members:
        # Get the docstring for the class
        docstring = (
            inspect.getdoc(class_obj)
            if inspect.getdoc(class_obj)
            else "No docstring found."
        )
        class_docstrings[class_name] = docstring

    result = []
    for class_name, docstring in class_docstrings.items():
        result.append(f"Class: {class_name}\n")
        result.append(f"Docstring:\n{docstring}\n")
        result.append("=" * 40 + "\n")

    return "".join(result)


def insight2pdf(path_to_insight_dict):
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os, copy
    from weasyprint import HTML

    insight_dict_fname = os.path.join(path_to_insight_dict, "insight_dict.json")
    insight_dict = load_json(insight_dict_fname)
    # insight_id = insight_dict["id"]
    # insight_dict["dataset"] = "csm"
    # insight_dict["model_name"] = "nowllm_mixtral_starcoder"
    # copy plot_image to current dir from insight_dict
    # import shutil

    # # ovveride
    # if os.path.exists("plot.jpg"):
    #     os.remove("plot.jpg")
    # print(insight_dict["plot_image"])
    # shutil.copy(insight_dict["plot_image"], ".")
    # # set plot image and plot html to "plot.jpg"
    # insight_dict["plot_image"] = "plot.jpg"
    # insight_dict["plot_html"] = "plot.jpg"

    # Path to the directory containing your template
    template_dir = "/mnt/home/projects/research-skilled-poirot/agentpoirot"

    # Ensure the template directory exists
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    # Create a Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )

    # Load the specific template
    template_name = "insight_card.html"
    try:
        template = env.get_template(template_name)
    except Exception as e:
        raise FileNotFoundError(
            f"Template '{template_name}' not found in '{template_dir}'"
        ) from e

    # Render the template with the dictionary
    rendered_html = template.render(insight_dict=insight_dict)

    # save the rendered_html to a file
    with open(os.path.join(path_to_insight_dict, "insight_card.html"), "w") as f:
        f.write(rendered_html)

    # From an HTML file
    HTML(os.path.join(path_to_insight_dict, "insight_card.html")).write_pdf(
        os.path.join(path_to_insight_dict, f"insight_card.pdf")
    )
    # print("PDF saved to", os.path.join(path_to_insight_dict, f"insight_card.pdf"))
