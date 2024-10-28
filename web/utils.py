import sys, os
import hashlib, os, json, pprint


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


def save_txt(fname, data, makedirs=True):
    """Save data into a text file.

    Parameters
    ----------
    fname : str
        Name of the text file
    data : [type]
        Data to save into the text file
    makedirs : bool, optional
        If enabled creates the folder for saving the file, by default True
    """
    # turn fname to string in case it is a Path object
    fname = str(fname)
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as txt_file:
        txt_file.write(data)


def save_json(fname, data, makedirs=True):
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
    fname = str(fname)
    dirname = os.path.dirname(fname)
    if makedirs and dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


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


from flask import Flask, request, render_template, jsonify, session
import os, re
import json
import pandas as pd
import utils as ut
import copy, time
import numpy as np

from agentpoirot import agents


# dataset_list = ["space_utilization"]
# model_list = ["dummy_model"]
# TIMESTAMP = int(time.time())


def filter_insights(insight_list_given, dataset_name, model_name):
    # get filenames under savedir_base
    insight_list = []
    path = os.path.join(app.config["FEEDBACK_FOLDER"], STATE["username"])
    for fname in os.listdir(path):
        insight_list += [ut.load_json(os.path.join(path, fname))["insight_meta"]]

    # remove the ones that have been loaded from data2["insights"]
    # get all that match the dataset_name and model_name
    insight_list = [
        f
        for f in insight_list
        if f["model_name"] == model_name and f["dataset_name"] == dataset_name
    ]
    # remove everything from data2['insights'] with matchines id==id in insight_list
    # Extract IDs from insight_list
    ids_to_remove = {f["id"] for f in insight_list}
    if len(ids_to_remove) > 0:
        print()
    # Remove items from data2['insights'] with matching IDs
    insights = [item for item in insight_list_given if item["id"] not in ids_to_remove]

    return insights


def load_txt(fname):
    with open(fname, "r") as f:
        return f.read()


def get_remaining(exp_list):

    remaining_datasets = []
    n_total = 0
    for exp_dict in exp_list:
        if exp_dict.get("agent_type") == "pandas":
            model_name = "pandas"
        else:
            model_name = "gpt-4o"
        dataset_id = exp_dict["dataset_id"]
        dataset_name = str(dataset_id)
        data = get_data_from_cba(dataset_name, dataset_id, model_name)

        if len(data["insights"]) > 0:
            remaining_datasets.append(
                {
                    "dataset_name": dataset_name,
                    "dataset_id": dataset_id,
                    "model_name": model_name,
                    "data": data,
                }
            )
            n_total += len(data["insights"])

    print("Total number of insights left: ", n_total)
    return remaining_datasets


import re


def remove_number_prefix(s):
    # This regular expression matches a number at the start of the string followed by a dot and a space
    return re.sub(r"^\d+\.\s*", "", s).strip()


def get_data_from_cba(dataset_name, dataset_id, model_name):
    data = {}
    output_dict = get_output_dict(
        model_name,
        dataset_id,
        dataset_path=app.config["STUDY_FOLDER"],
    )

    data["title"] = f"{dataset_id}"
    data["description"] = (
        "The dataset simulates ServiceNow incidents table, capturing incident management activities, reflecting the operational handling and urgency of issues across different locations and categories."
    )
    data["meta"] = {
        "persona": "Business Analytics Engineer.",
        "goal": "Find interesting trends in this dataset",
    }
    data["dataset_id"] = dataset_id
    data["model_name"] = model_name
    data["dataset_name"] = dataset_name
    # add insight id to output_dict
    for idx, insight in enumerate(output_dict["output"]):
        insight["id"] = idx
        # add short description
        insight["short_description"] = "Insight Card"
        # Add Action
        insight["action"] = "-"
        insight["plot_path"] = insight["plot"]
        insight["question"] = remove_number_prefix(insight["question"])
        insight["persona"] = data["meta"]["persona"]
        insight["goal"] = data["meta"]["goal"]

    # remove all those with plot doesnt exist
    insight_list_new = []
    for insight in output_dict["output"]:
        if os.path.exists(str(insight["plot"])):
            insight_list_new.append(insight)
    output_dict["output"] = insight_list_new
    insight_list = filter_insights(output_dict["output"], dataset_name, model_name)
    data["insights"] = insight_list

    return data


def get_exp_list():
    exp_list = []
    for exp_id in os.listdir(STUDY_FOLDER):
        exp_dict = json.load(open(f"{STUDY_FOLDER}/{exp_id}/exp_dict.json"))
        exp_list += [exp_dict]
    return exp_list


def get_insight_html(data, eval=False):
    """
    Generate HTML content for all insights in the data.

    Args:
        data (dict): The data containing insights.

    Returns:
        str: HTML string for all insights.
    """
    insight_html_list = []

    for insight in data["insights"]:
        insight_html = get_one_insight_html(insight, eval)
        insight_html_list.append(insight_html)
    if eval:
        return "\n".join(insight_html_list)
    return ""


def get_one_insight_html(value, eval=False):
    """
    Generate HTML content for a single insight.

    Args:
        value (dict): The data for a single insight.

    Returns:
        str: HTML string for the insight.
    """
    if eval:
        insight_html = render_template("fragments/insight_eval.html", value=value)
    else:
        insight_html = render_template("fragments/insight.html", value=value)

    return insight_html


def update_question_json(question):
    """
    Update the JSON file with a new question.

    Args:
        question (str): The new question to add.

    Returns:
        list: Updated list of questions.
    """
    output_file_path = os.path.join(app.config["OUTPUT_FOLDER"], "questions.json")

    # Load existing questions or initialize a new list
    if os.path.exists(output_file_path):
        with open(output_file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    # Append the new question and save the file
    data.append(question)
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)

    return data


import pandas as pd
import numpy as np


import pandas as pd
import plotly.express as px
import numpy as np


import pylab as plt
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table


def save_dataframe_head_as_image(df, filename, num_rows=5):
    # Create a DataFrame with only the first few rows
    df_head = df.head(num_rows)

    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 2 + num_rows * 0.5))  # Adjust size as needed
    ax.axis("off")  # Turn off the axis

    # Create a table and add it to the plot
    tbl = table(
        ax,
        df_head,
        loc="center",
        cellLoc="center",
        colWidths=[0.2] * len(df_head.columns),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)  # Adjust scale as needed

    # Save the plot as an image
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


import plotly.graph_objects as go


def save_dataframe_shape(df, html_filename, jpg_filename):
    # Get the number of rows and columns
    num_rows, num_columns = df.shape

    # Create a bar plot
    fig = go.Figure(
        data=[
            go.Bar(
                name="Number of Rows",
                x=["DataFrame"],
                y=[num_rows],
                marker_color="blue",
            ),
            go.Bar(
                name="Number of Columns",
                x=["DataFrame"],
                y=[num_columns],
                marker_color="orange",
            ),
        ]
    )

    # Update layout for better visualization
    fig.update_layout(
        title="Number of Rows and Columns in DataFrame",
        xaxis_title="DataFrame",
        yaxis_title="Count",
        barmode="group",
    )

    # Save the plot as an HTML file
    fig.write_html(html_filename)

    # Save the plot as a JPEG image
    fig.write_image(jpg_filename, format="jpeg")


def clean_email(email):
    email = email.split("@")[0]
    # List of special characters to remove
    special_chars = [
        "!",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "=",
        "+",
        "{",
        "}",
        "[",
        "]",
        "|",
        "\\",
        ":",
        ";",
        '"',
        "'",
        "<",
        ">",
        ",",
        ".",
        "?",
        "/",
        " ",
    ]

    # Iterate over each special character and replace it with an empty string
    for char in special_chars:
        email = email.replace(char, "_")

    return email
