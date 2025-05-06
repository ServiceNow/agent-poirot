import sys, os

# get upper path
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path)

# Set Matplotlib to use a non-interactive backend before any other imports
import matplotlib

matplotlib.use("Agg")  # Use the 'Agg' backend which doesn't require a GUI

import utils as ut
from flask import Flask, request, render_template, jsonify, session
import os, re
import json
import pandas as pd
import time
import numpy as np

from agentpoirot.agents import poirot


form_questions = {}
# Initialize the Flask application
app = Flask(__name__)
app.config["RESULTS_FOLDER"] = "results"
app.config["USERS_FOLDER"] = "static/users"
app.config["DATA_FOLDER"] = "data"


def get_insight_card(generate_question=False):
    response = request.get_json()

    dataset_name = response["dataset"]
    fname = f"{app.config['DATA_FOLDER']}/{dataset_name}/data.csv"
    assert os.path.exists(fname)
    df = pd.read_csv(fname)
    meta = ut.load_json(f"{app.config['DATA_FOLDER']}/{dataset_name}/meta.json")
    user_data_folder = get_user_data_folder(request)

    # Get Past Questions
    past_questions = load_question_list(user_data_folder)

    # load all insight cards from user_data_folder/cached
    cached_path = os.path.join(user_data_folder, "cached")
    question = response.get("text", "")

    def find_matching_cached_insight(cached_path, question):
        """
        Search through cached insights to find matching question

        Args:
            cached_path: Path to cached insights folder
            question: Question to search for

        Returns:
            Path to matching insight folder if found, None otherwise
        """
        # Create cached directory if it doesn't exist
        os.makedirs(cached_path, exist_ok=True)

        # Check cached insights for matching question
        cached_insights = [
            f for f in os.listdir(cached_path) if re.match(r"insight_card_\d+", f)
        ]

        for insight_folder in cached_insights:
            insight_dict_path = os.path.join(
                cached_path, insight_folder, "insight_dict.json"
            )
            if os.path.exists(insight_dict_path):
                with open(insight_dict_path) as f:
                    insight_dict = json.load(f)
                    if insight_dict.get("question", "").lower() == question.lower():
                        return os.path.join(cached_path, insight_folder)
        return None

    # Check if question exists in cache
    matching_insight = find_matching_cached_insight(cached_path, question)
    if matching_insight:
        # ut.save_txt(os.path.join(cached_path, "question.txt"), question)
        # get recommended questions
        recommended_questions = get_recommended_questions(
            user_data_folder, df, meta, n_questions=10
        )
        recommended_questions = [q for q in recommended_questions if q != question]
        return jsonify(
            {
                "isValid": True,
                "insight": load_insight_card(matching_insight),
                "questions": recommended_questions,
            }
        )

    ## Get New Question
    agent = poirot.Poirot(
        table=df,
        savedir=user_data_folder,
        model_name=args.model_name,
        meta_dict=meta,
        verbose=True,
    )
    if not generate_question:
        question = response["text"]
        isValid, explanation, _ = agent.verify_question(question)

        recommended_questions = get_recommended_questions(
            user_data_folder, df, meta, n_questions=10
        )
        recommended_questions = [q for q in recommended_questions if q != question]

        if not isValid:
            return jsonify(
                {
                    "isValid": False,
                    "explanation": explanation,
                    "questions": recommended_questions,
                }
            )

    else:
        question = agent.generate_questions(
            n_questions=1, past_questions=past_questions
        )[0]

    questions = [question]

    insight_id_list = []

    for q in questions:
        insight_id = get_insight_id(user_data_folder)
        insight_card_path = os.path.join(user_data_folder, f"insight_card_{insight_id}")

        # save question
        ut.save_txt(os.path.join(insight_card_path, "question.txt"), question)

        try:
            insight_dict = agent.answer_question(
                q, output_folder=insight_card_path, add_timestamp=False
            )
        except:
            # delete insight_card_path
            import shutil

            if os.path.exists(insight_card_path):
                shutil.rmtree(insight_card_path)
            explanation = (
                "The question cannot be answered with this dataset unfortunately."
            )
            # save error
            ut.save_txt(os.path.join(insight_card_path, "question.txt"), question)
            ut.save_txt(os.path.join(insight_card_path, "error.txt"), explanation)

            return jsonify(
                {
                    "isValid": False,
                    "explanation": f"The question cannot be answered with this dataset unfortunately.",
                    "questions": recommended_questions,
                }
            )
        print("insight id", insight_id, "done")
        insight_id_list += [str(insight_id)]

    data = {"isValid": True}
    ut.save_txt(os.path.join(insight_card_path, "question.txt"), question)
    data["insight"] = load_insight_card(insight_card_path)
    data["insight_id"] = ", ".join(insight_id_list)
    # save in json file
    data["question"] = question
    # get recommended questions
    recommended_questions = get_recommended_questions(
        user_data_folder, df, meta, n_questions=10
    )
    data["questions"] = recommended_questions

    # ut.save_dataframe_shape(df, plot_html, plot_jpg)
    return jsonify(data)


@app.route("/main_add_insight", methods=["POST"])
def add_insight():
    """
    Render the main page.

    Returns:
        str: Rendered HTML for the main page.
    """
    return get_insight_card(generate_question=False)


@app.route("/main_add_insight_lucky", methods=["POST"])
def add_insight_lucky():
    """
    Render the main page.

    Returns:
        str: Rendered HTML for the main page.
    """
    return get_insight_card(generate_question=True)


@app.route("/")
def index():
    """
    Render the index page.

    Returns:
        str: Rendered HTML for the index page.
    """
    return render_template("index.html")
    # if starting_page == "main":
    #     from flask import Flask, render_template, request, redirect, url_for

    #     return redirect(url_for("main", dataset=args.dataset_name, user="guest"))

    # if starting_page == "xray":
    #     from flask import Flask, render_template, request, redirect, url_for

    #     return redirect(url_for("xray"))

    # return render_template(f"{starting_page}.html")


@app.route("/eval_comparison")
def eval_comparison():
    """
    Render the evaluation comparison page.

    Returns:
        str: Rendered HTML for the evaluation comparison page.
    """
    return render_template("eval_comparison.html")


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


# MAIN FUNCTIONS
################
def save_first_pairplot(savedir, df):
    import plotly.express as px
    import pandas as pd

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["float64", "int64"])

    # Compute the correlation matrix
    corr = numeric_df.corr().round(2)

    # Create a heatmap using Plotly
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
    )

    # Truncate labels
    def truncate_labels(labels):
        return [f"{label[:5]}..." if len(label) > 5 else label for label in labels]

    # Extract current tick labels from the heatmap data
    x_labels = fig.data[0].x
    y_labels = fig.data[0].y

    # Function to truncate labels
    def truncate_labels(labels):
        return [
            f"{str(label)[:5]}..." if len(str(label)) > 5 else str(label)
            for label in labels
        ]

    # Update axis labels with truncated labels
    fig.update_xaxes(
        ticktext=truncate_labels(x_labels), tickvals=x_labels, tickangle=45
    )
    fig.update_yaxes(
        ticktext=truncate_labels(y_labels), tickvals=y_labels, tickangle=45
    )

    # Make the plot tight
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    # Save the plot as a static image (plot.jpg) and an HTML file (plot.html)
    fig.write_image(f"{savedir}/plot.jpg")
    fig.write_html(f"{savedir}/plot.html")


def get_user_data_folder(request):
    response = request.get_json()
    dataset_name = response["dataset"]
    user_name = response["user"]
    user_data_folder = os.path.join(app.config["USERS_FOLDER"], user_name, dataset_name)
    return user_data_folder


def load_insight_cards(user_data_folder, return_count=False):
    insight_list = []
    insight_folders = [
        f for f in os.listdir(user_data_folder) if re.match(r"insight_card_\d+", f)
    ]
    insight_folders = sorted(
        insight_folders, key=lambda x: int(re.match(r"insight_card_(\d+)", x).group(1))
    )
    for f in insight_folders[::-1]:
        insight_path = os.path.join(user_data_folder, f)
        # load insight card
        # check if there is insight_dict.json and plot.html
        if not os.path.exists(os.path.join(insight_path, "insight_dict.json")):
            continue
        insight_card = load_insight_card(insight_path)
        insight_list.append(insight_card)
    if return_count:
        return "\n".join(insight_list), len(insight_list)

    return "\n".join(insight_list)


def load_insight_card(insight_path, id=None):
    insight_dict = json.load(open(os.path.join(insight_path, "insight_dict.json")))
    if id is None:
        insight_dict["id"] = int(insight_path.split("_")[-1])
    else:
        insight_dict["id"] = id
    plot_jpg, plot_html = get_plot_paths(insight_path)

    insight_dict["plot_image"] = plot_jpg
    insight_dict["plot_html"] = plot_html
    if os.path.exists(os.path.join(insight_path, "feedback.json")):
        # if feedback submitted
        hidden_add = "hidden"
        hidden_submitted = ""
    else:
        hidden_add = ""
        hidden_submitted = "hidden"

    insight_card = render_template(
        "fragments/insight_card.html",
        insight_dict=insight_dict,
        hidden_add=hidden_add,
        hidden_submitted=hidden_submitted,
    )
    return insight_card


def load_comparison_insight_card(insight_path, ab, id=None, question=None, plot=None):
    insight_dict = json.load(open(os.path.join(insight_path, "insight_dict.json")))
    if id is None:
        insight_dict["id"] = int(insight_path.split("_")[-1])
    else:
        insight_dict["id"] = id
    insight_dict["plot_image"] = os.path.join(insight_path, "plot.jpg")
    insight_dict["plot_html"] = os.path.join(insight_path, "plot.html")

    if not os.path.exists(insight_dict["plot_html"]):
        insight_dict["plot_html"] = os.path.join(insight_path, "plot.png")
    if not os.path.exists(insight_dict["plot_image"]):
        insight_dict["plot_image"] = os.path.join(insight_path, "plot.png")

    if question is not None:
        insight_dict["question"] = question

    if os.path.exists(os.path.join(insight_path, "feedback.json")):
        # if feedback submitted
        hidden_add = "hidden"
        hidden_submitted = ""
    else:
        hidden_add = ""
        hidden_submitted = "hidden"

    insight_card = render_template(
        "fragments/insight_eval_compare.html",
        insight_dict=insight_dict,
        hidden_add=hidden_add,
        hidden_submitted=hidden_submitted,
        ab=ab,
    )
    return insight_card


def dict_to_html(data):
    # Helper function to generate HTML for each indicator
    def generate_indicator_html(idx, indicator):
        return f"""
        <div class='indicator'>
            <p class='header'>Indicator {idx + 1}: {indicator['name']}</p>
            <p><strong>Description:</strong> {indicator['description']}</p>
            <p><strong>Threshold:</strong> {indicator['threshold']}</p>
        </div>
        """

    # Create the main HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Overview</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .header {{ font-weight: bold; }}
            .indicator {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <p class='header'>Description</p>
        <p>{data['description']}</p>

        <p class='header'>Goal</p>
        <p>{data['goal']}</p>

        <p class='header'>Persona</p>
        <p>{data['persona']}</p>

        <p class='header'>Dataset Name</p>
        <p>{data['dataset_name']}</p>

        <p class='header'>Indicator List</p>
        {''.join(generate_indicator_html(idx, indicator) for idx, indicator in enumerate(data['indicator_list']))}
    </body>
    </html>
    """

    return html_content


@app.route("/main_first_insight", methods=["POST"])
def first_insight():
    """
    Render the main page.

    Returns:
        str: Rendered HTML for the main page.
    """
    response = request.get_json()
    dataset_name = response["dataset"]
    fname = f"{app.config['DATA_FOLDER']}/{dataset_name}/data.csv"
    assert os.path.exists(fname)
    df = pd.read_csv(fname)
    meta_fname = f"{app.config['DATA_FOLDER']}/{dataset_name}/meta.json"
    if os.path.exists(meta_fname):
        meta = ut.load_json(meta_fname)
    else:
        meta = {}
        # save empty meta.json
        ut.save_json(meta_fname, meta)

    user_data_folder = get_user_data_folder(request)
    insight_path = os.path.join(user_data_folder, "insight_card_1")
    meta["dataset_name"] = meta.get("dataset_name", dataset_name)
    meta["indicator_list"] = meta.get("indicator_list", ["No Indicators"])
    meta["persona"] = meta.get("persona", "General User")
    meta["goal"] = meta.get("goal", "Find trends from this dataset")
    meta["description"] = meta.get("description", "No Description")

    # Create Data
    data = {}
    data["title"] = render_template(
        "fragments/title.html",
        meta=meta,
        n_rows=len(df),
        n_cols=len(df.columns),
    )

    data["meta"] = render_template(
        "fragments/meta.html",
        description=meta.get("description", "No Description"),
        goal=meta.get("goal", "Find trends from this dataset"),
        persona=meta.get("persona", "General User"),
        dataset_name=meta.get("dataset_name", dataset_name),
        indicators=meta.get("indicator_list", ["No Indicators"]),
    )

    if not os.path.exists(insight_path):
        os.makedirs(insight_path, exist_ok=True)
        save_first_pairplot(insight_path, df)
        question = f"Give me an overview of this dataset."
        insight_dict = {
            "highlight": f"Stats for {dataset_name}",
            "question": question,
            "answer": (
                f"The dataset has {len(df)} rows and {len(df.columns)} columns ({len(df.select_dtypes(include=['float64', 'int64']).columns)} numerical)."
            ),
            "insight": f"See the 'Dataset' Tab to view the dataset and the correlation map on the right hand-side for any interesting values.",
            "indicator": f"See 'Persona, Goals, Indicators' Tab.",
            "action": f"Ask Poirot to give you a top insight, to answer one of its recommended questions or ask your own question.",
            "justification": f"The summary statistics provide an overview of the {dataset_name} distribution.",
            "plot_html": "N/A",
            "plot_image": "N/A",
            "followup": "What is a top insight for this dataset?",
            "severity": "clear",
        }

        # create a corr
        insight_dict["highlight"] = "Stats for column"
        # save_dataframe_head_as_image(df, "plot.jpg", num_rows=5)
        plot_jpg, plot_html = get_plot_paths(insight_path)

        insight_dict["plot_image"] = plot_jpg
        insight_dict["plot_html"] = plot_html
        insight_dict["id"] = 1

        ut.save_json(os.path.join(insight_path, "insight_dict.json"), insight_dict)
        ut.save_txt(os.path.join(insight_path, "question.txt"), question)
        print(f"saved in {insight_path}")

    # load all insights
    data["insight"], n_insights = load_insight_cards(
        user_data_folder, return_count=True
    )
    data["insight_id"] = str(n_insights)

    # save in json file
    questions = get_recommended_questions(user_data_folder, df, meta, n_questions=10)
    data["questions"] = questions
    return jsonify(data)


def get_plot_paths(insight_path):
    plot_jpg = os.path.join(insight_path, "plot.jpg")
    if not os.path.exists(plot_jpg):
        plot_jpg = os.path.join(insight_path, "plot.png")
    plot_html = os.path.join(insight_path, "plot.html")
    if not os.path.exists(plot_html):
        plot_html = plot_jpg

    return plot_jpg, plot_html


def get_recommended_questions(user_data_folder, df, meta, n_questions=10, reset=False):
    questions_path = os.path.join(user_data_folder, "questions.json")
    past_questions = load_question_list(user_data_folder)
    if os.path.exists(questions_path) and not reset:
        questions = ut.load_json(questions_path)
    else:
        agent = poirot.Poirot(
            table=df,
            savedir=user_data_folder,
            model_name=args.model_name,
            meta_dict=meta,
            verbose=True,
        )

        questions = agent.generate_questions(
            n_questions=n_questions, past_questions=past_questions
        )
        # save json
        ut.save_json(questions_path, questions)

    # remove all questions in past_questions
    questions = [q for q in questions if q not in past_questions]
    if len(questions) == 0:
        return get_recommended_questions(
            user_data_folder, df, meta, n_questions=10, reset=True
        )
    # remove duplicates
    questions = list(set(questions))

    return np.random.choice(questions, min(len(questions), 3), replace=False).tolist()


def get_insight_id(user_data_folder):
    # get all folders with insight_card_*
    insight_folders = [
        f for f in os.listdir(user_data_folder) if re.match(r"insight_card_\d+", f)
    ]
    if len(insight_folders) == 0:
        return 1
    else:
        return (
            max(
                [
                    int(re.match(r"insight_card_(\d+)", f).group(1))
                    for f in insight_folders
                ]
            )
            + 1
        )


def load_question_list(user_data_folder):
    insight_folders = [
        f for f in os.listdir(user_data_folder) if re.match(r"insight_card_\d+", f)
    ]
    insight_folders = sorted(
        insight_folders, key=lambda x: int(re.match(r"insight_card_(\d+)", x).group(1))
    )

    question_list = []
    for f in insight_folders:
        if not os.path.exists(os.path.join(user_data_folder, f, "question.txt")):
            continue
        question = ut.load_txt(os.path.join(user_data_folder, f, "question.txt"))
        question_list.append(question)
    return question_list


@app.route("/main_questions", methods=["POST"])
def get_questions():
    """
    Retrieve questions and remove duplicates from the questions JSON file.

    Returns:
        JSON response: List of questions excluding those already in the questions file.
    """
    response = request.get_json()
    dataset_name = "space_utilization"
    model_name = "dummy_model"
    data = ut.load_json(
        os.path.join(
            app.config["RESULTS_FOLDER"],
            f"{dataset_name}_{model_name}",
            "preds.json",
        )
    )
    questions = data["questions"]
    txt = request.json.get("insights", "")

    # Ignore questions that are already in the questions.json file
    if os.path.exists("output/questions.json"):
        questions_so_far = ut.load_json("output/questions.json")
    else:
        questions_so_far = []

    questions = [q for q in questions if q not in questions_so_far]
    questions = questions[:3]

    return jsonify(questions)


# EVAL FUNCTIONS
# --------------


@app.route("/eval_page")
def eval_page():

    return render_template("eval_arcade.html")


@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    """
    Handle submission of feedback data.

    Returns:
        JSON response: Success or error message.
    """

    # Parse the JSON data from the request
    feedback_data = request.get_json()
    user_data_folder = get_user_data_folder(request)
    insight_id = feedback_data.get("insight").split("-")[-1]
    output_folder = os.path.join(user_data_folder, f"insight_card_{insight_id}")
    insight_dict = json.load(open(os.path.join(output_folder, "insight_dict.json")))

    output_file = os.path.join(output_folder, "feedback.json")
    # feedback_data["insight_dict"] = insight_dict
    with open(output_file, "w") as f:
        json.dump(feedback_data, f, indent=4)
    print(f"feedback saved in {output_file}")
    return jsonify({"status": "success", "message": "Feedback submitted successfully!"})


@app.route("/get_dataframe", methods=["GET"])
def get_dataframe():
    """Get the current dataset as JSON for DataTables"""
    try:
        # Get dataset name from URL parameters
        dataset_name = request.args.get("dataset", "human_resources")  # default dataset

        # Load the CSV file
        fname = f"{app.config['DATA_FOLDER']}/{dataset_name}/data.csv"
        df = pd.read_csv(fname)

        # Convert DataFrame to dictionary format suitable for DataTables
        data = {
            "data": df.values.tolist(),
            "columns": [{"title": col} for col in df.columns],
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/main")
def main():
    """
    Render the main page.

    Returns:
        str: Rendered HTML for the main page.
    """
    return render_template("main.html")


def load_xray():
    pass


# ----- XRAY ----- #
@app.route("/xray")
def xray():
    """
    Render the main page.

    Returns:
        str: Rendered HTML for the main page.
    """
    return render_template("xray.html")


@app.route("/xray_load_insight_card", methods=["POST"])
def xray_load_insight_card():
    # pass

    folder_path = "static/question_1728394089"
    insight_card = load_insight_card(folder_path, id="A")
    prompt_insight = ut.load_txt(os.path.join(folder_path, "prompt_insight.txt"))
    data = {}
    data["insight_card_a"] = (
        f'<textarea rows="60" cols="100">{prompt_insight}</textarea>'
    )
    data["prompt"] = insight_card
    data["dataset"] = args.dataset_name
    data["timestamp"] = timestamp = str(int(time.time()))
    return jsonify(data)


@app.route("/xray_run_prompt", methods=["POST"])
def xray_run_prompt():
    # pass

    folder_path = "static/question_1728394089"
    insight_card = load_insight_card(folder_path, id="A")
    fname = f"{app.config['DATA_FOLDER']}/{args.dataset_name}/data.csv"
    df = pd.read_csv(fname).head()
    meta = ut.load_json(f"{app.config['DATA_FOLDER']}/{args.dataset_name}/meta.json")
    agent = poirot.Poirot(
        table=df,
        savedir=folder_path,
        model_name=args.model_name,
        verbose=True,
        meta_dict=meta,
    )
    code_output = ut.load_json(os.path.join(folder_path, "code_output.json"))
    insight_dict = agent.generate_insight(code_output, with_pdf=False)
    prompt_insight = ut.load_txt(os.path.join(folder_path, "prompt_insight.txt"))
    data = {}
    data["insight_card_a"] = (
        f'<textarea rows="60" cols="100">{prompt_insight}</textarea>'
    )
    data["prompt"] = insight_card
    data["dataset"] = args.dataset_name
    data["timestamp"] = timestamp = str(int(time.time()))
    return jsonify(data)


@app.route("/xray_submit_prompt", methods=["POST"])
def xray_submit_prompt():
    pass


if __name__ == "__main__":
    # create port args
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=7867)
    parser.add_argument("-s", "--starting_page", type=str, default="main")
    # model_name
    parser.add_argument("-m", "--model_name", type=str, default="gpt-4o")
    parser.add_argument("-d", "--dataset_name", type=str, default="csm")

    args = parser.parse_args()
    starting_page = args.starting_page

    app.run(debug=True, port=args.port, host="0.0.0.0", use_reloader=False)
