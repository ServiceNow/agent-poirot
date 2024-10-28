import os
import argparse
import shutil
import pandas as pd

from agentpoirot.agents import poirot
from agentpoirot import utils as ut


def main(exp_dict, savedir_base):
    dataset_name = exp_dict["dataset_name"]
    model_name = exp_dict["model_name"]
    savedir = os.path.join(savedir_base, f"{dataset_name}_{model_name}")

    # rm savedir with shutil
    if os.path.exists(savedir):
        # exp_dict.json must exist
        assert os.path.exists(os.path.join(savedir, "exp_dict.json"))

        shutil.rmtree(savedir)

    # Print Hyperparameters and save them in savedir
    print("Hyperparameters")
    print(exp_dict)
    ut.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)

    print("Saving in ", savedir)

    # Stage 1: Read dataset
    table = pd.read_csv(f"data/{dataset_name}/data.csv")

    # Create Agent Poirot the detective
    agent = poirot.Poirot(model_name=model_name, table=table, savedir=savedir)

    # Stage 2 Generate Questions
    questions = [
        "How is the case resolution duration for each assignment group?",
        "How are the cases distributed?",
    ]
    # questions = agent.generate_questions(n_questions=3)

    insight_list = []
    for q in questions:
        # stage 3: generate code
        code_output = agent.generate_code(q)

        # stage 4: get insight
        insight_dict = agent.generate_insight(code_output, with_pdf=False)

        insight_list += [insight_dict]

    insights_str = agent.display_insights(insight_list)
    print(insights_str)
    ut.save_txt(os.path.join(savedir, "insights.txt"), insights_str)

    print("Experiment saved in", savedir)


# create main
if __name__ == "__main__":

    # define model name from command line
    parser = argparse.ArgumentParser(description="Integrate Agent Poirot")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="Model name to use for inference",
    )
    # dataset name
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default="human_resources",
        help="Dataset name to use for inference",
    )
    # savedir
    parser.add_argument(
        "-s",
        "--savedir_base",
        type=str,
        default="results",
        help="Save directory",
    )
    # openai api key
    parser.add_argument(
        "-o",
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key",
    )
    args = parser.parse_args()
    model_name = args.model_name

    # create exp_dict
    exp_dict = {"dataset_name": args.dataset_name, "model_name": args.model_name}

    # set openai api key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # run main
    main(
        exp_dict=exp_dict,
        savedir_base=args.savedir_base,
    )
