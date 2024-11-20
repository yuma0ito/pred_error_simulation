import numpy as np
import pandas as pd


def format_data():
    # read raw data
    df_response = pd.read_csv("raw_data/collect_essay.csv")
    df_score = pd.read_csv("raw_data/collect_score.csv")

    # response
    df_response.columns = ["exam_id", "respondent_id", "response"]
    df_response = df_response.pivot(
        index="respondent_id", columns="exam_id", values="response"
    ).reset_index()
    df_response.columns = ["examinee"] + [
        f"item{i+1}" for i in range(len(df_response.columns) - 1)
    ]

    # score
    df_score = (
        df_score.groupby(["exam_id", "respondent_id"])["score"]
        .mean()
        .apply(lambda x: np.floor(x + 0.5))
        .reset_index()
    )
    df_score = df_score.pivot(
        index="respondent_id", columns="exam_id", values="score"
    ).reset_index()
    item_columns = [f"item{i+1}" for i in range(len(df_score.columns) - 1)]
    df_score.columns = ["examinee"] + item_columns
    df_score[item_columns] = df_score[item_columns].astype(int)

    # save csv file
    df_response.to_csv("response.csv", index=False)
    df_score.to_csv("score.csv", index=False)


if __name__ == "__main__":
    # create a csv file from a raw data file
    format_data()
