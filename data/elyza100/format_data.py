import pandas as pd


def format_data():
    # read excel file
    excel_file_path = "ELYZA-tasks-100.xlsx"
    df_response_list, df_score_list = [], []
    for sheet_idx in range(2, -1, -1):
        df = pd.read_excel(excel_file_path, sheet_name=sheet_idx)
        df_r = df.iloc[1:, list(range(3, df.shape[1], 3))]
        df_s = df.iloc[1:, list(range(5, df.shape[1], 3))]
        df_response_list.append(df_r)
        df_score_list.append(df_s)

    # if there are identical column names, the latest one is used
    df_response = pd.concat(df_response_list, axis=1).loc[
        :, ~pd.concat(df_response_list, axis=1).columns.duplicated(keep="last")
    ]
    df_score = pd.concat(df_score_list, axis=1).loc[
        :, ~pd.concat(df_score_list, axis=1).columns.duplicated(keep="last")
    ]

    # change index name to item1, item2, ...
    df_response.index = [f"item{i+1}" for i in range(len(df_response))]
    df_score.index = [f"item{i+1}" for i in range(len(df_score))]

    # remove eval_ and _mean from column names
    df_score.columns = df_score.columns.str.replace(
        "^eval_", "", regex=True
    ).str.replace("_mean$", "", regex=True)

    # round the average score to the nearest whole number
    df_score = (df_score + 0.5).astype(int)

    # transpose so that each row is the examinee and each column is the item
    df_response = df_response.T
    df_score = df_score.T

    # save csv file
    df_response.index.name = "examinee"
    df_score.index.name = "examinee"
    df_response.to_csv("response.csv")
    df_score.to_csv("score.csv")


if __name__ == "__main__":
    # create a csv file from a raw data file
    format_data()
