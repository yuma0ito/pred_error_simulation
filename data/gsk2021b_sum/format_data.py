import argparse
import os
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def get_evaluators(theme):
    if theme in ["global", "science", "easia", "criticize", "gdp1", "koumin", "sciexp"]:
        return ["i", "s"]
    elif theme in ["creativity", "populism"]:
        return ["kk", "y"]
    else:
        raise ValueError()


def format_data(theme, response_path, score_path):
    if theme == "sciexp":
        item_names = ["問", "問１", "問２"]
        score_names = ["問1理解力", "問2-1理解力", "問2-2理解力"]
    else:
        item_names = ["問１", "問２", "問３"]
        score_names = ["問1理解力", "問2理解力", "問3理解力"]

    # read response data
    response_dict = {}
    for root, _, filenames in os.walk(response_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if not os.path.isfile(file_path):
                continue
            if not filename.endswith((".xlsx", ".xls")):
                continue

            df = pd.read_excel(file_path)
            examinee_id = str(int(df.iloc[0, 3]))
            response = df[df.iloc[:, 0].isin(item_names)].iloc[:, 1].fillna("").tolist()
            assert len(response) == 3
            response_dict[examinee_id] = response

    # read score data
    score_list = []
    for filename in os.listdir(score_path):
        file_path = os.path.join(score_path, filename)
        if not os.path.isfile(file_path):
            continue

        # get recommended evaluators
        evaluators = get_evaluators(theme)
        if filename.split("_")[2] not in evaluators:
            continue

        sheets = pd.read_excel(file_path, sheet_name=None)
        for df in sheets.values():
            # change column name
            df.columns = df.iloc[1]
            df = df.drop([0, 1]).reset_index(drop=True)

            # convert examination number to one-byte character string
            zen_to_han_table = str.maketrans("０１２３４５６７８９", "0123456789")

            def convert_to_half_width(x):
                if isinstance(x, int):
                    return str(x)
                elif isinstance(x, str):
                    return x.translate(zen_to_han_table)
                else:
                    return x

            df["受験番号"] = df["受験番号"].apply(convert_to_half_width)

            # extract only the rows where 'attendance' is not 'absent'
            df_filtered = df[df["出欠"] != "欠席"]

            # create a dictionary of scores with the examination number as the key
            score_dict_per_eval = (
                df_filtered.set_index("受験番号")[score_names]
                .dropna(how="all")
                .apply(list, axis=1)
                .to_dict()
            )
            score_list.append(score_dict_per_eval)

    score_dict = {}
    for score_dict_per_eval in score_list:
        for examinee_id, score_per_eval in score_dict_per_eval.items():
            if examinee_id not in score_dict:
                score_dict[examinee_id] = [[score] for score in score_per_eval]
            else:
                for i, score in enumerate(score_per_eval):
                    score_dict[examinee_id][i].append(score)

    sum_score_dict = {}
    for key in score_dict:
        sum_score_dict[key] = [
            sum(score_per_item) for score_per_item in score_dict[key]
        ]

    response_dict = {key: response_dict[key] for key in sorted(response_dict)}
    sum_score_dict = {key: sum_score_dict[key] for key in response_dict}
    assert response_dict.keys() == sum_score_dict.keys()

    df_response = pd.DataFrame(response_dict).T
    df_response.reset_index(inplace=True)
    df_response.columns = ["examinee"] + [
        f"item{i+1}" for i in range(len(df_response.columns) - 1)
    ]

    df_score = pd.DataFrame(sum_score_dict).T
    df_score.reset_index(inplace=True)
    item_columns = [f"item{i+1}" for i in range(len(df_score.columns) - 1)]
    df_score.columns = ["examinee"] + item_columns
    df_score[item_columns] = df_score[item_columns].astype(int)

    # save csv file
    os.makedirs(theme, exist_ok=False)
    df_response.to_csv(os.path.join(theme, "response.csv"), index=False)
    df_score.to_csv(os.path.join(theme, "score.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--theme",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    response_path = os.path.join("raw_data", args.theme, "response")
    score_path = os.path.join("raw_data", args.theme, "score")

    # create a csv file from a raw data file
    format_data(args.theme, response_path, score_path)
