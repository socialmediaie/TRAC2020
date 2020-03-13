import os
import pandas as pd
import numpy as np
import pprint
import random

#print(os.getcwd())

if 'TRAC_PATH' not in os.environ:
    os.environ['TRAC_PATH'] = os.getcwd()

BASE_PATH = os.environ.get('TRAC_PATH')
DATA_PATHS_TRAIN = {
    "ENG": f"{BASE_PATH}/data/raw/eng/trac2_eng_train.csv",
    "IBEN": f"{BASE_PATH}/data/raw/iben/trac2_iben_train.csv",
    "HIN": f"{BASE_PATH}/data/raw/hin/trac2_hin_train.csv"
}
DATA_PATHS_DEV = {
    "ENG": f"{BASE_PATH}/data/raw/eng/trac2_eng_dev.csv",
    "IBEN": f"{BASE_PATH}/data/raw/iben/trac2_iben_dev.csv",
    "HIN": f"{BASE_PATH}/data/raw/hin/trac2_hin_dev.csv"
}
DATA_PATHS_TEST = {
    "ENG": f"{BASE_PATH}/data/raw/eng/trac2_eng_test.csv",
    "IBEN": f"{BASE_PATH}/data/raw/iben/trac2_iben_test.csv",
    "HIN": f"{BASE_PATH}/data/raw/hin/trac2_hin_test.csv"
}

#print(DATA_PATHS_DEV)

DATA_COLUMNS = ["row_id", "text", "Sub-task A", "Sub-task B"]

NUM_LANGUAGES = len(DATA_PATHS_TRAIN)
print(NUM_LANGUAGES)
TASK_LABEL_IDS = {
    "Sub-task A": ["OAG", "NAG", "CAG"],
    "Sub-task B": ["GEN", "NGEN"],
    #"Sub-task C": ["OAG-GEN", "OAG-NGEN", "NAG-GEN", "NAG-NGEN", "CAG-GEN", "CAG-NGEN"]
}

def gen_data(args):
    all_lang_dfs = {}
    all_task_cols = []
    for data_type, DATA_PATHS in [("train", DATA_PATHS_TRAIN), ("dev", DATA_PATHS_DEV), ("test", DATA_PATHS_TEST)]:
        print(data_type)
        for lang, path in DATA_PATHS.items():
            df = pd.read_csv(path, sep=",")
            if data_type == "test":
                df = df.assign(**{
                    k: v[0]
                    for k,v in TASK_LABEL_IDS.items()
                })
            df["Sub-task C"] = df["Sub-task A"].str.cat(df["Sub-task B"], "-")
            #elif lang == "DE":
            #    df.loc[df.task_1 == "NOT", "task_3"] = "NONE"
            #    df.loc[df.task_1 != "NOT", "task_3"] = "TIN"

            # This is a fix for fixing errors in teasor data. Should not apply to test
            #if data_type != "test":
            #    df.loc[df["task_1"] == "NOT", ["task_2", "task_3"]] = "NONE"

            #if data_type != "test":
                #for task in ["task_1", "task_2", "task_3"]:
                #    df[task] = df[task].str.upper().replace("NULL", "NONE")
            #df["task_4"] = df["task_1"].str.cat(df[["task_2", "task_3"]].astype(str), sep="-")
            task_cols = df.filter(regex=r'Sub-task *', axis=1).columns
            for task in task_cols:
                #if task == "task_3" and lang == "DE":
                #    continue
                y = df[task]
                idx = (y != "NONE")
                df_t = df[idx]
                df_bert = pd.DataFrame({
                  'id': list(range(df_t.shape[0])),
                  'label': y[idx],
                  'alpha': ['a']*df_t.shape[0],
                  'text': df_t["Text"].replace(r'\s+', ' ', regex=True)
                })
                if args.normalize:
                    df_bert["text"] = df_bert["text"].replace(r'\[(#\w+)\]\(.*?\)', r'\1', regex=True)
                    df_bert["text"] = df_bert["text"].replace(r'\[.*?\]\(.*?\)', '__TIMEURL__', regex=True)
                    df_bert["text"] = df_bert["text"].replace(r'@[^\s]+', '@USER', regex=True)
                    df_bert["text"] = df_bert["text"].replace(r'http[s]?://[^\s]+', '__URL__', regex=True)
                os.makedirs(os.path.join("./", lang, task), exist_ok=True)
                bert_format_path = os.path.join("./", lang, task, f"{data_type}.tsv")
                print(bert_format_path)
                df_bert.to_csv(bert_format_path, sep='\t', index=False, header=False)
                bert_format_path = os.path.join("./", lang, task, f"{data_type}.json")
                print(bert_format_path)
                df_bert.to_json(bert_format_path, orient="records", lines=True)
                if args.all_langs:
                    all_task_cols = task_cols
                    all_lang_dfs[data_type] = all_lang_dfs.get(data_type, {k: [] for k in task_cols})
                    all_lang_dfs[data_type][task].append(df_bert.assign(id=df_bert["id"].apply(lambda x: f"{lang}-{x}")))
    if args.all_langs:
        lang = "ALL"
        for data_type, DATA_PATHS in [("train", DATA_PATHS_TRAIN), ("dev", DATA_PATHS_DEV), ("test", DATA_PATHS_TEST)]:
            for task in all_task_cols:
                df_bert = pd.concat(all_lang_dfs[data_type][task])
                os.makedirs(os.path.join("./", lang, task), exist_ok=True)
                bert_format_path = os.path.join("./", lang, task, f"{data_type}.tsv")
                print(bert_format_path)
                df_bert.to_csv(bert_format_path, sep='\t', index=False, header=False)
                bert_format_path = os.path.join("./", lang, task, f"{data_type}.json")
                print(bert_format_path)
                df_bert.to_json(bert_format_path, orient="records", lines=True)
    

def get_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--normalize', action="store_true",
                        help="Normalize text to replace mentions and urls")
    parser.add_argument('--all_langs', action="store_true",
                        help="All language data")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    gen_data(args)
