import os
import json
import csv
from datasets import Dataset
from pathlib import Path
from typing import List, Dict
import re
import unicodedata # for normalize_str()


# For pushing results to a remote GoogleSheet
import gspread
from oauth2client.service_account import ServiceAccountCredentials





def jsonl_into_hfdataset(path_to_jsonl: str) -> Dataset:
        """Function that converts a .jsonl dataset file into a corresponding HF Dataset.
                - Assumes that all entries have the same collumns, if a collumn misses a value.

        Args:
            path_to_jsonl (str): path to the location of the .jsonl file

        Returns:
            Dataset: a HF Dataset instance that can later be added to a HF DatasetDict or not
        """
        store_all_lists = None
        keys = None # A list that stores all the keys from each entry

        with open(path_to_jsonl, "r", encoding="UTF-8") as file:
                for i, line in enumerate(file):
                        try:
                                entry = json.loads(line)
                        except json.JSONDecodeError:
                                print(f"Skipping malformed line {i}")
                                continue

                        # Initialize schema on first line
                        if store_all_lists is None:
                                keys = list(entry.keys())
                                store_all_lists = {key: [] for key in keys}

                        # Append values, is case some key is missing it will append a None value
                        for k in keys:
                                store_all_lists[k].append(entry.get(k,None))
                
                return Dataset.from_dict(store_all_lists)
        

from datasets import load_dataset, Dataset

def loading_PT_wikiRAG_from_HF(token: str = None):
    """
    Loads the PT_wikiRAG dataset directly from HuggingFace Hub.
    Since right now the dataset is private, the token will need to be passed.
    NOTE: I should add support to load the HFtoken from the environment variable.

    Args:
        token (str): HuggingFace Token

    Returns:
        list: A list with two entries, the first is the subset1 with whole documents and the second is the subset 2 with questions, answers ans citations from the source documents
    """
    subset1 = load_dataset("nunoFcul/PT_wikiRAG-docs", token=token)
    subset2 = load_dataset("nunoFcul/PT_wikiRAG-qac", token=token)

        
    return {"subsplit1": subset1["train"].to_list(), "subsplit2": subset2["train"].to_list()}

def load_jsonl_file(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts, skipping malformed lines."""
    data = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):  # enumerate starting at 1 (line numbers)
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping malformed line {i} in {path.name}")
    return data

def load_jsonl_PT_wikiRAG_locally(s1_path: str, s2_path: str) -> Dict[str, List[dict]]:
    """Load two JSONL files (subsplit1, subsplit2) and return as a dict of lists."""
    return {
        "subsplit1": load_jsonl_file(s1_path),
        "subsplit2": load_jsonl_file(s2_path),
    }


def remove_special_characters(text: str) -> str:
    """
    Remove all special characters from a string,
    keeping only normal letters (a-z, A-Z).
    """
    return re.sub(r'[^a-zA-Z]', '', text)


def make_alias(model_name: str) -> str:
    """
    Convert a model name into a safe alias (only lowercase letters and numbers).

    example: 
    
    PORTULAN/serafim-100m-portuguese-pt-sentence-encoder-ir -> portulanserafim100mportugueseptsentenceencoderir
    """
    # lowercase everything
    alias = model_name.lower()

    # removel all non-alphanumeric characters
    alias = re.sub(r'[^a-z0-9]', '', alias)

    return alias


def normalize_str(s: str) -> str:
    return unicodedata.normalize("NFC", s)





def save_results_csv(score_results, model_list, output_dir, output_file="retrieval_eval_results.csv", delimiter=","):
    """
    Save evaluation results into a CSV/TSV file.

    Parameters
    ----------
    score_results : dict
        Nested dict with structure {model_name: {chunking_strat: {"MRR@k": score}}}
    model_list : list
        List of model names to write results for
    output_dir : str
        Directory where CSV/TSV will be saved
    output_file : str
        Name of the file to write
    delimiter : str
        Separator ("," for CSV, "\\t" for TSV)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["model_name", "chunking_strat", "top_k", "MRR_score"])

        for model_name in model_list:
            for path, result_dict in score_results[model_name].items():
                for metric, score in result_dict.items():
                    top_k = metric.split("@")[1]
                    writer.writerow([model_name, path, top_k, score])

    print(f"✅ Results saved to {output_path}")


def append_to_google_sheet(sheet_name, worksheet_name, score_results, model_list, creds_path="service_account.json"):
    """
    Append results into a Google Sheet.

    Parameters
    ----------
    sheet_name : str
        Name of the Google Sheet (must exist)
    worksheet_name : str
        Name of the worksheet/tab (must exist)
    score_results : dict
        Nested dict {model_name: {chunking_strat: {"MRR@k": score}}}
    model_list : list
        List of models evaluated
    creds_path : str
        Path to Google service account JSON credentials
    """
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)

    sheet = client.open(sheet_name).worksheet(worksheet_name)

    rows = []
    for model_name in model_list:
        for path, result_dict in score_results[model_name].items():
            for metric, score in result_dict.items():
                top_k = metric.split("@")[1]
                rows.append([model_name, path, top_k, score])

    for row in rows:
        sheet.append_row(row, value_input_option="RAW")

    print(f"✅ Results appended to Google Sheet: {sheet_name} → {worksheet_name}")
