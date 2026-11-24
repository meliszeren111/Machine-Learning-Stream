
import pandas as pd

def load_txt_file_with_comma_sep(file_path):
    df = pd.read_csv(file_path, sep=",")
    return df

def load_txt_file_with_space_sep(file_path):
    df = pd.read_csv(file_path, sep="\t")
    return df