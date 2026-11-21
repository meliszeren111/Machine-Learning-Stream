
import pandas as pd

def load_xlsx_file(file_path):
    df = pd.read_excel(file_path)
    return df


