
import pandas as pd

def load_xlsx(file_path):
    df = pd.read_excel(file_path)
    return df


