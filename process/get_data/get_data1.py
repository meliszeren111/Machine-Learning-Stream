import io

import pandas as pd
import tensorflow as tf
from datasets import load_dataset
import torchvision.datasets as tv_datasets
import sklearn.datasets as sk_datasets

def get_data_from_source(source: str, dataset_name: str = None, **kwargs):
    source = source.lower()
    if source == "huggingface":
        return load_dataset(dataset_name, **kwargs)
    elif source == "sklearn":
        func_name = dataset_name if dataset_name.startswith("load_") or dataset_name.startswith("fetch_") else f"load_{dataset_name}"
        if hasattr(sk_datasets, func_name):
            return getattr(sk_datasets, func_name)(**kwargs)
        elif hasattr(sk_datasets, f"fetch_{dataset_name}"):
            return getattr(sk_datasets, f"fetch_{dataset_name}")(**kwargs)
        else:
            raise ValueError(f"Desteklenmeyen sklearn veri seti: {dataset_name}")
    elif source in ["torchvision", "pytorch", "torch"]:
        if hasattr(tv_datasets, dataset_name):
            dataset_cls = getattr(tv_datasets, dataset_name)
            return dataset_cls(**kwargs)
        else:
            raise ValueError(f"Desteklenmeyen torchvision veri sınıfı: {dataset_name}")
    elif source in ["tensorflow", "tf", "keras"]:
        keras_datasets = tf.keras.datasets
        if hasattr(keras_datasets, dataset_name):
            ds_module = getattr(keras_datasets, dataset_name)
            return ds_module.load_data(**kwargs)
        else:
            import tensorflow_datasets as tfds
            return tfds.load(dataset_name, **kwargs)
    else:
        raise ValueError(f"Desteklenmeyen kaynak: {source}")


def get_data_from_file(file_path: str):
    extension = file_path.split(".")[-1].lower()
    if extension == "csv":
        return pd.read_csv(file_path)
    elif extension == "tsv":
        return pd.read_csv(file_path, sep="\t")
    elif extension in ["xlsx", "xls"]:
        return pd.read_excel(file_path)
    elif extension == "txt":
        return pd.read_csv(file_path, sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def get_data_from_file_binary(contents: bytes, file_name: str):
    extension = file_name.split(".")[-1].lower()
    file = io.BytesIO(contents)
    if extension == "csv":
        return pd.read_csv(file)
    elif extension == "tsv":
        return pd.read_csv(file, sep="\t")
    elif extension in ["xlsx", "xls"]:
        return pd.read_excel(file)
    elif extension == "txt":
        return pd.read_csv(file, sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported file format: {extension}")