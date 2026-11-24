
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import split_dataset
from datasets import Dataset, load_dataset


def split_sklearn_dataset(X, y, test_size, train_size, shuffle=True, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, shuffle=shuffle, random_state=random_state)
    return X_train, X_test, y_train, y_test


def split_tensorflow_dataset(dataset, left_size=0.8, right_size=0.2, shuffle=True, seed=42,
                             preferred_backend="tensorflow"):
    (X_train, y_train), (X_test, y_test) = split_dataset(dataset, left_size, right_size, shuffle, seed, preferred_backend)
    return X_train, X_test, y_train, y_test

def split_torch_dataset(dataset, left_size=0.8, right_size=0.2, shuffle=True, seed=42, preferred_backend="torch"):
    (X_train, y_train), (X_test, y_test) = split_dataset(dataset, left_size, right_size, shuffle, seed, preferred_backend)
    return X_train, X_test, y_train, y_test


# 'train', ['train', 'test'], 'train+test', 'train[10:20]', 'train[:10%]', 'train[:10%]+train[-80%:]'
# [f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)], datasets.ReadInstruction('train')
# datasets.ReadInstruction('train', to=10, unit='%')
# [datasets.ReadInstruction('train', from_=k, to=k+10, unit='%') for k in range(0, 100, 10)]
def split_huggingface_dataset(
    dataset_name,
    split="train",
    seed=42,
    shuffle=True,
    ylabel="target",
    test_size=0.2
):
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    if ylabel not in df.columns:
        raise KeyError(f"'{ylabel}' sütunu Hugging Face veri setinde bulunamadı.")
    X = df.drop(columns=[ylabel])
    y = df[ylabel]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test

def split_uploaded_dataset(df, ylabel, test_size, train_size, shuffle=True, random_state=42):
    if ylabel not in df.columns:
        raise KeyError(f"'{ylabel}' column not found in dataset.")
    y = df[ylabel]
    X = df.drop(columns=[ylabel])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, shuffle=shuffle, random_state=random_state)
    return X_train, X_test, y_train, y_test

def split_data(df=None, huggingface_dataset=None, data_type="sklearn", ylabel="target", test_size=0.2, train_size=0.8, split_by="train", left_size=0.2, right_size=0.8, shuffle=True, random_state=42):
    if data_type == "sklearn":
        if ylabel not in df.columns:
            raise KeyError(f"'{ylabel}' column not found in dataset.")
        X = df.drop(columns=[ylabel])
        y = df[ylabel]
        return split_sklearn_dataset(X, y, test_size, train_size, shuffle=shuffle, random_state=random_state)
    elif data_type in ["xlsx", "xls", "csv", "tsv", "txt", "ordinal", "nonordinal", "image"]:
        return split_uploaded_dataset(df, ylabel, test_size, train_size, shuffle=shuffle, random_state=random_state)
    elif data_type == "tensorflow":
        return split_tensorflow_dataset(df, left_size=left_size, right_size=right_size, shuffle=shuffle, seed=random_state)
    elif data_type == "torch":
        return split_torch_dataset(df, left_size=left_size, right_size=right_size, shuffle=shuffle, seed=random_state)
    elif data_type == "huggingface":
        if not huggingface_dataset:
            raise ValueError("Hugging Face veri seti seçildi ancak dataset adı belirtilmedi.")
        return split_huggingface_dataset(dataset_name=huggingface_dataset, split=split_by, seed=random_state, shuffle=shuffle, ylabel=ylabel, test_size=test_size)
    else:
        raise ValueError(f"Desteklenmeyen veri tipi: {data_type}")