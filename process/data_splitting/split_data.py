
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import split_dataset
from datasets import Dataset
def split_sklearn_dataset(X, y, test_size, train_size, shuffle=True, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, shuffle=shuffle, random_state=random_state)
    return X_train, X_test, y_train, y_test

def split_tensorflow_dataset(dataset, left_size, right_size, shuffle=True, seed=42, preferred_backend="tensorflow"):
    train_dataset, test_dataset = split_dataset(dataset, left_size, right_size, shuffle, seed, preferred_backend)
    return train_dataset, test_dataset
#
def split_torch_dataset(dataset, left_size, right_size, shuffle=True, seed=42, preferred_backend="torch"):
    train_dataset, test_dataset = split_dataset(dataset, left_size, right_size, shuffle, seed, preferred_backend)
    return train_dataset, test_dataset
# 'train', ['train', 'test'], 'train+test', 'train[10:20]', 'train[:10%]', 'train[:10%]+train[-80%:]'
# [f'train[{k}%:{k+10}%]' for k in range(0, 100, 10)], datasets.ReadInstruction('train')
# datasets.ReadInstruction('train', to=10, unit='%')
# [datasets.ReadInstruction('train', from_=k, to=k+10, unit='%') for k in range(0, 100, 10)]
def split_huggingface_dataset(dataset, split):
    train_dataset, test_dataset = dataset.load_dataset(dataset, split)
    return train_dataset, test_dataset