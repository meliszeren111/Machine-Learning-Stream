# SKLEARN DATASETS
from sklearn.datasets import  load_iris, load_diabetes, load_digits, fetch_california_housing
# SEABORN DATASETS
import seaborn as sns
penguins_df = sns.load_dataset('penguins')
titanic_df = sns.load_dataset('titanic')
tips_df = sns.load_dataset('tips')
# TORCHVISION DATASETS
from torchvision.datasets import MNIST, CIFAR10
# TENSORFLOW DATASETS
from tensorflow.keras.datasets import mnist, cifar10, imdb
# HUGGINGFACE DATASETS DATASETS
from datasets import load_dataset
dataset = load_dataset("imdb", split="train", streaming=True) # (Internet Movie Database)
