from wilds import get_dataset
import os

print("Python script is running in the directory:")
print(os.getcwd())
print("Listing contents of the current directory:")
print(os.listdir(os.getcwd()))

dataset = get_dataset(root_dir = "./Data",dataset="rxrx1", download=True)