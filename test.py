import os

DIRECTORY = r"C:\Users\RaihanSatar\OneDrive\Desktop\Soft Computing Face Recognition\test"
CATEGORIES = ["test3", "test4"]

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for subfolder in os.listdir(path): 
        subpath = os.path.join(path, subfolder)
        for file in os.listdir(subpath):
            print(file)