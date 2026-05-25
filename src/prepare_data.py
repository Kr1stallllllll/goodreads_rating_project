import kagglehub
import pandas as pd
import os
import shutil

def download_data():
    print("Скачивание датасета Goodreads Books...")
    path = kagglehub.dataset_download("jealousleopard/goodreadsbooks")
    source_file = os.path.join(path, "books.csv")

    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    destination = os.path.join(data_dir, 'goodreads_books.csv')
    shutil.copy2(source_file, destination)
    
    print(f"✅ Датасет загружен в {destination}")
    return destination

if __name__ == "__main__":
    download_data()