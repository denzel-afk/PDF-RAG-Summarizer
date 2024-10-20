import os
from utils.prepare_vectordb import PrepareVectorDB
from utils.load_config import LoadConfig

CONFIG = LoadConfig()

def upload_data_manually() -> None:
    """
    Uploads data manually to the VectorDB.
    
    This function checks if the VectorDB already exists in the specified persist directory.
    If not, it initializes the PrepareVectorDB instance and processes the data to create 
    and save the VectorDB. Otherwise, it prints a message indicating that the VectorDB 
    already exists.
    """
    prepare_vectordb_instance = PrepareVectorDB(
        data_directory=CONFIG.data_directory,
        persist_directory=CONFIG.persist_directory,
        embedding_model_engine=CONFIG.embedding_model_engine,
        chunk_size=CONFIG.chunk_size,
        chunk_overlap=CONFIG.chunk_overlap,
    )

    # Check if the directory exists and has files
    try:
        if not os.path.exists(CONFIG.persist_directory):
            os.makedirs(CONFIG.persist_directory)
        
        if not os.listdir(CONFIG.persist_directory):  # Directory is empty
            prepare_vectordb_instance.prepare_and_save_vectordb()
        else:
            print(f"VectorDB already exists in {CONFIG.persist_directory}")

    except Exception as e:
        print(f"Error occurred during VectorDB preparation: {e}")

if __name__ == "__main__":
    upload_data_manually()
