# Simple Retrieval Augmented Generation (RAG) Chatbot using OpenAI GPT Model

**RAGPT** is made using OpenAI GPT Model, LangChain, ChromaDB as our vector Database, and Gradio App for the User Interface

I implemented the project by my own by referencing into some documentations and also YouTube resources (ShoutOut to the Author: `https://www.youtube.com/watch?v=1FERFfut4Uw&t=1515s`) to make this RAGPT Project. In fact I refined many things from the existing project, so it can give me more flexibilities on processing various PDFs.

The chatbot provides three distinct functionalities for you to choose from:

1. **Offline Documents**: Pre-processed and vectorized documents are readily available for integration into your chat sessions, allowing seamless access to their content.

2. **Real-time Uploads**: You can upload new documents during your chat sessions, and the chatbot will process and analyze them in real-time to respond to your questions.

3. **Summarization Requests**: You can ask the chatbot to summarize the content of a PDF in a single interaction, making it easier to retrieve key information quickly and efficiently.

- You can take full advantage of these functionalities through the user interface I've developed using the Gradio app. This intuitive platform allows you to interact with the chatbot seamlessly while also enabling you to upload and modify your own documents within the project’s data.
- The project includes comprehensive guidance on configuring various settings, such as adjusting the GPT model's temperature to achieve optimal performance. Designed for a user-friendly experience, the Gradio interface ensures ease of navigation and interaction.
- Additionally, the model incorporates memory capabilities, allowing it to retain user questions and answers for a more personalized and enhanced experience. With each response, you can access the relevant retrieved content and conveniently view the corresponding PDF for further context.

## RAGPT User Interface

## Project Schema

<div align = "center">
    <img src = "images/RAGPT_schema.png" alt = "Schema">
</div>

Note:

1. This picture is taken from the YouTube resources that I have put above, while the UI for the project I will modify it by myself, and the backend system that is happening behind also I modify it by myself
2. The database here is still manually inserted, maybe further in the future I will try using MongoDB as my database system to upload all the PDFs inside.

## Document Storage

Documents are stored in two separate folders within the `data` directory:

- `data/docs`: For files that should be **pre-processed**.
- `data/docs_2`: For files that you want to **upload**.

## Server Setup

The `serve.py` module leverages these folders to create an **HTTPS server** that hosts the PDF files, making them accessible for user viewing.

## Database Creation

Vector databases (vectorDBs) are generated within the `data` folder, facilitating the project's functionality, once again this is not the saftest part to save your DB, it should be bettr for you to use other DBMS, such as MongoDB or PostgreSQL. This project is more like a learning processes.

## Important Considerations

- The current file management system is meant solely for educational purposes.
- For any production environment, it is highly recommended to develop a more secure and scalable document management solution.
- Please ensure that files are correctly placed in the appropriate directories (data/docs_2 and data/docs) to ensure the project works as expected.

## Running the Project

First of all, you will need to set up the environment and install all necessary dependencies, in fact all of the dependencies are written already in the `requirements.txt`, so you may access it or maybe install it individually for some new dependencies for you. But, here are the general steps:

1. Open `env.example` copy it, then erase the `.example` so it will become `.env` after that fill in your Open Ai credentials.
2. Run python -m venv .venv on your terminal to make a local virtual environment
3. On your repositories run `pip install -r requirements.txt`
4. Run the application by executing

- In Terminal 1:

```
python src\serve.py
```

In Terminal 2:

```
python src\raggpt_app.py
```

5. Chat with your documents.
