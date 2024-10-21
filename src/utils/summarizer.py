from langchain.document_loaders import PyPDFLoader
from utils.utilities import count_num_tokens
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

class Summarizer:
    """
    A class for summarizing PDF documents using OpenAI's ChatGPT engine.
    """

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initializes the Summarizer class with a text splitter.
        Now accepts character_overlap as a parameter.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )

    def summarize_the_pdf(
        self,
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gpt_model: str,
        temperature: float,
        summarizer_llm_system_role: str,
        final_summarizer_llm_system_role: str,
        character_overlap: int,
    ):
        """
        Summarizes the content of a PDF file using OpenAI's ChatGPT engine.
        """
        docs = PyPDFLoader(file_dir).load()
        chunked_docs = self.text_splitter.split_documents(docs)
        max_summarizer_output_token = int(max_final_token / len(chunked_docs)) - token_threshold

        full_summary = ""
        counter = 1

        print("Generating the summary..")
        for i, chunk in enumerate(chunked_docs):
            prompt = chunk.page_content
            formatted_role = summarizer_llm_system_role.format(
                max_summarizer_output_token)

            # Get response
            response = self.get_llm_response(
                gpt_model=gpt_model,
                temperature=temperature,
                llm_system_role=formatted_role,
                prompt=prompt
            )

            full_summary += response
            print(f"Chunk {counter} summarized.")
            
            counter += 1

        print("\nFull summary token length:", count_num_tokens(
            full_summary, model=gpt_model))

        # Final summary
        final_summary = self.get_llm_response(
            gpt_model=gpt_model,
            temperature=temperature,
            llm_system_role=final_summarizer_llm_system_role,
            prompt=full_summary
        )

        return final_summary

    @staticmethod
    def get_llm_response(gpt_model: str, temperature: float, llm_system_role: str, prompt: str):
        """
        Retrieves the response from the ChatGPT engine for a given prompt.
        """
        response = openai.ChatCompletion.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1500
        )

        return response['choices'][0]['message']['content']
