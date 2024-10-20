from langchain.document_loaders import PyPDFLoader
from utils.utilities import count_num_tokens
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai


class Summarizer:
    """
    A class for summarizing PDF documents using OpenAI's ChatGPT engine.

    Attributes:
        None

    Methods:
        summarize_the_pdf:
            Summarizes the content of a PDF file using OpenAI's ChatGPT engine.

        get_llm_response:
            Retrieves the response from the ChatGPT engine for a given prompt.
    """

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initializes the Summarizer class with a text splitter.

        Args:
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
        confidence_threshold: float
    ):
        """
        Summarizes the content of a PDF file using OpenAI's ChatGPT engine.

        Args:
            file_dir (str): The path to the PDF file.
            max_final_token (int): The maximum number of tokens in the final summary.
            token_threshold (int): The threshold for token count reduction.
            gpt_model (str): The ChatGPT engine model name.
            temperature (float): The temperature parameter for ChatGPT response generation.
            summarizer_llm_system_role (str): The system role for the summarizer.
            final_summarizer_llm_system_role (str): The system role for the final summary.
            character_overlap (int): The number of characters to overlap between chunks.
            confidence_threshold (float): The minimum confidence threshold for valid responses.

        Returns:
            str: The final summarized content if it passes the confidence threshold.
        """
        docs = PyPDFLoader(file_dir).load()
        chunked_docs = self.text_splitter.split_documents(docs)
        max_summarizer_output_token = int(max_final_token / len(chunked_docs)) - token_threshold

        full_summary = ""
        counter = 1

        print("Generating the summary..")
        for i, chunk in enumerate(chunked_docs):
            prompt = chunk.page_content
            summarizer_llm_system_role = summarizer_llm_system_role.format(
                max_summarizer_output_token)

            # Get response and check against the confidence threshold
            response = Summarizer.get_llm_response(
                gpt_model=gpt_model,
                temperature=temperature,
                llm_system_role=summarizer_llm_system_role,
                prompt=prompt
            )

            # Confidence check (e.g., word length, or you can define your own criteria)
            if count_num_tokens(response, gpt_model) / max_final_token >= confidence_threshold:
                full_summary += response
                print(f"Chunk {counter} summarized.")
            else:
                print(f"Chunk {counter} failed to meet the confidence threshold.")
            
            counter += 1

        print("\nFull summary token length:", count_num_tokens(
            full_summary, model=gpt_model))

        # Final summary
        final_summary = Summarizer.get_llm_response(
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

        Args:
            gpt_model (str): The ChatGPT engine model name.
            temperature (float): The temperature parameter for ChatGPT response generation.
            llm_system_role (str): The system role for the LLM.
            prompt (str): The input prompt for the ChatGPT engine.

        Returns:
            str: The response content from the ChatGPT engine.
        """
        response = openai.ChatCompletion.create(
            engine=gpt_model,
            messages=[
                {"role": "system", "content": llm_system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1500
        )
        return response.choices[0].message.content
