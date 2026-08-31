#####                  #####
#####   For Chunking   #####
#####                  #####

#####              #####
#####   Imports    #####
#####              #####
import re
import numpy as np


def fixed_size_chunking(text:str, chunk_size:int = 200) -> list[str]:
    """
    Splits a given text into chunksof a specified fixed size.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int): The maximum number of words pre chunk.

    Returns:
        List[str]: A list of text chunks, each containing up to 'chunk_size' words
    """

    # Split the input text into individual words
    # Split on runs of 3+ newlines or spaces to preserve paragraph structure
    text_words = re.split(r"\n{3,}| +", text)

    # Initialize a list to hold the chunks of words
    chunks = []

    # Iterate over the word indices in steps of 'chunk_size'
    for i in range(0, len(text_words), chunk_size):
        chunk_of_words = text_words[i:i+chunk_size] # From the index where it's currently at until the index after adding the chunk_size chosen

        # Join the selected words into a single string with spaces in between
        chunk = " ".join(chunk_of_words)

        # Add the chunk to the list of chunks
        chunks.append(chunk)

    # Return the list of word chunks
    return chunks


def fixed_size_chunking_with_overlap(text: str, chunk_size: int=200, overlap_factor: float=0.2) -> list[str]:
    """
    Splits a given text into chunks of a fixed size with a specified overlap fraction between consecutive chunks.

    Parameters:
    - text (str): The input text to be split into chunks.
    - chunk_size (int): The number of words each chunk should contain.
    - overlap_fraction (float): The fraction of the chunk size that should overlap with the adjacent chunk.
      For example, an overlap_fraction of 0.2 means 20% of the chunk size will be used as overlap.

    Returns:
    - List[str]: A list of chunks (each a string) where each chunk might overlap with its adjacent chunk.
    """

    # Split the text into individial words
    # Split on runs of 3+ newlines or spaces to preserve paragraph structure
    text_words = re.split(r"\n{3,}| +", text)

    # Initialize the list to hold the chunks
    chunks = []

    # Calculate the overlap step as an integer number of words
    overlap_chunks = int(chunk_size * overlap_factor)
    
    # Iterate over the text in steps of chunk_size to create chunks

    for i in range(0, len(text_words), chunk_size):

        # Determine the start and end indices for the current chunk,
        # taking into account the overlap with the previous chunk
        chunk_words = text_words[max(i-overlap_chunks,0):i+chunk_size]

        # Join the selected words into a single chunk
        chunk = " ".join(chunk_words)

        # Append the chunk into the list of chunks
        chunks.append(chunk)

    # Return the list of chunks
    return chunks


def variable_size_chunking_character_based(text: str, characters: list[str], regex_mode: bool = False):
    """
    Splits a given text into chunks based on certain characters or based on regular expressions.
    By default it is set to take a list of characters, if you want to use regular expressions just declare regex_mode=True and then pass your regular expressions inside the characters list.

    Args:
        text (str): The text to be split into chunks.
        characters (list[str]): A list of characters/regular expressions depending on the use case.
        regex_mode (bool, optional): A parameter to change the behaviour of the function in order to support regular expressions. Defaults to False.

    Returns:
        list[str]: A list of chunks (each a string), according to the criteria set by the user.
    """
        
    if regex_mode:
        # Trust that the user knows how to write regular expressions
        create_regular_expression = "|".join(characters)
    else:
        # This means that the function is on character based mode
        create_regular_expression = "|".join(re.escape(c) for c in characters) # Escape and join all characters

    # Split the document into chunks
    chunks = re.split(create_regular_expression, text)

    # Removing empty chunks in case they appear
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    # Extra safety in case text was blank
    if not chunks:
        raise ValueError(f"text passed into variable_size_chunking_character_based function is empty")

    return chunks


def special_mixed_chunking(text: str, min_char_chunk: int = 30):
    """
    Splits a given text into chunks based on double newlines, merging short chunks
    that fall below min_char_chunk characters with the following chunk.

    Designed to handle the document structure of the PT_wikiRAG dataset.

    Args:
        text (str): The input text to be split.
        min_char_chunk (int, optional): Minimum character length a chunk must have
            to be kept as a standalone chunk. Defaults to 30.

    Returns:
        list[str]: A list of text chunks.
    """
    min_characters = min_char_chunk

    chunks = re.split(r"\n{2,}", text)

    new_chunks = []
    chunk_buffer = ""
    for chunk in chunks:
        if chunk_buffer != "":
            new_chunk = chunk_buffer + "\n" + chunk
            new_chunks.append(new_chunk)
            chunk_buffer = ""
        if len(chunk) < min_characters:
            chunk_buffer += chunk
        else:
            new_chunks.append(chunk)
    
    return new_chunks

    
    