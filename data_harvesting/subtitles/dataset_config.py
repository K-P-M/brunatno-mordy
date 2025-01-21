"""File contains class that gives access to all paths to files."""


class DatasetConfig:

    """Class that contains all paths to files.

    This class defines constants for paths to various JSONL files used
    in the data harvesting pipeline. These include files for processing
    captions and scraped movie data.
    """

    TO_WHISPE_R = "../data/movies_to_whisper.jsonl"
    ALL_SCRAPED_MOVIES = "../data/all_movies.jsonl"
    NON_GENERATED = "../data/bomba_captions.jsonl"
    TRANSCRIPTION = "../data/transcriptions.jsonl"
