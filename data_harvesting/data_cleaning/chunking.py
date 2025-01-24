import nltk
from nltk.tokenize import sent_tokenize
import jsonlines
from data_harvesting.subtitles.dataset_config import DatasetConfig


class ChunkingTranscription:
    def __init__(self, file):
        self.file = file
        nltk.download('punkt_tab')

    def chunking(self):
        with jsonlines.open(self.file, mode='r') as f:
            transcriptions = [obj for obj in f]

        chunks = []

        for t_obj in transcriptions:
            transcription = t_obj['transcription']

            splits = sent_tokenize(transcription, language='polish')
            i = 1
            combined_chunks = ' '
            for chunk in splits:
                if i <= 8:
                    combined_chunks += f" {chunk}"
                    i = i + 1
                else:
                    i = 1

                    chunks.append({
                            "epizod": t_obj['title'],
                            "chunks": combined_chunks
                    })
                    combined_chunks = ''
            print(chunks)
            with jsonlines.open('chunked_2.jsonl', mode='w') as writer:
                for processed in chunks:
                    writer.write(processed)


if __name__ == '__main__':
    s = ChunkingTranscription(DatasetConfig.CAPTIONS)
    s.chunking()
