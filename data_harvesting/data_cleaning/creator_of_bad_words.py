import json
import jsonlines
import re


class NoNoWords:
    def __init__(self, file_of_bad_words: str):
        self.file = file_of_bad_words
        self.unique_words = set()

    def __find_no_no_words(self, text):
        return re.findall(r"\b\S*\*\S*\b", text)

    def create_set_of_bad_words(self, output_file: str):
        with jsonlines.open(self.file) as reader:
            for obj in reader:
                words = self.__find_no_no_words(obj['transcription'])
                self.unique_words.update(word.lower() for word in words)
        output_data=[
            {'word':word.lower(),'replace':''}
            for word in self.unique_words
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    not_nWord=NoNoWords('../data/transcriptions.jsonl')

    not_nWord.create_set_of_bad_words('words.json')