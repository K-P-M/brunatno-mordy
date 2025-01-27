from data_processing.deepseek_api import DeepSeekInputCreator
import os
import json


def load_chunks():
    chunks = []

    with open('../data_harvesting/data_cleaning/chunked_3.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            chunks.append(data['chunks'])

    return chunks


if __name__ == "__main__":
    api_key = os.environ.get("API_KEY")
    chunks = load_chunks()

    creator = DeepSeekInputCreator(api_key)
    descriptions = creator.create_descriptions(chunks)

    with open('train_datasets/descriptions_manual.jsonl', 'w', encoding='utf-8') as file:
        for description in descriptions:
            json.dump(description, file, ensure_ascii=False)
            file.write('\n')