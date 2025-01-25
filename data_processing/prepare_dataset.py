import json

def load_and_format_training_data(file_path: str) -> list[list[dict]]:
    system_context = {
        "role": "system",
        "content": "Odpowiadaj w stylu Bartosza Walaszka."
    }

    formatted_data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            formatted_data.append([
                system_context,
                {"role": "user", "content": data["user"]},
                {"role": "assistant", "content": data["assistant"]}
            ])

    return formatted_data


if __name__ == '__main__':
    file_path = 'train_datasets/descriptions.jsonl'
    formatted_data = load_and_format_training_data(file_path)
    print(formatted_data)