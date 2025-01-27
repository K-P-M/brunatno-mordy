import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class DeepSeekInputCreator:
    def __init__(self, api_key: str, temperature: float = 1.3):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

        self.temperature = temperature

        self.system_prompt = """
        Sparafrazuj podany tekst w mowie potocznej podmieniając przekleństwa na bardziej kulturalne słowa. Nie dodawaj na koniec komentarza od siebie. Zwróć wynik w postacji json. Teskt umieść w polu "user", a niezmieniony tekst w polu "assistant".

        EXAMPLE USER INPUT:
        Nazywam się Cezary Baryka, od dwudziestu minut jestem właścicielem tego oto szklanego domu. Powoli zaczynam żałować zakupu. W nocy pizga, w dzień parówa. Zero wentylacji i brak kanalizacji robią swoje.
        
        EXAMPLE JSON OUTPUT:
        {
            "user": Jestem Cezary Baryka, od jakichś 20 minut mam ten szklany dom. I już trochę żałuję, że go kupiłem. W nocy nieubłagalnie zimno, a w dzień upał nie do wytrzymania. Zero przewiewu i brak kanalizacji",
            "assistant": "Nazywam się Cezary Baryka, od dwudziestu minut jestem właścicielem tego oto szklanego domu. Powoli zaczynam żałować zakupu. W nocy pizga, w dzień parówa. Zero wentylacji i brak kanalizacji robią swoje."
        }
        """

    def create_descriptions(self, chunks: list[str]) -> list[dict[str, str]]:
        chunks = chunks[:-1]
        try:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._create_description, chunk) for chunk in chunks]

                descriptions = []
                with tqdm(total=len(chunks), desc="Generating Descriptions", unit="chunk") as pbar:
                    for future in as_completed(futures):
                        descriptions.append(future.result(timeout=10))
                        pbar.update(1)

        except Exception as e:
            with open('train_datasets/error.jsonl', 'a', encoding='utf-8') as file:
                for description in descriptions:
                    json.dump(description, file, ensure_ascii=False)
                    file.write('\n')
            raise

        return descriptions

    def _create_description(self, chunk: str) -> dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": chunk}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={
                'type': 'json_object'
            },
            temperature=self.temperature,
        )

        return json.loads(response.choices[0].message.content)
