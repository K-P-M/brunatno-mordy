from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "speakleash/Bielik-7B-Instruct-v0.1",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
# format
# chat =[
#     {
#         "role": "system",
#     "content":"Odpowiadaj w stylu Bartosza Walaszka"
# },
# {
#     "role": "user",
#     "content": "Jestem Cezary Baryka, od jakichś 20 minut"
# },
# {
#     "role": "assistant",
#     "content": "I już trochę żałuję, że go kupiłem. W nocy zimno jak cholera, a w dzień upał nie do wytrzymania. Zero przewiewu i brak kanalizacji"
# },
# ]



tokenizer = get_chat_template(
    tokenizer,
    chat_template = "mistral",
)

def formatting_prompts_func(examples):
    convos = examples
    texts = [tokenizer.apply_chat_template(convo, tokenize = False) for convo in convos]
    return { "text" : texts }


train_data =

loaded_data = formatting_prompts_func(train_data)



