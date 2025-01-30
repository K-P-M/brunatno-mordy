max_seq_length = 2048
dtype = None
load_in_4bit = False
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/teamspace/studios/this_studio/brunatno-mordy-7B-Instruct-Torpeda", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model.generation_config.repetition_penalty = 2.0
tokenizer = get_chat_template(
        tokenizer,
        chat_template="mistral",
)

FastLanguageModel.for_inference(model)

messages = [
    {
        "role": "system",
        "content":"Opodwiadaj stylem bartosza walaszka"
    },
    {
        "role": "user",
        "content":"Jaki jest sens w życiu"
    },
]
tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 200, use_cache = True)