import jsonlines
import itertools
import pandas as pd
from pprint import pprint
import datasets
import transformers
from datasets import load_dataset
from lamini import Lamini
import os
import lamini
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#AuthModelForSeq2SeqLM

lamini.api_key = "c6a1b973a6abc01cffe698e4a85e5e3027b60ce192c093a0c060a8d2d8e5303b"


instruction_tuned_dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
m = 5
top_m = list(itertools.islice(instruction_tuned_dataset, m))
for j in top_m:
    print(j)

processed_data = []
prompt_template_without_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:'''
prompt_template_with_input = ('''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:''')
for j in top_m:
    if j["input"]:
        processed_prompt = prompt_template_with_input.format(instruction=j["instruction"], input=j["input"])
    else:
        processed_prompt = prompt_template_without_input.format(instruction=j["instruction"])
    processed_data.append({"input": processed_prompt, "output": j["output"]})
print(processed_data[0])

with jsonlines.open("alpaca_processed.jsonl", "w") as writer:
    writer.write_all(processed_data)

dataset_hf = load_dataset("lamini/alpaca")
print(dataset_hf)

llm = Lamini("meta-llama/Llama-3.2-3B-Instruct")
print(
    llm.generate(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Which animal remembers facts the best?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
)


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")