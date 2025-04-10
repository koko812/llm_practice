
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = "### exa1:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

print(type(collator))

max_length=100

print(dataset[0])
ex = formatting_prompts_func(dataset[0:10])
print(ex[0])
print(len(ex))

batch_encoding = tokenizer(ex[0], padding='max_length', max_length=max_length)
t_input = collator([batch_encoding])
#print(t_input)


print(batch_encoding[0])
print(t_input)
#training_args = SFTConfig(output_dir="tmp")
#print(training_args)