from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#元々は英語だが，何言ってるかわからんので日本語に変更した
prompt = "LLMとはなんですか．100文字以内で答えてください．"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
input_ids = model_inputs['input_ids']
eos_token_id = tokenizer.eos_token_id
max_length = 50

with torch.no_grad():
    for i in range(max_length):
        output = model(input_ids)
        next_token_logits = output.logits[0, -1, :]
        sorted_ids = torch.argsort(next_token_logits, descending=True)
        next_token_id = sorted_ids[0]
        #print(next_token_id)
        #print(' '.join( tokenizer.decode(sorted_ids[:5]) ))
        print(tokenizer.decode(next_token_id))
        #print(next_token_id.item(), eos_token_id)
        if next_token_id.item() == eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)


"""
print('--- outputs ---')
print(output)
print('--- ids ---')
print(generated_ids)


response = tokenizer.batch_decode(torch.squeeze( generated_ids ), skip_special_tokens=True)
print(response)


print('--- inputs ---')
print(model_inputs['input_ids'].shape)
print('--- outputs ---')
print(output.logits.shape)
generated_ids = torch.max( output.logits , axis = 2).indices
print('--- ids ---')
print(generated_ids)

print(tokenizer.batch_decode(model_inputs['input_ids']))
print(tokenizer.batch_decode(generated_ids))
"""