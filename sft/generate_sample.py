from transformers import AutoModelForCausalLM, AutoTokenizer

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

#inputs は tokenize した直後に gpu に送ってて，合理的だと思う
#tokenizer や collator などはリストが入力されることを基本的に想定してる（はず）
#なので，リストで入力してる．（どっちでも受け付けるように実装しとけよ？）
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

#この場合はアウトプットの id が表示されるようになってるっぽい？
#logits などを出したい場合は，model に直にぶち込めばいい
#出力の logits を制御したい場合は，logits_processor みたいなのを使えばいい 
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)

#ここは正直何をしてるのかよく分かってない
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)