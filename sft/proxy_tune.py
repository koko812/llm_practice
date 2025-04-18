import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "llm-jp/llm-jp-3-980m"
ft_model_name = "llm-jp/llm-jp-3-980m-instruct3"
untune_model_name = "llm-jp/llm-jp-3-980m"
# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
)
ft_model = AutoModelForCausalLM.from_pretrained(
    ft_model_name,
    torch_dtype="auto",
)
untune_model = AutoModelForCausalLM.from_pretrained(
    untune_model_name,
    torch_dtype="auto",
)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
ft_tokenizer = AutoTokenizer.from_pretrained(ft_model_name)

# 元々は英語だが，何言ってるかわからんので日本語に変更した
prompt = "次の文に続く文を生成してください．\n吾輩は猫である．"
jp_messages = [
    {
        "role": "system",
        "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
    },
    {"role": "user", "content": prompt},
]

jp_text = ft_tokenizer.apply_chat_template(
    jp_messages, tokenize=False, add_generation_prompt=True
)

max_length = 50

jp_model_inputs = ft_tokenizer([jp_text], return_tensors="pt").to(base_model.device)
jp_input_ids = jp_model_inputs["input_ids"]
jp_eos_token_id = ft_tokenizer.eos_token_id


def inf_tokens_logits(tokenizer, model, input_ids, max_length):
    tokens = []
    prob_list = []
    jp_generating = True

    for i in range(max_length):
        if jp_generating:
            output = base_model(input_ids)
            next_token_logits = output.logits[0, -1, :]
            sorted_ids = torch.argsort(next_token_logits, descending=True)
            next_token_id = sorted_ids[0]
            prob_list.append(next_token_logits[next_token_id].item())
            next_token = tokenizer.decode(next_token_id)
            tokens.append(next_token)
            if next_token_id.item() == jp_eos_token_id:
                jp_generating = False
            input_ids = torch.cat(
                [input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1
            )

    return tokens, prob_list

tokens, prob_list = inf_tokens_logits(ft_tokenizer, ft_model, jp_input_ids, 50)
print(tokens)
print(prob_list)