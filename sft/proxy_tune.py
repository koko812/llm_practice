import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "llm-jp/llm-jp-3-980m"
sft_model_name = "llm-jp/llm-jp-3-3.7B-instruct3"
untune_model_name = "llm-jp/llm-jp-3-980m"
# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
    )
    return model

base_model = load_model(base_model_name)
sft_model = load_model(sft_model_name)
untune_model = load_model(untune_model_name)

# base と sft_tokenizer で ChatTemplate の有無があるので分離
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_name)

prompt = "次の文に続く文を生成してください．\n吾輩は猫である．"
sft_messages = [
    {
        "role": "system",
        "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。",
    },
    {"role": "user", "content": prompt},
]

sft_text = sft_tokenizer.apply_chat_template(
    sft_messages, tokenize=False, add_generation_prompt=True
)

max_length = 20

sft_model_inputs = sft_tokenizer([sft_text], return_tensors="pt").to(base_model.device)
sft_input_ids = sft_model_inputs["input_ids"]
sft_eos_token_id = sft_tokenizer.eos_token_id


def inf_tokens_logits(tokenizer, model, input_ids, max_length):
    tokens = []
    candidate_tokens_list = []
    candidate_logits_list = []
    logits_list = []
    sft_generating = True

    for i in range(max_length):
        if sft_generating:
            output = base_model(input_ids)

            next_token_logits = output.logits[0, -1, :]
            sorted_ids = torch.argsort(next_token_logits, descending=True)

            next_token_id = sorted_ids[0]
            candidate_token_ids = sorted_ids[:3]

            candidate_logits = [next_token_logits[id].item() for id in candidate_token_ids]
            candidate_logits_list.append(candidate_logits)
            candidate_tokens = [tokenizer.decode(id) for id in candidate_token_ids]
            candidate_tokens_list.append(candidate_tokens)

            logits_list.append(next_token_logits[next_token_id].item())
            next_token = tokenizer.decode(next_token_id)
            tokens.append(next_token)

            # EOS ならば，生成終了
            if next_token_id.item() == sft_eos_token_id:
                sft_generating = False

            # これまでの入力に次のトークンを繋げてループ先頭へ
            input_ids = torch.cat(
                [input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1
            )

    return tokens, logits_list, candidate_tokens_list, candidate_logits_list

tokens, logits_list, candidtate_tokens_list, candidate_logits_list = \
    inf_tokens_logits(sft_tokenizer, sft_model, sft_input_ids, max_length)

print(tokens)
print(logits_list)

for t,p in zip(tokens, logits_list):
    st = str(t)
    sp = str(p)
    print(st+'\t'+sp)


for ct,cp in zip(candidtate_tokens_list, candidate_logits_list):
    sct = ',\t'.join([str(token) for token in ct])
    scp = ',\t'.join([str(prob) for prob in cp])
    print(sct+'\t'+scp)