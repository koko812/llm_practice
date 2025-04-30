import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "llm-jp/llm-jp-3-3.7B"
sft_model_name = "llm-jp/llm-jp-3-980m-instruct3"
untune_model_name = "llm-jp/llm-jp-3-980m"
# device_map をサンプルの時点から実装してるのはとても偉いと思う（当たり前か）


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    return model


# base_model = load_model(base_model_name)
sft_model = load_model(sft_model_name)
untune_model = load_model(untune_model_name)

# base と sft_tokenizer で ChatTemplate の有無があるので分離
sft_tokenizer = AutoTokenizer.from_pretrained(sft_model_name)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)


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

base_model_inputs = base_tokenizer([prompt], return_tensors="pt").to(untune_model.device)
sft_model_inputs = sft_tokenizer([sft_text], return_tensors="pt").to(sft_model.device)
base_input_ids = base_model_inputs["input_ids"]
sft_input_ids = sft_model_inputs["input_ids"]
base_eos_token_id = base_tokenizer.eos_token_id
sft_eos_token_id = sft_tokenizer.eos_token_id


model_index = ["ft", "untune"]

def simultaneous_generation(tokenizers, models, input_ids, max_length):
    tokens = []
    sft_generating = True

    for i in range(max_length):
        if sft_generating:
            next_tokens = {}
            tmp_logits = [] 
            for j in range(len(models)):
                output = models[j](input_ids[j])

                next_token_logits = output.logits[0, -1, :]
                tmp_logits.append(next_token_logits)
                # この output.logits が実際どんな形なのか分かりたい
                # 次の argsort の仕組みもよくわからないゆえに

                sorted_ids = torch.argsort(next_token_logits, descending=True)
                next_token_id = sorted_ids[0]

                next_token = tokenizers[j].decode(next_token_id)
                next_tokens[model_index[j]] = {
                    next_token: next_token_logits[next_token_id].item()
                }

                # EOS ならば，生成終了
                if next_token_id.item() == sft_eos_token_id:
                    sft_generating = False

                # これまでの入力に次のトークンを繋げてループ先頭へ
                input_ids[j] = torch.cat(
                    [input_ids[j], next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1
                )

            print(f'{i} next_tokens', next_tokens)
            tokens.append(next_tokens)

            all_temp_tokens = []
            for k, v in next_tokens.items():
                all_temp_tokens.append(list(v.keys())[0])
                
            print('--- logits ---')
            print(all_temp_tokens)

            for tkn in all_temp_tokens:
                for j in range(len(tokenizers)):
                    id = tokenizers[j].encode(tkn)[-1]
                    r_tkn = tokenizers[j].decode(id)
                    print(model_index[j], tkn, id, r_tkn, tmp_logits[j][id].item())

                    
            # 混ぜながら inference できてるのはすごくいい話
            # あとはどうしようかという話はいまだに残っている
            # とりあえず，混ぜたやつから，次のやつを inference する必要がある，今までの系列に付け足してね
            # ただ，やっぱりどうするんだろう感が結構強いんだが？
            # beam を何本か作っておくのが吉なんだろうか
            # 全然 proxy tune をする気がなくて笑う
            # まあ，どちらにせよ，両方できる必要があるのでな，それは当然の話
            # やっぱり，llm-jp-eval に組み込むのはなかなかにハードルが高い話だな
            # 前やってた，MoE も完成させたいところは必ずある


            """
            other_logits = {}
            print('\n ----- get mutual logits ----')
            
            for i in range(len(tokenizers)):
                logits = {}
                for token in next_tokens[model_index[i]].keys():
                    id = tokenizers[i].encode(token)[1]
                    print('token', token)
                    print('id', id)
                    logit = tmp_logits[i][id].item()
                    logits[token] = logit
                other_logits[model_index[i]] = logits

            for k,v in other_logits.items():
                print(k,v)

            print('\n ---------------------- \n')
            """    

                
    return tokens


tokenizers = [sft_tokenizer, base_tokenizer]
models = [sft_model, untune_model]
input_ids = [sft_input_ids, base_input_ids]
tokens = simultaneous_generation(tokenizers, models, input_ids, max_length)


print('\n------ result -------')
for t in tokens:
    print(t)
"""
for t, p in zip(tokens, logits_list):
    st = str(t)
    sp = str(p)
    print(st + "\t" + sp)


for ct, cp in zip(candidtate_tokens_list, candidate_logits_list):
    sct = ",\t".join([str(token) for token in ct])
    scp = ",\t".join([str(prob) for prob in cp])
    print(sct + "\t" + scp)
"""