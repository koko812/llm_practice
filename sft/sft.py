from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import os

os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&Bプロジェクトの名前を指定
# このクソが勝手に checkpoint を保存しやがって wandb のストレージが吹っ飛ぶ
# huggingface public に保存しましょう
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルチェックポイントをログ
os.environ["WANDB_LOG_MODEL"] = "false"  # すべてのモデルチェックポイントをログ

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = SFTConfig(
    max_length=512,
    output_dir="tmp",
    report_to="wandb",
    # こいつで保存数を制限できる
    save_total_limit=3
)

# adnis だと，モデル一つに乗り切らなかった，残念
# 10GB しかないらしくてやばみ
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)
# なぜか device を指定しないと，trainset の総数などがバグってうまく訓練できないというカス仕様
# ロードするときに device_map = auto みたいなのを入れたら良かったのかもしれない
# とりあえず学習はできたので評価していきたい

# なんか urakasumi がちょっと空いたので，種々の並列化手法を試しどれくらい早くなるかを試してみよう
trainer.train()
