from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import os

os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # W&Bプロジェクトの名前を指定
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # すべてのモデルチェックポイントをログ

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = SFTConfig(
    max_length=512,
    output_dir="tmp",
    report_to="wandb",
    save_total_limit=3
)

# adnis だと，モデル一つに乗り切らなかった，残念
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
)
trainer.train()