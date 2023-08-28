# line-llm-chat

ChatGPT っぽい UI で LINE の LLM モデルとお話できるやつ

( `line-corporation/japanese-large-lm-3.6b-instruction-sft` )

## Installation

`requirements.txt` をお好みの環境で。

```bash
conda create -n linellm python=3.10
conda activate linellm
pip install -r requirements.txt
```

## Usage

初回はモデルのダウンロード&量子化に時間がかかります

```bash
python main.py
# open http://127.0.0.1:8000/
```
