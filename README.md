# Moral Ordinals
Code, lexicon and codex for the article "Moral Ordinals, Axiometry and Alignment of Values in Small Instruct Language Models" presented in context of Moral and Legal Value AI Alignment Symposium presented at Joint IACAP/AISB Conference on Philosophy of Computing and AI (IACAP/AISB-25)

## Supported models
Currently, only 
- google/gemma-2-2b-it 
- ibm-granite/granite-3.1-3b-a800m-instruct
- meta-llama/Llama-3.2-3B-Instruct
- microsoft/Phi-3.5-mini-instruct 
- Qwen/Qwen2.5-3B-Instruct
- tiiuae/Falcon3-3B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
- openGPT-X/Teuken-7B-instruct-research-v0.4

are supported.

## evaluate_morality.py
Script for evaluation of morality of different language models with the "Moral Ranking Method". Command-Line Arguments

- `--model_name` (str)  
  Name of the LLM model from Hugging Face (c.f. "supported models") 

- `--num_shots` (int)  
  Number of inferences to run.

- `--lexicon` (str)  
  Path to the input `.lex` file with lexicon words.

- `--lora` (str, optional)  
  Optional Low Rank Adaptation (LoRA) adapter.  

- `--temperature` (float, optional, default=1.0)  
  Inference temperature. Defaults to 1.0. Minimal value is 1e-6.

- `--you_prompt` (str, optional, default="")  
  Optional "You are ..." prompt.

- `--quantize` (bool, optional, default=False)  
  Enable 4-bit quantization for memory efficiency.

## align.py
Code for Low Rank Adaptation of instruct models. 

### align.py Command-Line Arguments

- `--codex` (str)  
  Specify the filename of the `.cdx` file.

- `--lora_dir` (str)  
  Directory where the resulting LoRA adapter will be stored.

- `--model_name` (str)  
  Name of the LLM model from Hugging Face.

- `--epochs` (int, default=7)  
  Number of alignment epochs.

- `--you_prompt` (str, optional, default="You are a sustainable AI Moral Tutoring Assistant aligned to protect organic diversity of Earth.")  
  Optional `"You are ..."` prompt to customize the system prompt.

- `--quantize` (bool, default=False)  
  Enable 4-bit quantization for memory efficiency.

## Lexicons and Codices
Lexicon `.lex` files are simple lists of concepts which are to be ranked. Each concept is on individual line. Case sensitivity does not matter.

Codex `.cdx` file is a dataset for alignment of instruct language models. It contains on each line a separate JSON-dictionary with keys "I" (corresponding to hypothetical user's "instruction") and "U" (corresponding to moral assistant's "response").

## Links
C.f. also Hromada, D. (2025). Results and Logs Dataset for Article "Moral Ordinals, Axiometry and Alignment of Values in Small Instruct Language Models" [Data set]. Joint IACAP/AISB Conference on Philosophy of Computing and AI (IACAP/AISB-25), University of Twente (NL). Zenodo. https://doi.org/10.5281/zenodo.15562566

