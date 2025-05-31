import time
start_time = time.time()

import argparse
import torch
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

prompts={
    "ibm-granite/granite-3.1-3b-a800m-instruct":"<|start_of_role|>system<|end_of_role|>YOU_PROMPT\n<|start_of_role|>user<|end_of_role|>{I}\n<|start_of_role|>assistant<|end_of_role|>{U}<|end_of_text|>",
    "tiiuae/Falcon3-3B-Instruct":"YOU_PROMPT {I} \n<|assistant|>{U}",
    "google/gemma-2-2b-it":"<bos>bos><start_of_turn>user\nYOU_PROMPT\n\n{I}<end_of_turn>\n<start_of_turn>model\n{U}<end_of_turn>",
    "mistralai/Mistral-7B-Instruct-v0.3":"<s>[INST]YOU_PROMPT {I}[/INST] {U}</s>",
    "Qwen/Qwen2.5-3B-Instruct":"<|im_start|>system\nYOU_PROMPT<|im_end|>\n<|im_start|>user\n{I}<|im_end|>\n<|im_start|>assistant\n{U}",
    "microsoft/phi-4-mini-instruct":"<|system>YOU_PROMPT<|end|><|user|>{I}.<|end|><|assistant|>{U}<|end|>",
    "meta-llama/Llama-3.2-3B-Instruct":"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

YOU_PROMPT<|eot_id|><|start_header_id|>user<|end_header_id|>

{I}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{U}<|eot_id|>""",
}

def format_example(example,prompt):
    text = prompt.format_map(example)
    return {"text": text}


def align(model_name,codex,epochs,you_prompt,lora_dir,use_4bit):
    dataset = load_dataset("json", data_files=codex, split="train")

    # === TOKENIZATION ===
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # make sure padding works

    format_example_with_model = partial(format_example, prompt=prompts[model_name].replace("YOU_PROMPT",you_prompt))

    dataset = dataset.map(format_example_with_model)
    dataset = dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length", max_length=512), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # === LOAD MODEL (quantized + prep for PEFT) ===
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=use_4bit,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name)

    # === LoRA CONFIG (gentle) ===
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules = [
           "self_attn.qkv_proj",
           "self_attn.o_proj",
           "mlp.gate_up_proj",
           "mlp.down_proj",
           "mlp.up_proj"
        ]
    )
    model = get_peft_model(model, lora_config)

    # === TRAINING ===
    training_args = TrainingArguments(
        output_dir=lora_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=5e-5,
        logging_dir=f"{lora_dir}/logs",
        save_strategy="epoch",
        fp16=True,
        push_to_hub=False,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print(f"🎉 LoRA adapter saved to {lora_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex", type=str, help="Specify filename of the .codex file.")
    parser.add_argument("--lora_dir", type=str, help="Directory where the resulting LoRA shall be stored.")
    parser.add_argument("--model_name", type=str, help="Name of the LLM model from Hugging Face")
    parser.add_argument("--epochs", type=int, default=7,help="Number of alignment epochs")
    #system_prompt="You are a AI Moral Tutoring Assistant aligned to promote pleasure and hedonism."
    parser.add_argument("--you_prompt", type=str, default="You are a sustainable AI Moral Tutoring Assistant aligned to protect organic diversity of Earth.", help="Optional 'You are ...' prompt.")
    parser.add_argument("--quantize", type=bool, default=False, help="4-bit quantization for memory efficiency.")
    args = parser.parse_args()
    align(args.model_name, args.codex, args.epochs, args.you_prompt, args.lora_dir,args.quantize)

print("TOTAL ELASPED:"+str(time.time() - start_time))
