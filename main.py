import os
import gc
import argparse
import random
import torch
import transformers
import peft
import datasets
import gradio as gr

model = None
tokenizer = None
current_peft_model = None

def load_base_model():
    global model
    print('Loading base model...')
    model = transformers.LlamaForCausalLM.from_pretrained(
        'decapoda-research/llama-7b-hf',
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map={'':0}
    )

def load_tokenizer():
    global tokenizer
    print('Loading tokenizer...')
    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        'decapoda-research/llama-7b-hf',
    )

def load_peft_model(model_name):
    global model
    print('Loading peft model ' + model_name + '...')
    model = peft.PeftModel.from_pretrained(
        model, model_name,
        torch_dtype=torch.float16
    )

def reset_model():
    global model
    global tokenizer
    global current_peft_model

    del model
    del tokenizer

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    model = None
    tokenizer = None
    current_peft_model = None

def generate_text(
    peft_model,
    text, 
    temperature, 
    top_p, 
    top_k, 
    repetition_penalty, 
    max_new_tokens,
    progress=gr.Progress(track_tqdm=True)
):
    global model
    global tokenizer
    global current_peft_model

    if (peft_model == 'None'): peft_model = None

    if (current_peft_model != peft_model):
        if (current_peft_model is None):
            if (model is None): load_base_model()
        else:
            reset_model()
            load_base_model()
            load_tokenizer()

        current_peft_model = peft_model
        if (peft_model is not None):
            load_peft_model(peft_model)

    if (model is None): load_base_model()
    if (tokenizer is None): load_tokenizer()

    assert model is not None
    assert tokenizer is not None

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    generation_config = transformers.GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        num_beams=1,
    )

    with torch.no_grad():
        output = model.generate(  # type: ignore
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=generation_config
        )[0].cuda()

    return tokenizer.decode(output, skip_special_tokens=True).strip()

def tokenize_and_train(
    training_text,
    max_seq_length,
    micro_batch_size,
    gradient_accumulation_steps,
    epochs,
    learning_rate,
    lora_r,
    lora_alpha,
    lora_dropout,
    model_name,
    progress=gr.Progress(track_tqdm=True)
):
    global model
    global tokenizer

    if (model is None): load_base_model()
    if (tokenizer is None): 
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "decapoda-research/llama-7b-hf", add_eos_token=True
        )

    assert model is not None
    assert tokenizer is not None

    tokenizer.pad_token_id = 0

    paragraphs = training_text.split("\n\n\n")
    paragraphs = [x.strip() for x in paragraphs]

    print("Number of samples: " + str(len(paragraphs)))
        
    def tokenize(item):
        assert tokenizer is not None
        result = tokenizer(
            item["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

    def to_dict(text):
        return {"text": text}

    paragraphs = [to_dict(x) for x in paragraphs]
    data = datasets.Dataset.from_list(paragraphs)
    data = data.shuffle().map(lambda x: tokenize(x))

    model = peft.prepare_model_for_int8_training(model)

    model = peft.get_peft_model(model, peft.LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    ))

    output_dir = f"lora-{model_name}"

    print("Training...")

    training_args = transformers.TrainingArguments(
        # Set the batch size for training on each device (GPU, CPU, or TPU).
        per_device_train_batch_size=micro_batch_size, 

        # Number of steps for gradient accumulation. This is useful when the total 
        # batch size is too large to fit in GPU memory. The effective batch size 
        # will be the product of 'per_device_train_batch_size' and 'gradient_accumulation_steps'.
        gradient_accumulation_steps=gradient_accumulation_steps,  

        # Number of warmup steps for the learning rate scheduler. During these steps, 
        # the learning rate increases linearly from 0 to its initial value. Warmup helps
        #  to reduce the risk of very large gradients at the beginning of training, 
        # which could destabilize the model.
        # warmup_steps=100, 

        # The total number of training steps. The training process will end once this 
        # number is reached, even if not all the training epochs are completed.
        # max_steps=1500, 

        # The total number of epochs (complete passes through the training data) 
        # to perform during the training process.
        num_train_epochs=epochs,  

        # The initial learning rate to be used during training.
        learning_rate=learning_rate, 

        # Enables mixed precision training using 16-bit floating point numbers (FP16). 
        # This can speed up training and reduce GPU memory consumption without 
        # sacrificing too much model accuracy.
        fp16=True,  

        # The frequency (in terms of steps) of logging training metrics and statistics 
        # like loss, learning rate, etc. In this case, it logs after every 20 steps.
        logging_steps=20, 

        # The output directory where the trained model, checkpoints, 
        # and other training artifacts will be saved.
        output_dir=output_dir, 

        # The maximum number of checkpoints to keep. When this limit is reached, 
        # the oldest checkpoint will be deleted to save a new one. In this case, 
        # a maximum of 3 checkpoints will be kept.
        save_total_limit=3,  
    )


    trainer = transformers.Trainer(
        # The pre-trained model that you want to fine-tune or train from scratch. 
        # 'model' should be an instance of a Hugging Face Transformer model, such as BERT, GPT-2, T5, etc.
        model=model, 

        # The dataset to be used for training. 'data' should be a PyTorch Dataset or 
        # a compatible format, containing the input samples and labels or masks (if required).
        train_dataset=data, 

        # The TrainingArguments instance created earlier, which contains various 
        # hyperparameters and configurations for the training process.
        args=training_args, 

        # A callable that takes a batch of samples and returns a batch of inputs for the model. 
        # This is used to prepare the input samples for training by batching, padding, and possibly masking.
        data_collator=transformers.DataCollatorForLanguageModeling( 
            tokenizer,  
            # Whether to use masked language modeling (MLM) during training. 
            # MLM is a training technique used in models like BERT, where some tokens in the 
            # input are replaced by a mask token, and the model tries to predict the 
            # original tokens. In this case, MLM is set to False, indicating that it will not be used.
            mlm=False, 
        ),
    )

    model.config.use_cache = False
    result = trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(output_dir)

    del data
    reset_model()

    return result

def random_hyphenated_word():
    word_list = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig']
    word1 = random.choice(word_list)
    word2 = random.choice(word_list)
    return word1 + '-' + word2

def training_tab():
    with gr.Tab("Finetuning"):

        with gr.Column():
            training_text = gr.Textbox(lines=12, label="Training Data", info="Each sequence must be separated by 2 blank lines")

            max_seq_length = gr.Slider(
                minimum=1, maximum=4096, value=512,
                label="Max Sequence Length", 
                info="The maximum length of each sample text sequence. Sequences longer than this will be truncated."
            )

        with gr.Row():
            with gr.Column():
                micro_batch_size = gr.Slider(
                    minimum=1, maximum=100, value=1, 
                    label="Micro Batch Size", 
                    info="The number of examples in each mini-batch for gradient computation. A smaller micro_batch_size reduces memory usage but may increase training time."
                )

                gradient_accumulation_steps = gr.Slider(
                    minimum=1, maximum=10, value=1, 
                    label="Gradient Accumulation Steps", 
                    info="The number of steps to accumulate gradients before updating model parameters. This can be used to simulate a larger effective batch size without increasing memory usage."
                )

                epochs = gr.Slider(
                    minimum=1, maximum=100, value=1, 
                    label="Epochs",
                    info="The number of times to iterate over the entire training dataset. A larger number of epochs may improve model performance but also increase the risk of overfitting.")

                learning_rate = gr.Slider(
                    minimum=0.00001, maximum=0.01, value=3e-4,
                    label="Learning Rate",
                    info="The initial learning rate for the optimizer. A higher learning rate may speed up convergence but also cause instability or divergence. A lower learning rate may require more steps to reach optimal performance but also avoid overshooting or oscillating around local minima."
                )

            with gr.Column():
                lora_r = gr.Slider(
                    minimum=1, maximum=16, value=8, 
                    label="LoRA R",
                    info="The rank parameter for LoRA, which controls the dimensionality of the rank decomposition matrices. A larger lora_r increases the expressiveness and flexibility of LoRA but also increases the number of trainable parameters and memory usage."
                )

                lora_alpha = gr.Slider(
                    minimum=1, maximum=128, value=16, 
                    label="LoRA Alpha",
                    info="The scaling parameter for LoRA, which controls how much LoRA affects the original pre-trained model weights. A larger lora_alpha amplifies the impact of LoRA but may also distort or override the pre-trained knowledge."
                )
                
                lora_dropout = gr.Slider(
                    minimum=0, maximum=1, value=0.01,
                    label="LoRA Dropout",
                    info="The dropout probability for LoRA, which controls the fraction of LoRA parameters that are set to zero during training. A larger lora_dropout increases the regularization effect of LoRA but also increases the risk of underfitting."
                )

                with gr.Column():
                    model_name = gr.Textbox(
                        lines=1, label="LoRA Model Name", value=random_hyphenated_word()
                    )

                    with gr.Row():
                        train_btn = gr.Button(
                            "Train", variant="primary", label="Train", 
                        )

                        abort_button = gr.Button(
                            "Abort", label="Abort", 
                        )
    
        output_text = gr.Text("Training Status")

        train_progress = train_btn.click(
            fn=tokenize_and_train,
            inputs=[
                training_text,
                max_seq_length,
                micro_batch_size,
                gradient_accumulation_steps,
                epochs,
                learning_rate,
                lora_r,
                lora_alpha,
                lora_dropout,
                model_name
            ],
            outputs=output_text
        )

        abort_button.click(None, None, None, cancels=[train_progress])

def inference_tab():
    with gr.Tab("Inference"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                        lora_model = gr.Dropdown(
                            label="LoRA Model",
                        )
                        refresh_models_list = gr.Button(
                            "Reload Models",
                            elem_id="refresh-button"
                        )
                inference_text = gr.Textbox(lines=7, label="Input Text")   
            inference_output = gr.Textbox(lines=12, label="Output Text")
        with gr.Row():
            with gr.Column():
                #  temperature, top_p, top_k, repeat_penalty, max_new_tokens
                temperature = gr.Slider(
                    minimum=0.01, maximum=1.99, value=0.1, step=0.01,
                    label="Temperature",
                    info="Controls the 'temperature' of the softmax distribution during sampling. Higher values (e.g., 1.0) make the model generate more diverse and random outputs, while lower values (e.g., 0.1) make it more deterministic and focused on the highest probability tokens."
                )

                top_p = gr.Slider(
                    minimum=0, maximum=1, value=0.75, step=0.01,
                    label="Top P",
                    info="Sets the nucleus sampling threshold. In nucleus sampling, only the tokens whose cumulative probability exceeds 'top_p' are considered  for sampling. This technique helps to reduce the number of low probability tokens considered during sampling, which can lead to more diverse and coherent outputs."
                )

                top_k = gr.Slider(
                    minimum=0, maximum=200, value=50, step=1,
                    label="Top K",
                    info="Sets the number of top tokens to consider during sampling. In top-k sampling, only the 'top_k' tokens with the highest probabilities are considered for sampling. This method can lead to more focused and coherent outputs by reducing the impact of low probability tokens."
                )

                repeat_penalty = gr.Slider(
                    minimum=0, maximum=2.5, value=1.2, step=0.01,
                    label="Repeat Penalty",
                    info="Applies a penalty to the probability of tokens that have already been generated, discouraging the model from repeating the same words or phrases. The penalty is applied by dividing the token probability by a factor based on the number of times the token has appeared in the generated text."
                )

                max_new_tokens = gr.Slider(
                    minimum=0, maximum=4096, value=50, step=1,
                    label="Max New Tokens",
                    info="Limits the maximum number of tokens generated in a single iteration."
                )
            with gr.Column():
                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate", variant="primary", label="Generate", 
                    )
            
        generate_btn.click(
            fn=generate_text,
            inputs=[
                lora_model,
                inference_text,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                max_new_tokens
            ],
            outputs=inference_output,
        )

        def update_models_list():
            return gr.Dropdown.update(choices=["None"] + [
                d for d in os.listdir() if os.path.isdir(d) and d.startswith('lora-')
            ], value="None")

        refresh_models_list.click(
            update_models_list,  
            inputs=None, 
            outputs=lora_model,
        )

with gr.Blocks(
    css="#refresh-button { max-width: 32px }", 
    title="Simple LLaMA Finetuner") as demo:
        training_tab()
        inference_tab()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple LLaMA Finetuner")
    parser.add_argument("-s", "--share", action="store_true", help="Enable sharing of the Gradio interface")
    args = parser.parse_args()

    demo.queue().launch(share=args.share)
