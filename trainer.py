import os
import gc
import torch
import transformers
import peft
import datasets
from contextlib import nullcontext

from config import (
    HAS_CUDA, 
    MODEL, 
    DEVICE_MAP, 
    TRAINING_PARAMS, 
    LORA_TRAINING_PARAMS, 
    GENERATION_PARAMS
)

class Trainer():
    def __init__(self):
        self.model = None
        self.model_name = None
        self.lora_name = None
        self.loras = {}

        self.tokenizer = None
        self.trainer = None

        self.should_abort = False

    def unload_model(self):
        del self.model
        del self.tokenizer

        self.model = None
        self.model_name = None
        self.tokenizer = None

        if (HAS_CUDA):
            with torch.no_grad():
                torch.cuda.empty_cache()

        gc.collect()

    def load_model(self, model_name, force=False, **kwargs):
        assert model_name is not None

        if (model_name == self.model_name and not force):
            return
        
        if (self.model is not None):
            self.unload_model()

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=DEVICE_MAP,
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )
        #Clear the collection that tracks which adapters are loaded, as they are associated with self.model
        self.loras = {}

        if model_name.startswith('decapoda-research/llama'):
            self.tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token_id = 0
        self.model_name = model_name

    def load_lora(self, lora_name, replace_model=True):
        assert self.model is not None
        assert lora_name is not None

        if (lora_name == self.lora_name):
            return

        if lora_name in self.loras:
            self.lora_name = lora_name
            self.model.set_adapter(lora_name)
            return
        
        peft_config = peft.PeftConfig.from_pretrained(lora_name)
        if not replace_model:
            assert peft_config.base_model_name_or_path == self.model_name
        
        if peft_config.base_model_name_or_path != self.model_name:
            self.load_model(peft_config.base_model_name_or_path)

        assert self.model_name is not None
        assert self.model is not None
        
        if hasattr(self.model, 'load_adapter'):
            self.model.load_adapter(lora_name, adapter_name=lora_name)
        else:
            self.model = peft.PeftModel.from_pretrained(self.model, lora_name, adapter_name=lora_name)
            
        self.model.set_adapter(lora_name)
        if (self.model_name.startswith('cerebras')):
            self.model.half()

        self.lora_name = lora_name
        self.loras[lora_name] = True

    def unload_lora(self):
        self.lora_name = None

    def generate(self, prompt, **kwargs):
        assert self.model is not None
        assert self.model_name is not None
        assert self.tokenizer is not None

        kwargs = { **GENERATION_PARAMS, **kwargs }

        inputs = self.tokenizer(str(prompt), return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        if self.model.config.pad_token_id is None:
            kwargs['pad_token_id'] = self.model.config.eos_token_id

        if (kwargs['do_sample']):
            del kwargs['num_beams']

        generation_config = transformers.GenerationConfig(
            use_cache=False,                                 
            **kwargs
        )

        disable_lora = nullcontext()
        if self.lora_name is None and hasattr(self.model, 'disable_adapter'):
            disable_lora = self.model.disable_adapter()

        with torch.no_grad(), disable_lora:
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                generation_config=generation_config
            )[0].to(self.model.device)

        return self.tokenizer.decode(output, skip_special_tokens=True).strip()
    
    def tokenize_sample(self, item, max_seq_length, add_eos_token=True):
        assert self.tokenizer is not None
        result = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

        result = {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < max_seq_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        return result

    def tokenize_training_text(self, training_text, max_seq_length, separator="\n\n\n", **kwargs):
        samples = training_text.split(separator)
        samples = [x.strip() for x in samples]
        def to_dict(text):
            return { 'text': text }

        samples = [to_dict(x) for x in samples]

        training_dataset = datasets.Dataset.from_list(samples)
        training_dataset = training_dataset.shuffle().map(
            lambda x: self.tokenize_sample(x, max_seq_length), 
            batched=False
        )

        return training_dataset

    def train(self, training_text=None, new_peft_model_name=None, **kwargs):
        assert self.should_abort is False
        assert self.model is not None
        assert self.model_name is not None
        assert self.tokenizer is not None

        kwargs = { **TRAINING_PARAMS, **LORA_TRAINING_PARAMS, **kwargs }

        self.lora_name = None
        self.loras = {}

        train_dataset = self.tokenize_training_text(training_text, **kwargs)

        if hasattr(self.model, 'disable_adapter'):
            self.load_model(self.model_name, force=True)
            
        self.model = peft.prepare_model_for_int8_training(self.model)
        self.model = peft.get_peft_model(self.model, peft.LoraConfig(
            r=kwargs['lora_r'],
            lora_alpha=kwargs['lora_alpha'],
            lora_dropout=kwargs['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        ))

        if not os.path.exists('lora'):
            os.makedirs('lora')

        sanitized_model_name = self.model_name.replace('/', '_').replace('.', '_')
        output_dir = f"lora/{sanitized_model_name}_{new_peft_model_name}"

        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=kwargs['micro_batch_size'], 
            gradient_accumulation_steps=kwargs['gradient_accumulation_steps'],
            num_train_epochs=kwargs['epochs'],
            learning_rate=kwargs['learning_rate'],
            fp16=True,  
            optim='adamw_torch',
            logging_steps=20, 
            save_total_limit=3,  
            output_dir=output_dir, 
        )

        # _trainer = self
        # class LoggingCallback(transformers.TrainerCallback):
        #     def on_log(self, args, state, control, logs=None, **kwargs):
        #         _trainer.log += json.dumps(logs) + '\n'

        def should_abort():
            return self.should_abort
        
        def reset_abort():
            self.should_abort = False

        class AbortCallback(transformers.TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if should_abort():
                    print("Stopping training...")
                    control.should_training_stop = True
             

            def on_train_end(self, args, state, control, **kwargs):
                if should_abort():
                    control.should_save = False


        # class CustomTrainer(transformers.Trainer):
        #     def __init__(self, *args, **kwargs):
        #         super().__init__(*args, **kwargs)
        #         self.abort_training = False

        #     def stop_training(self):
        #         print("Stopping training...")
        #         self.abort_training = True

        #     def training_step(self, model, inputs):
        #         if self.abort_training:
        #             raise RuntimeError("Training aborted.")
        #         return super().training_step(model, inputs)

        self.trainer = transformers.Trainer(
            model=self.model, 
            train_dataset=train_dataset, 
            args=training_args, 
            data_collator=transformers.DataCollatorForLanguageModeling( 
                self.tokenizer,  
                mlm=False,
            ),
            callbacks=[AbortCallback()]
        )

        self.model.config.use_cache = False
        result = self.trainer.train(resume_from_checkpoint=False)

        if not should_abort():
            self.model.save_pretrained(output_dir)

        reset_abort()
        return result
    
    def abort_training(self):
        self.should_abort = True

        
if __name__ == '__main__':
    t = Trainer()
    t.load_model(MODEL)

    prompt = "Human: How is cheese made?\n\nAssistant:"
    print(t.generate(prompt))

    t.load_lora('lora/melon-mango-orange')
    print(t.generate(prompt))

    t.unload_lora()
    print(t.generate(prompt))