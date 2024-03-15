import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, SFTTrainer, prepare_model_for_kbit_training
from datasets import Dataset
class FineTuner():
    """
    A class designed for fine-tuning language models using the transformers and PEFT libraries with 
    support for BitsAndBytes quantization and LoRA configuration.

    Attributes:
        model (AutoModelForCausalLM): The model to be fine-tuned.

        device (torch.device): Device on which the model will be trained (e.g., CPU or CUDA).

        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.

        model_id (str): Identifier for the model to be loaded from Hugging Face's Model Hub.

        tokenizer_id (str): Identifier for the tokenizer to be loaded.

        data (dict): A dictionary containing training and validation datasets.

        output_dir (str): Directory where training outputs will be saved.

        default_lora_config (dict): Default LoRA configuration settings.

        default_bnb_config (dict): Default BitsAndBytes quantization settings.

        default_training_args (dict): Default training arguments.

    Methods:
        configure_lora(**kwargs): Configures LoRA parameters for model fine-tuning.

        configure_bnb(**bnb_config_kwargs): Configures BitsAndBytes quantization settings.

        load_model_and_tokenizer(): Loads the specified model and tokenizer.

        print_trainable_parameters(): Prints the number of trainable parameters in the model.

        train_model(dataset_text_field="Text", max_seq_length=1024): Trains the model using the provided datasets and configurations.

        generate_text(text, max_new_tokens=50): Generates text based on a prompt.

        prepare_for_kbit_training(): Prepares the model for training with k-bit quantization.
        
        fine_tune(): Executes the fine-tuning process.

    Example usage:
        >>> data = {"train": train_dataset, "validation": val_dataset}
        >>> fine_tuner = FineTuner("gpt2", "gpt2", data, "./output")
        >>> fine_tuner.configure_bnb(load_in_4bit=True)
        >>> fine_tuner.configure_lora(r=4)
        >>> fine_tuner.fine_tune()
    """
    def __init__(self, model_id:str, tokenizer_id:str, data:Dataset, output_dir:str,training_args=None, device=None,bnb_config_use=False):
        self.model = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.data = data
        self.training_args=training_args
        self.output_dir = output_dir
        self.bnb_config_use=bnb_config_use
        # Set default configurations
        self.default_lora_config = {}
        self.default_bnb_config = {}
        self.default_training_args = {}
    
    def get_underlying_model(self):
        # If the model is wrapped by DataParallel, access the underlying model
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        else:
            return self.model

        
    def configure_lora(self, **kwargs):
        default_lora_args = {
            "r": 8,
            "target_modules": ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            "task_type": "CAUSAL_LM",
        }
        # Merge and override defaults with user-provided arguments
        self.default_lora_config = {**default_lora_args, **kwargs}
    def configure_training_args(self, **custom_args):
        # Specified default training arguments
        default_args = {
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "optim": "paged_adamw_32bit",
            "logging_steps": 1,
            "learning_rate": 1e-4,
            "fp16": True,
            "max_grad_norm": 0.3,
            "num_train_epochs": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 0.2,
            "warmup_ratio": 0.05,
            "save_strategy": "epoch",
            "group_by_length": True,
            "output_dir": self.output_dir,
            "report_to": "tensorboard",
            "save_safetensors": True,
            "lr_scheduler_type": "cosine",
            "seed": 42,
        }
        # Merge and override defaults with user-provided arguments
        self.default_training_args = {**default_args, **custom_args}

    def configure_bnb(self, **bnb_config_kwargs):
        default_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16
        }
        # Merge and override defaults with user-provided configuration
        self.default_bnb_config = {**default_config, **bnb_config_kwargs}

    def load_model_and_tokenizer(self):
        try: 
            if self.bnb_config_use:
                bnb_config = BitsAndBytesConfig(**self.default_bnb_config)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, 
                    quantization_config=bnb_config
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id)

            # If more than one GPU is available, wrap the model with DataParallel
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                self.model = DataParallel(self.model)

            self.model.to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to load model and tokenizer. Error: {e}")

    
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}"
        )
    
    def train_model(self, dataset_text_field="Text", max_seq_length=1024):
        underlying_model = self.get_underlying_model()
        # Ensure custom training arguments are set
        if not self.training_args:
            self.training_args = TrainingArguments(**self.default_training_args)
        
        lora_config = LoraConfig(**self.default_lora_config)
        
        trainer = SFTTrainer(
            model=underlying_model,
            train_dataset=self.data["train"],
            eval_dataset=self.data['validation'],
            args=self.training_args,  # Use the stored training arguments
            peft_config=lora_config,
            dataset_text_field=dataset_text_field,
            max_seq_length=max_seq_length,
        )
        trainer.train()

    
    def generate_text(self, text,max_new_tokens=50):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def prepare_for_kbit_training(self):
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
    
    def fine_tune(self):
        self.load_model_and_tokenizer()
        self.print_trainable_parameters()
        self.prepare_for_kbit_training()
        self.train_model()
    