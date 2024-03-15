## README for GitHub

### FineTuner: A Class for Model Fine-Tuning and Quantization

The `FineTuner` class facilitates the fine-tuning of Transformer models, with special features for quantization and low-rank adaptation via the PEFT (Parameter Efficient Fine-Tuning) library. It integrates seamlessly with PyTorch and the Hugging Face Transformers library, offering a streamlined approach to enhance pre-trained models with your data.

### Features

- **Model Fine-Tuning**: Easily fine-tune pre-trained models on your dataset.
- **Quantization Support**: Utilize BitsAndBytes for model quantization to reduce model size and potentially increase inference speed, with optional usage based on your needs.
- **Low-Rank Adaptation**: Configure LoRA parameters directly to introduce parameter-efficient updates to the model.
- **Multi-GPU Training**: Automatically leverage multiple GPUs if available to accelerate training.
- **Customizable Training Parameters**: Tailor training arguments to optimize model performance.

### Installation

Ensure you have PyTorch, Transformers, PEFT, and TRL libraries installed. You can install these dependencies via pip:

```bash
pip install -U datasets
pip3 install -q -U bitsandbytes==0.42.0
pip3 install -q -U peft==0.8.2
pip3 install -q -U trl==0.7.10!pip3 install -q -U accelerate==0.27.1
pip3 install -q -U transformers==4.38.1
```

### Usage

#### Initialization

```python
from datasets import Dataset
from your_module import FineTuner  # Ensure FineTuner is accessible

# Define your training and validation data
data = {
    "train": train_dataset,  # Instance of `Dataset`
    "validation": val_dataset  # Instance of `Dataset`
}

# Initialize the FineTuner
fine_tuner = FineTuner(
    model_id="gpt2",
    tokenizer_id="gpt2",
    data=data,
    output_dir="./fine_tuned_model",
    bnb_config_use=True
)
```

#### Configuration

```python
# Configure model adaptation and quantization settings
fine_tuner.configure_lora(r=4)
fine_tuner.configure_bnb(load_in_4bit=True)

# Optional: Customize training arguments
fine_tuner.configure_training_args(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5
)
```

#### Training

```python
# Load the model and tokenizer
fine_tuner.load_model_and_tokenizer()

# Start the fine-tuning process
fine_tuner.fine_tune()
```

#### Text Generation

```python
# Generate text with the fine-tuned model
print(fine_tuner.generate_text("The quick brown fox", max_new_tokens=50))
```

### Methods Overview

- `configure_lora(**kwargs)`: Set parameters for Low-Rank Adaptation.
- `configure_bnb(**bnb_config_kwargs)`: Enable and configure model quantization.
- `load_model_and_tokenizer()`: Load the specified tokenizer and model, preparing them for training or inference.
- `print_trainable_parameters()`: Display the count of trainable parameters.
- `train_model(dataset_text_field="Text", max_seq_length=1024)`: Commence the training process with the provided data.
- `generate_text(text, max_new_tokens=50)`: Generate text based on a given prompt.
- `prepare_for_kbit_training()`: Prepare the model for k-bit quantization training.
- `fine_tune()`: Execute the fine-tuning process, wrapping together loading, preparation, and training steps.

### Contributions

Contributions are welcome! Please submit pull requests or open issues to discuss potential improvements or report bugs.

### License

Apache 2.0.