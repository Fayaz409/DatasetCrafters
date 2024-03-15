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
pip3 install -q -U trl==0.7.10
pip3 install -q -U accelerate==0.27.1
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

### Using the `generate_text` Method for Inference Before and After Fine-Tuning

The `FineTuner` class includes a `generate_text` method designed for generating text based on a given prompt. This method can be particularly useful for evaluating the performance of your model both before and after the fine-tuning process. It allows you to directly observe the impact of your fine-tuning efforts on the model's text generation capabilities.

#### Method Overview

```python
def generate_text(self, text, max_new_tokens=50):
    inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
    return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
```

- **Input**: The method takes a string of `text` as input, which serves as the prompt for text generation, and an optional `max_new_tokens` parameter that controls the maximum number of tokens to generate.
- **Processing**: It tokenizes the input text, converts it into a format suitable for the model, generates output tokens in response to the input, and then decodes these tokens back into human-readable text.
- **Output**: The method returns the generated text as a string.

#### Inference Before Fine-Tuning

Before you fine-tune your model on a specific dataset, it's insightful to run inference using the pre-trained version of the model. This gives you a baseline understanding of the model's capabilities and helps set expectations for what fine-tuning might achieve:

```python
# Initialize the FineTuner with your chosen pre-trained model
fine_tuner = FineTuner(
    model_id="gpt2",
    tokenizer_id="gpt2",
    data=None,  # Data is not required for inference
    output_dir="./model_output",
    bnb_config_use=False  # Quantization is optional for initial inference
)

# Load the pre-trained model and tokenizer
fine_tuner.load_model_and_tokenizer()

# Generate text using the pre-trained model
print(fine_tuner.generate_text("The quick brown fox", max_new_tokens=50))
```

#### Inference After Fine-Tuning

After fine-tuning your model, use the same `generate_text` method to evaluate how the model's text generation has changed. This comparison allows you to directly assess the impact of fine-tuning on the model's performance:

```python
# Assuming fine-tuning has been completed, and the FineTuner instance is set up

# Generate text using the fine-tuned model
print(fine_tuner.generate_text("The quick brown fox", max_new_tokens=50))
```

#### Benefits of Pre- and Post-Fine-Tuning Inference

- **Baseline Comparison**: Generates a clear before-and-after picture of the model's performance, highlighting the benefits of fine-tuning.
- **Qualitative Evaluation**: Provides an immediate, qualitative sense of improvements in the model's text generation abilities, complementing quantitative metrics like loss and accuracy.
- **Guidance for Further Training**: Helps in deciding whether further fine-tuning or adjustments to the training process are necessary based on the observed output.
  

### Conclusion

The `generate_text` method of the `FineTuner` class is a powerful tool for evaluating model performance through text generation. By utilizing it both before and after fine-tuning, you can gain valuable insights into how your training efforts have enhanced the model's capabilities, guiding you towards achieving your desired outcomes.
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
