# Dataset Processing with `DataPreparer` and `DatasetPreparer`

This documentation provides an overview of two Python classes designed for data processing tasks, particularly for machine learning applications. These classes, `DataPreparer` and `DatasetPreparer`, offer flexible ways to prepare datasets from various sources, apply custom formatting functions, and manage dataset splits (training, validation, testing).

## `DataPreparer` Class

The `DataPreparer` class is intended for scenarios where data is initially stored in file formats such as CSV, JSON, or Excel. It automates the process of reading data from files, splitting it into training, validation, and testing sets, applying custom formatting functions, and optionally removing specified columns.

### Key Features:

- **File Reading**: Supports CSV, JSON, and Excel files.
- **Dataset Splitting**: Automatically splits data into training, validation, and testing sets.
- **Custom Formatting**: Applies a user-defined formatting function to dataset entries.
- **Column Management**: Allows specifying essential columns and columns to be removed.

### Usage Example:

```python
from DataPreparer import DataPreparer

def custom_format(data_point):
    return f"Review: {data_point['text']}\nLabel: {data_point['label']}"

data_preparer = DataPreparer(
    filepath='/path/to/dataset.csv',
    formatting_function=custom_format,
    essential_columns=['text', 'label'],
    columns_to_remove=['id']
)

prepared_datasets = data_preparer.prepare_datasets()
data_preparer.show_random_instance(prepared_datasets['train'])
```

## `DatasetPreparer` Class

The `DatasetPreparer` class extends `DataPreparer` to work directly with Hugging Face datasets. It's tailored for situations where data is already loaded into a `Dataset` object, providing functionalities to process the data without reading from files.

### Key Features:

- **Direct Dataset Processing**: Operates on Hugging Face `Dataset` objects.
- **Automatic Splitting**: Splits the dataset into training, validation, and testing sets if not provided.
- **Custom Formatting and Filtering**: Similar to `DataPreparer`, it applies custom formatting functions and manages columns.

### Usage Example:

```python
from datasets import load_dataset
from DatasetPreparer import DatasetPreparer

def format_review(data_point):
    label = "Positive" if data_point['label'] == 1 else "Negative"
    return f"Review: {data_point['text']}\nLabel: {label}"

dataset = load_dataset('imdb', split='train')
processor = DatasetPreparer(
    dataset=dataset,
    formatting_function=format_review,
    essential_columns=['text', 'label'],
    columns_to_remove=[]
)

processed_datasets = processor.prepare_datasets()
processor.show_random_instance(processed_datasets['train'])
```

## Installation Requirements

Before using `DataPreparer` and `DatasetPreparer`, ensure you have the following packages installed:

```sh
pip install pandas scikit-learn datasets
```

## Conclusion

Both `DataPreparer` and `DatasetPreparer` offer robust solutions for preparing datasets for machine learning models. Choose `DataPreparer` when starting with data in file formats and `DatasetPreparer` for pre-loaded Hugging Face datasets. These classes simplify the data preparation process, allowing for custom data formatting and efficient dataset management.
