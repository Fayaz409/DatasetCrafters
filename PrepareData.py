import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from typing import Callable, Dict
import random
import os

class DataPreparer():
    """
    Prepares datasets for machine learning tasks by reading from various file types, 
    splitting into training, validation, and test sets, and applying custom formatting functions.

    Attributes:
        filepath (str): Path to the input file containing the dataset.
        formatting_function (callable): A function that takes data points as input and returns a formatted string.
        essential_columns (list, optional): List of column names that are essential for the formatting function. 
            If not provided, all columns are considered essential.
        columns_to_remove (list, optional): List of column names to be removed from the dataset before processing.

    Methods:
        read_data: Reads the data from the filepath provided during initialization.
        split_data: Splits the dataframe into training, validation, and test sets.
        convert_to_dataset: Converts the split dataframes into `Dataset` objects from the `datasets` library.
        generate_instruction_dataset: Applies the formatting function to data points, considering only essential columns.
        process_dataset: Processes the dataset, removing specified columns and applying the formatting function.
        prepare_datasets: Orchestrates reading, splitting, and processing of data to prepare the dataset.
        show_random_instance: Displays a random instance from a given dataset.

    Example:
        >>> from your_project.data_preparer import DataPreparer
        >>> def data_format(question, context, answer):
        ...     return f"Context: {context}\\n Question: {question}\\n Answer: {answer}"
        >>> data_preparer = DataPreparer(
        ...     '/path/to/dataset.json',
        ...     data_format,
        ...     essential_columns=['context', 'question', 'answer'],
        ...     columns_to_remove=['id', 'metadata']
        ... )
        >>> dataset = data_preparer.prepare_datasets()
        >>> data_preparer.show_random_instance(dataset['train'])
        
        Make sure to have pandas and datasets installed:
        >>> pip install pandas datasets
        >>> pip install scikit-learn
        Also make sure to import these:
        >>> import pandas as pd
        >>> from sklearn.model_selection import train_test_split
        >>> from datasets import Dataset
        >>> from typing import Callable, Dict
        >>> import random
        >>> import os

        Supported file types are .csv, .json, and Excel files (.xls, .xlsx). The class automatically
        determines the file type based on the file extension and reads the data accordingly.
        
        Note: The `formatting_function` must be capable of accepting keyword arguments that match the
        names of the `essential_columns`. Any additional columns not listed as essential or marked for removal
        will not be included in the formatted output.
    """
    def __init__(self, filepath: str, formatting_function: Callable[[Dict[str,str]], str],essential_columns:list[str]=None,columns_to_remove:list[str]=None):
        self.filepath = filepath
        self.formatting_function=formatting_function
        self.essential_columns = essential_columns if essential_columns is not None else []
        self.columns_to_remove = columns_to_remove if columns_to_remove is not None else []
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def read_data(self):
        _, file_extension = os.path.splitext(self.filepath)
        try:
            if file_extension == '.csv':
                df = pd.read_csv(self.filepath)
            elif file_extension == '.json':
                df = pd.read_json(self.filepath)
            elif file_extension in ['.xls', '.xlsx']:
                df = pd.read_excel(self.filepath)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # If essential_columns not provided, treat all columns as essential
            if not self.essential_columns:
                # Automatically set essential_columns to all columns minus columns_to_remove
                self.essential_columns = [col for col in df.columns if col not in self.columns_to_remove]
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.filepath}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"File appears to be empty: {self.filepath}")
        except (PermissionError, OSError) as e:
            raise IOError(f"Error reading file: {e}")

    def split_data(self, df):
        # Splitting data into train, test, and validation sets
        train, test = train_test_split(df, test_size=0.2)
        val, test_data = train_test_split(test, test_size=0.15)
        return train, val, test_data

    def convert_to_dataset(self, train, val, test_data):
        # Convert pandas DataFrames to Dataset objects
        self.train_dataset = Dataset.from_pandas(train)
        self.val_dataset = Dataset.from_pandas(val)
        self.test_dataset = Dataset.from_pandas(test_data)

    def generate_instruction_dataset(self, data_point, formatting_function):
        filtered_data_point = {key: value for key, value in data_point.items()
                               if key in self.essential_columns or key not in self.columns_to_remove}

        # Check if essential columns are all present
        if all(key in filtered_data_point for key in self.essential_columns):
            # Call the formatting function with the filtered data_point as arguments
            formatted_text = formatting_function(**filtered_data_point)
        else:
            # Handle the case where required keys are missing or return a placeholder
            formatted_text = "Missing essential data for formatting."

        return {'text': formatted_text}


    def process_dataset(self, data):
        # Print all columns in the dataset before processing
        print("Columns in the dataset:", ', '.join(data.column_names))

        # Ensure columns_to_remove does not include essential columns
        columns_to_remove = set(self.columns_to_remove) - set(self.essential_columns)

        # Provide feedback on the final columns to be removed and preserved
        print("Essential columns to be preserved:", ', '.join(self.essential_columns))
        print("Columns to be removed:", ', '.join(columns_to_remove))

        # Identify valid columns to remove (present in the dataset)
        valid_columns_to_remove = [col for col in columns_to_remove if col in data.column_names]

        # Identify invalid column removal attempts
        invalid_columns = columns_to_remove - set(valid_columns_to_remove)
        if invalid_columns:
            # Notify about non-existent columns attempted to be removed
            print(f"Warning: Attempted to remove non-existent or essential columns: {', '.join(invalid_columns)}")

        # Proceed to remove the valid specified columns and process the dataset
        processed_data = (
            data.shuffle(seed=42)
            .map(lambda data_point: self.generate_instruction_dataset(data_point, self.formatting_function))
            .remove_columns(valid_columns_to_remove)  # Use only validated columns
        )
        return processed_data


    def prepare_datasets(self):
        # High-level function to prepare all datasets
        df = self.read_data()
        train, val, test_data = self.split_data(df)
        self.convert_to_dataset(train, val, test_data)

        dataset = {}
        dataset['train'] = self.process_dataset(self.train_dataset)
        dataset['validation'] = self.process_dataset(self.val_dataset)
        dataset['test'] = self.process_dataset(self.test_dataset)
        # Optionally process test_dataset if needed

        return dataset

    def show_random_instance(self, dataset):
        # Function to show a random instance from the dataset
        random_index = random.randint(0, len(dataset) - 1)
        print(dataset[random_index]['text'])
    

class DatasetPreparer:
    """
    Processes Hugging Face datasets by applying custom formatting functions
    and optionally filtering columns. Now also handles splitting into training,
    validation, and test datasets.

    Attributes:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset (optional).
        test_dataset (Dataset): Test dataset (optional).
        formatting_function (Callable): Function to format data points.
        essential_columns (List[str], optional): Essential columns for the formatting function.
        columns_to_remove (List[str], optional): Columns to be removed before processing.

    Example usage:
        >>> from datasets import load_dataset
        >>> dataset = load_dataset('imdb', split='train')
        >>> def format_review(data_point):
        ...     label = "Positive" if data_point['label'] == 1 else "Negative"
        ...     return f"Review: {data_point['text']}\nLabel: {label}"
        >>> processor = DatasetPreparer(
        ...     train_dataset=dataset,
        ...     formatting_function=format_review,
        ...     essential_columns=['text', 'label'],
        ...     columns_to_remove=[]
        ... )
        >>> processed_datasets = processor.prepare_datasets()
        >>> processor.show_random_instance(processed_datasets['train'])
    """

    def __init__(self, train_dataset: Dataset, formatting_function: Callable[[Dict[str, str]], str], essential_columns: List[str] = None, columns_to_remove: List[str] = None, val_dataset: Dataset = None, test_dataset: Dataset = None, split_ratios: Tuple[float, float] = (0.1, 0.1)):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.formatting_function = formatting_function
        self.essential_columns = essential_columns if essential_columns is not None else []
        self.columns_to_remove = columns_to_remove if columns_to_remove is not None else []
        self.split_ratios = split_ratios

        if not self.val_dataset or not self.test_dataset:
            self._split_dataset()

    def _split_dataset(self):
        """
        Splits the train_dataset into validation and test datasets if they are not provided.
        """
        val_split_ratio, test_split_ratio = self.split_ratios
        # Calculate split sizes
        val_size = int(len(self.train_dataset) * val_split_ratio)
        test_size = int(len(self.train_dataset) * test_split_ratio)

        # Split the dataset
        train_test_split = self.train_dataset.train_test_split(test_size=test_size + val_size)
        temp_train_dataset = train_test_split['train']
        test_val_split = train_test_split['test'].train_test_split(test_size=test_size)

        # Assign splits
        self.train_dataset = temp_train_dataset
        if not self.val_dataset:
            self.val_dataset = test_val_split['train']
        if not self.test_dataset:
            self.test_dataset = test_val_split['test']

    def process_dataset(self, dataset):
        """
        Processes the provided dataset by applying the formatting function.
        """
        if not self.essential_columns:
            # Automatically set essential_columns based on dataset column names
            self.essential_columns = [col for col in dataset.column_names if col not in self.columns_to_remove]
        
        processed_data = dataset.map(lambda data_point: self.generate_instruction_dataset(data_point, self.formatting_function))
        return processed_data

    def generate_instruction_dataset(self, data_point, formatting_function):
        """
        Applies the formatting function to a data point, considering only essential columns.
        """
        filtered_data_point = {key: value for key, value in data_point.items() if key in self.essential_columns}
        formatted_text = formatting_function(**filtered_data_point)
        return {'Text': formatted_text}

    def prepare_datasets(self):
        """
        Prepares the training, validation, and test datasets.
        """
        processed_train = self.process_dataset(self.train_dataset)
        
        processed_val = self.process_dataset(self.val_dataset) if self.val_dataset else None
        processed_test = self.process_dataset(self.test_dataset) if self.test_dataset else None

        return {'train': processed_train, 'validation': processed_val, 'test': processed_test}

    def show_random_instance(self, dataset):
        """
        Displays a random instance from the specified dataset.
        """
        if dataset:
            random_index = random.randint(0, len(dataset) - 1)
            print(dataset[random_index]['Text'])
        else:
            print("Dataset is None.")