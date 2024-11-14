import os
import pandas as pd
from scipy.stats import pearsonr
from collections import defaultdict
from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def load_csv(file_path):
    """
    Load CSV file into a Pandas DataFrame.
    """
    data = pd.read_csv(file_path)

    return data


def extract_data(csv_path):
    """
    Extract data from CSV files based on their column names.
    """
    data = load_csv(csv_path)
    if "Variable" in data.columns and "Value" in data.columns:
        categories = (
            data.loc[data["Variable"] == "exp_type", "Value"].astype("category").values
        )
        data.loc[data["Variable"] == "exp_type", "Value"] = pd.Categorical(
            data.loc[data["Variable"] == "exp_type", "Value"], categories=categories
        )  # ordered=True)
        return {"inputs": data.set_index("Variable")["Value"].to_dict()}
    elif "Name" in data.columns and "Value" in data.columns:
        result_data = {"results": data.set_index("Name")["Value"].to_dict()}

        # Include 'Units' if it exists in the DataFrame
        if "Units" in data.columns:
            result_data["units"] = data.set_index("Name")["Units"].to_dict()

        return result_data
    else:
        return {}


def calculate_correlation(data_dict):
    """
    Calculate correlation between input and output variables for all samples.
    """
    correlation_dict = {}

    # Extract input and result variables
    input_variables = list(data_dict.values())[0]["inputs"].keys()
    result_variables = list(data_dict.values())[0]["results"].keys()

    for input_var in input_variables:
        for result_var in result_variables:
            # Skip specified variables
            if input_var in [
                "save_data",
                "file",
                "exp_type",
                "rows_to_skip",
                "nrows",
            ] or result_var in [
                "Date",
                "Time",
                "File Name",
                "Run Time",
                "Experiment Type",
                "Bounds",
            ]:
                continue

            # Extract values for the pair of variables
            input_values_all = []
            result_values_all = []

            for file_data in data_dict.values():
                input_values_all.extend(file_data["inputs"][input_var].values())
                result_values_all.extend(
                    file_data["results"][result_var]["Value"].values
                )

            if any(math.isnan(x) for x in input_values_all) or any(
                math.isnan(x) for x in input_values_all
            ):
                print("list(s) for correlation contain NaN values.")

            # Calculate correlation for all samples
            correlation, _ = pearsonr(input_values_all, result_values_all)

            # Store correlation in the dictionary
            correlation_dict[f"{input_var} - {result_var}"] = correlation

    return correlation_dict


def calculate_correlation(data_dict):
    """
    Calculate correlation between input and output variables for all samples.
    """
    correlation_dict = {}

    # Extract input and result variables from the first file prefix
    first_prefix_data = list(data_dict.values())[0]
    # input_variables = first_prefix_data.keys()
    input_variables = list(first_prefix_data["inputs"].keys())
    # result_variables = first_prefix_data.keys()
    result_variables = list(first_prefix_data["results"].keys())

    for input_var in input_variables:
        for result_var in result_variables:
            # Skip specified variables
            if input_var in [
                "save_data",
                "file",
                "exp_type",
                "rows_to_skip",
                "nrows",
                "user_bounds",
            ] or result_var in [
                "Date",
                "Time",
                "File Name",
                "Run Time",
                "Experiment Type",
                "Bounds",
            ]:
                continue

            # Extract values for the pair of variables
            input_values_all = []
            result_values_all = []

            # try:
            #     for file_data in data_dict.values():
            #         input_values_all.append(float(file_data["inputs"][input_var]))
            #         result_values_all.append(float(file_data["results"][result_var]))
            # except KeyError as e:
            #     print(f"Exception: {e}")
            #     continue

            # for file_data in data_dict.values():
            for file_prefix in data_dict.keys():
                file_data = data_dict[file_prefix]
                # try:
                #     input_values_all.append(float(file_data["inputs"][input_var]))
                #     result_values_all.append(float(file_data["results"][result_var]))
                # except KeyError as e:
                #     print(f"Exception for {file_prefix}: {e}")
                #     continue
                if (
                    input_var in file_data["inputs"].keys()
                    and result_var in file_data["results"].keys()
                ):
                    input_values_all.append(float(file_data["inputs"][input_var]))
                    result_values_all.append(float(file_data["results"][result_var]))

            # Calculate correlation for all samples
            print(
                f"inp:{input_var},len:{len(input_values_all)} to out:{result_var},len:{len(result_values_all)}"
            )
            correlation_coefficient, p_value = pearsonr(
                input_values_all, result_values_all
            )

            # Store correlation in the dictionary
            correlation_dict[f"{input_var} - {result_var}"] = correlation_coefficient

    return correlation_dict


def process_folder(folder_path):
    """
    Process pairs of input and result files with the same prefix in a folder.
    """
    file_prefixes = set()

    # Find all files in the folder
    all_files = os.listdir(folder_path)

    # Extract prefixes from input and result files
    for file in all_files:
        if file.endswith("inputs.csv"):
            file_prefix = os.path.splitext(file)[0][: -len("inputs")]
            file_prefixes.add(file_prefix)

    # Process each file prefix
    data_dict = defaultdict(lambda: {})
    correlation_dict = {}
    for file_prefix in file_prefixes:
        input_file = f"{file_prefix}inputs.csv"
        result_file = f"{file_prefix}results.csv"

        input_path = os.path.join(folder_path, input_file)
        result_path = os.path.join(folder_path, result_file)

        if os.path.exists(input_path) and os.path.exists(result_path):
            input_data = extract_data(input_path)
            result_data = extract_data(result_path)

            data_dict[file_prefix]["inputs"] = input_data["inputs"]
            data_dict[file_prefix]["results"] = result_data["results"]

    # Calculate correlation for all pairs of variables across all samples
    # data_dict = dict(islice(data_dict.items(), 2))
    correlation_dict = calculate_correlation(data_dict)

    return data_dict, correlation_dict


def create_correlation_heatmap(correlation_dict, output_file):
    """
    Create a heatmap of correlation results and save it as a PNG image.
    """
    correlation_matrix = pd.DataFrame(
        list(correlation_dict.items()), columns=["Variable Pair", "Correlation"]
    )
    correlation_matrix.set_index("Variable Pair", inplace=True)
    # correlation_matrix = pd.DataFrame(correlation_dict).T
    plt.figure(figsize=(100, 50))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(output_file)
    plt.show()


if __name__ == "__main__":
    # folder_path = "/srv/hemi01-j01/dmref/laser_shock_lab/data/pdv_data"
    parser = argparse.ArgumentParser(
        description="Process a folder and create a correlation heatmap."
    )

    # Add the folder_path argument
    parser.add_argument("folder_path", help="Path to the pdv data to be processed.")

    args = parser.parse_args()

    # Call the function with the provided folder_path
    data_dict, correlation_dict = process_folder(args.folder_path)

    create_correlation_heatmap(correlation_dict, "correlation_heatmap.png")
