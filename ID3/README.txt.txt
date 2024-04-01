
# Decision Tree Classifier

This Python script implements a basic decision tree classifier using the ID3 algorithm. It reads data from CSV files, builds a decision tree based on the training data, and then classifies test instances using the constructed tree.

## Overview

The decision tree classifier learns from a given dataset consisting of instances and corresponding target labels. It constructs a tree where each internal node represents an attribute test, and each leaf node represents a class label. The ID3 algorithm is used to recursively build the tree by selecting the best attribute to split the data at each node.

## Requirements

- Python 3.x
- csv

## Usage

1. **Input Data**: Prepare your training data and test data in CSV format. The CSV files should contain the attribute values along with the target labels.

2. **Run the Code**: Execute the Python script (`decision_tree_classifier.py`).

    ```bash
    python decision_tree_classifier.py
    ```

3. **Output**: The script will display the decision tree constructed from the training data and the predicted labels for the test instances.

## File Structure

- `PlayTennis.csv`: CSV file containing the training data.
- `PlayTennisTestData.csv`: CSV file containing the test data.
- `decision_tree_classifier.py`: Python script implementing the decision tree classifier.
- `README.md`: This README file providing an overview of the project.

## Algorithm Explanation

The ID3 algorithm proceeds as follows:

1. **Build Tree**: Recursively select the best attribute to split the data based on the information gain and construct the decision tree.

2. **Print Tree**: Display the decision tree structure with attribute tests at internal nodes and class labels at leaf nodes.

3. **Classify Instances**: Classify test instances by traversing the decision tree based on attribute values.

## Example

For example, given a dataset about playing tennis, the decision tree might include attributes such as weather conditions, temperature, and humidity to predict whether to play tennis or not.

## Author

CodeX

