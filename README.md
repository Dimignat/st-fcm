# A Self-Training algorithm based on Fuzzy C-Means clustering

## Usage
1. Create a dataloader for your dataset (inherit from `AbstractDataLoader`)
2. Pick a classifier model, optionally a kernel approximation algorithm
3. Initialize the `SemiSupervisedModel` class instance
4. Increase the labeled part of the dataset by calling the `increase_labeled` method
5. Fit (`fit`) and predict (`predict`)
