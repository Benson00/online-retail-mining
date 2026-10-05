# Online Retail Mining

This repository contains the implementation and experimental results for a **frequent itemset mining** study on the Online Retail dataset. The project focuses on evaluating the performance and scalability of three classical frequent itemset mining algorithms:

* **Apriori**
* **FP-Growth**
* **SON**

The repository also includes a **Quantity-Aware (QA) analysis**, which extends the evaluation by taking the purchased quantities into account.

## Repository Structure

```text
online-retail-mining/
│
├── plots/
│   ├── ...
│   └── ...
│
├── qa_results/
│   ├── ...
│   └── ...
│
├── frequent_itemsets.py
├── plot_results.py
├── plot_results_2.py
├── qa.py
└── .gitignore
```

### `frequent_itemsets.py`

Main script for the frequent itemset mining experiments.

It contains the implementation and execution logic for the three algorithms considered in the project:

* **Apriori**
* **FP-Growth**
* **SON**

The script is used to run the experiments across the different configurations considered in the scalability analysis.

### `plot_results.py` and `plot_results_2.py`

Scripts used to generate the plots for the experimental results.

The generated figures are stored in the [`plots`](https://github.com/Benson00/online-retail-mining/tree/main/plots) directory.

### `qa.py`

Implementation of the **Quantity-Aware Analysis**.

Unlike standard frequent itemset mining, where transactions are generally treated as sets of items, the quantity-aware analysis also considers the number of units purchased for each item.

The results and plots produced by this analysis are stored in [`qa_results`](https://github.com/Benson00/online-retail-mining/tree/main/qa_results).

### `plots/`

Contains the plots and results related to the **scalability analysis**.

The plots are generated using `plot_results.py` and `plot_results_2.py`.

The analysis compares the behavior of Apriori, FP-Growth, and SON under different experimental configurations, with particular attention to their computational performance and scalability.

### `qa_results/`

Contains the plots and experimental results produced by the **Quantity-Aware Analysis** implemented in `qa.py`.

---


## Results

### Scalability Results

The [`plots`](https://github.com/Benson00/online-retail-mining/tree/main/plots) directory contains the results of the scalability experiments, allowing the performance of Apriori, FP-Growth, and SON to be compared across the tested configurations.

### Quantity-Aware Results

The [`qa_results`](https://github.com/Benson00/online-retail-mining/tree/main/qa_results) directory contains the results and plots of the quantity-aware experiments.

---

## Summary

The project provides an experimental comparison of three frequent itemset mining algorithms and investigates their behavior from two complementary perspectives:

1. **Scalability analysis**, comparing the computational behavior of Apriori, FP-Growth, and SON.
2. **Quantity-Aware analysis**, investigating how incorporating purchase quantities affects the mining process and results.

All source code, plots, and experimental results are included in the repository.
