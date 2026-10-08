import argparse
import csv
import json
import math
import os
import time
import itertools
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F



def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantity-aware Frequent Itemset Mining experiments using PySpark."
    )

    parser.add_argument(
        "--cores",
        type=int,
        default=20,
        help="Number of Spark execution threads. Default: 20."
    )

    parser.add_argument(
        "--partitions",
        type=int,
        default=40,
        help="Number of Spark partitions. Default: 40."
    )

    parser.add_argument(
        "--supports",
        type=float,
        nargs="+",
        default=[0.03, 0.025, 0.02],
        help="Minimum support ratios. Default: 0.03 0.025 0.02."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to the Online Retail dataset (.xlsx or .csv). "
             "If omitted, the dataset is downloaded through ucimlrepo."
    )

    parser.add_argument(
        "--dataset-id",
        type=int,
        default=352,
        help="UCI dataset ID. Default: 352."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="qa_results",
        help="Directory where results and plots are stored. Default: qa_results."
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top frequent itemsets to save. Default: 10."
    )

    return parser.parse_args()


def create_spark_session(cores):
    master = f"local[{cores}]"

    spark = (
        SparkSession.builder
        .appName("QuantityAwareMining")
        .master(master)
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "40")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 70)
    print("Spark configuration")
    print("=" * 70)
    print(f"Master: {master}")
    print(f"Execution threads: {cores}")
    print("=" * 70)

    return spark


def load_dataset(input_path=None, dataset_id=352):
    """
    Load the Online Retail dataset from a local file or fetch it from UCI ML Repository.
    """
    # Load dataset from a local file if a path is provided
    if input_path is not None:
        print(f"Loading dataset from: {input_path}")

        if input_path.lower().endswith(".csv"):
            pdf = pd.read_csv(input_path)
        elif input_path.lower().endswith((".xlsx", ".xls")):
            pdf = pd.read_excel(input_path)
        else:
            raise ValueError(
                "Unsupported input format. Use CSV or Excel."
            )

    else:
        # load from UCI ML Repository using ucimlrepo
        print(f"Fetching UCI dataset {dataset_id}...")
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise RuntimeError(
                "The 'ucimlrepo' package is not installed. "
                "Install it with: pip install ucimlrepo"
            ) from exc

        online_retail = fetch_ucirepo(id=dataset_id)
        pdf = online_retail.data.original.copy()

    required_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
    ]

    missing = [c for c in required_columns if c not in pdf.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return pdf


def prepare_quantity_aware_baskets(spark, pdf):
    """Prepare transactions while keeping item quantities."""

    print("Preparing quantity-aware transaction baskets...")

    for column in ["InvoiceNo", "StockCode", "Description", "CustomerID"]:
        if column in pdf.columns:
            pdf[column] = pdf[column].astype(str)

    # Convert the Pandas DataFrame to a Spark DataFrame
    df = spark.createDataFrame(pdf)

    total_raw_records = df.count()

    # Clean the dataset by filtering out invalid records
    df_cleaned = (
        df.filter(~F.col("InvoiceNo").rlike("^[Cc]"))
          .filter(F.col("Quantity") > 0)
          .filter(F.col("Description") != "nan")
    )

    valid_records = df_cleaned.count()

    # Build a mapping from stock codes to item descriptions
    item_dict_rows = (
        df_cleaned
        .select("StockCode", "Description")
        .dropDuplicates(["StockCode"])
        .collect()
    )

    item_dict = {
        row["StockCode"]: row["Description"]
        for row in item_dict_rows
    }

    # Sum quantities for each item within each invoice
    df_grouped = (
        df_cleaned
        .groupBy("InvoiceNo", "StockCode")
        .agg(F.sum("Quantity").alias("Quantity"))
    )

    # Create baskets containing items and their quantities
    baskets_df = (
        df_grouped
        .groupBy("InvoiceNo")
        .agg(
            F.collect_list("StockCode").alias("items"),
            F.collect_list("Quantity").alias("quantities")
        )
        .filter(F.size(F.col("items")) > 1) # Keep only baskets containing more than one item
    )

    baskets_df.cache()

    total_baskets = baskets_df.count()

    print(f"Raw records: {total_raw_records}")
    print(f"Valid records after cleaning: {valid_records}")
    print(f"Valid multi-item baskets: {total_baskets}")

    local_baskets = []

    # Convert Spark baskets into local dictionaries
    for row in baskets_df.collect():
        basket = {
            item: float(quantity)
            for item, quantity
            in zip(row["items"], row["quantities"])
        }

        local_baskets.append(basket)

    baskets_df.unpersist()

    return local_baskets, item_dict




# QA - APRIORI

def generate_candidates(prev_frequent_itemsets, k):
    candidates = set()
    prev_itemsets = list(prev_frequent_itemsets.keys())

    for i in range(len(prev_itemsets)):
        for j in range(i + 1, len(prev_itemsets)):
            union_set = prev_itemsets[i] | prev_itemsets[j]

            if len(union_set) != k:
                continue

            valid = True

            for subset in itertools.combinations(union_set, k - 1):
                if frozenset(subset) not in prev_frequent_itemsets:
                    valid = False
                    break

            if valid:
                candidates.add(union_set)

    return candidates


def distributed_apriori_qa(spark, baskets_rdd, total_baskets, min_support_ratio):
    start_time = time.time()

    min_support_count = int(total_baskets * min_support_ratio)

    all_frequent_itemsets = {}
    level_stats = {}

    print("\n" + "=" * 70)
    print(
        f"Quantity-Aware A-Priori | "
        f"Support = {min_support_ratio * 100:.2f}%"
    )
    print("=" * 70)

    level_start = time.time()

    
    def map_l1(basket):
        """
        For each item in the basket, creates a 1-item candidate with its transaction count, 
        log-scaled quantity, and original quantity, which are used to calculate its weighted support.
        """
        # Create a weighted 1-item candidate for each item in the basket
        for item, quantity in basket.items():
            yield (
                frozenset([item]),  # Represent the item as a 1-item set
                (
                    1,                  # Number of transactions containing the item
                    math.log1p(quantity), # Log-scaled quantity
                    quantity             # Original quantity
                )
            )





    l1_rdd = (
        baskets_rdd
        .flatMap(map_l1) # Generate one 1-item candidate for each item in every basket
        .reduceByKey(
            lambda a, b: (
                a[0] + b[0], # Sum the number of transactions
                a[1] + b[1], # Sum the log-scaled quantities
                max(a[2], b[2]) # Keep the maximum original quantity
            )
        ) # Combine candidates with the same item across all baskets
    )

    total_unique_items = l1_rdd.count()

    # filter out candidates that do not meet the minimum support count
    # and collect the remaining frequent 1-item candidates into a dictionary
    current_frequent = dict(
        l1_rdd
        .filter(
            lambda x: x[1][0] >= min_support_count
        )
        .collect()
    )

    all_frequent_itemsets.update(current_frequent)

    level_stats[1] = {
        "candidates": total_unique_items,
        "frequent": len(current_frequent),
        "time": time.time() - level_start
    }

    # Unpersist the RDD to free up memory
    l1_rdd.unpersist()

    k = 2

    while current_frequent:
        level_start = time.time()

        # generate the candidates
        candidates = generate_candidates(
            current_frequent,
            k
        )

        if not candidates:
            break

        # broadcast the candidates to all nodes
        broadcast_candidates = (
            spark.sparkContext.broadcast(candidates)
        )

        def map_candidates(basket):
            """ 
            Map a basket to its frequent itemsets. 
            """
            basket_items = set(basket.keys())

            for candidate in broadcast_candidates.value:
                if candidate.issubset(basket_items):
                    # select the minimum quantity among the items in the candidate
                    # The minimum quantity is used because the joint purchase volume of an itemset is
                    # limited by the item purchased in the smallest quantity
                    volume = min(
                        basket[item]
                        for item in candidate
                    )

                    yield (
                        candidate,
                        (
                            1,
                            math.log1p(volume),
                            volume
                        )
                    )

        # Count the support for each candidate across all baskets
        counts_rdd = (
            baskets_rdd
            .flatMap(map_candidates) # map a basket to its frequent itemsets
            .reduceByKey(
                lambda a, b: (
                    a[0] + b[0],
                    a[1] + b[1],
                    max(a[2], b[2])
                )
            )  # Combine candidates with the same itemset across all baskets
            .filter(
                lambda x: x[1][0] >= min_support_count
            ) # filter out candidates that do not meet the minimum support count
        )

        current_frequent = dict(
            counts_rdd.collect()
        )

        all_frequent_itemsets.update(
            current_frequent
        )

        level_stats[k] = {
            "candidates": len(candidates),
            "frequent": len(current_frequent),
            "time": time.time() - level_start
        }

        counts_rdd.unpersist()
        broadcast_candidates.unpersist()

        k += 1

    total_time = time.time() - start_time

    statistics = {
        "total_time": total_time,
        "total_itemsets": len(all_frequent_itemsets),
        "levels": level_stats
    }

    return all_frequent_itemsets, statistics



# QA - SON

def local_apriori(baskets_partition, local_min_support):
    """Used in the SON algorithm to find local frequent itemsets within a partition."""
    item_counts = defaultdict(int)

    for basket in baskets_partition:
        for item in basket.keys():
            item_counts[
                frozenset([item])
            ] += 1

    current_frequent = {
        itemset: count
        for itemset, count in item_counts.items()
        if count >= local_min_support
    }

    all_local_frequent = list(
        current_frequent.keys()
    )

    k = 2

    while current_frequent:
        candidates = set()
        previous = list(
            current_frequent.keys()
        )

        for i in range(len(previous)):
            for j in range(i + 1, len(previous)):
                union_set = (
                    previous[i] | previous[j]
                )

                # consider only candidates of size k
                if len(union_set) != k:
                    continue

                valid = True
                # check if all (k-1)-subsets of the candidate are frequent
                for subset in itertools.combinations(
                    union_set,
                    k - 1
                ):
                    if frozenset(subset) not in current_frequent:
                        valid = False
                        break # if not all subsets are frequent, skip this candidate

                if valid: # if all (k-1)-subsets are frequent, add the candidate to the set
                    candidates.add(union_set)

        if not candidates:
            break

        candidate_counts = defaultdict(int)

        for basket in baskets_partition:
            basket_items = set(
                basket.keys()
            )

            for candidate in candidates:
                if candidate.issubset(
                    basket_items
                ):
                    candidate_counts[candidate] += 1

        current_frequent = {
            candidate: count
            for candidate, count
            in candidate_counts.items()
            if count >= local_min_support
        }   # filter out candidates that do not meet the local minimum support

        all_local_frequent.extend(
            current_frequent.keys()
        )

        k += 1

    return all_local_frequent


def quantity_aware_son(
    spark,
    baskets_rdd,
    total_baskets,
    min_support_ratio
):
    start_time = time.time()

    global_min_support = (
        total_baskets * min_support_ratio
    )

    print("\n" + "=" * 70)
    print(
        f"Quantity-Aware SON | "
        f"Support = {min_support_ratio * 100:.2f}%"
    )
    print("=" * 70)

    phase1_start = time.time()

    def phase1_mapper(partition_iterator):
        """
        Map each partition to its local frequent itemsets using the local Apriori algorithm.
        """
        partition = list(partition_iterator)

        if not partition:
            return iter([])

        local_threshold = (
            len(partition)
            * min_support_ratio
        )

        candidates = local_apriori(
            partition,
            local_threshold
        )

        return iter(candidates)

    candidate_rdd = (
        baskets_rdd
        .mapPartitions(phase1_mapper)   # map each partition to its local frequent itemsets
        .distinct() # remove duplicate candidates across partitions
    )

    global_candidates = candidate_rdd.collect()

    phase1_time = time.time() - phase1_start

    candidate_rdd.unpersist()

    print(
        f"Phase 1 candidates: "
        f"{len(global_candidates)}"
    )

    phase2_start = time.time()

    # broadcast the global candidates to all nodes for phase 2
    broadcast_candidates = (
        spark.sparkContext.broadcast(
            global_candidates
        )
    )

    def phase2_mapper(basket):
        basket_items = set(basket.keys())

        for candidate in broadcast_candidates.value:
            if candidate.issubset(
                basket_items
            ):
                volume = min(
                    basket[item]
                    for item in candidate
                )

                yield (
                    candidate,
                    (
                        1,
                        math.log1p(volume),
                        volume
                    )
                )

    final_rdd = (
        baskets_rdd
        .flatMap(phase2_mapper)
        .reduceByKey(
            lambda a, b: (
                a[0] + b[0],
                a[1] + b[1],
                max(a[2], b[2])
            )
        )
        .filter(
            lambda x:
            x[1][0] >= global_min_support
        )
    )

    final_itemsets = dict(
        final_rdd.collect()
    )

    phase2_time = time.time() - phase2_start

    final_rdd.unpersist()
    broadcast_candidates.unpersist()

    total_time = time.time() - start_time

    statistics = {
        "total_time": total_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "global_candidates": len(
            global_candidates
        ),
        "total_itemsets": len(
            final_itemsets
        )
    }

    return final_itemsets, statistics


def save_json(data, path):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            indent=2
        )


def save_csv(rows, path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def build_quantity_dataframe(
    itemsets,
    item_dict
):
    rows = []

    for itemset, (
        support,
        total_log_volume,
        max_quantity
    ) in itemsets.items():

        if len(itemset) < 2:
            continue

        average_log_quantity = (
            total_log_volume / support
        )

        item_names = " + ".join(
            item_dict.get(
                item,
                "UNKNOWN"
            )
            for item in itemset
        )

        rows.append({
            "Itemset": item_names,
            "Support": support,
            "Avg_Log_Qty": average_log_quantity,
            "Max_Qty": max_quantity,
            "Length": len(itemset)
        })

    return pd.DataFrame(rows)


def save_top_itemsets(df, path, top_n):
    top = (
        df.sort_values(
            "Support",
            ascending=False
        )
        .head(top_n)
    )

    top.to_csv(
        path,
        index=False
    )

    return top


def plot_apriori_dashboard(
    experiment_results,
    output_path
):
    supports = sorted(
        experiment_results.keys(),
        reverse=True
    )

    labels = [
        f"{support * 100:.1f}%"
        for support in supports
    ]

    fig = plt.figure(
        figsize=(18, 12)
    )

    total_times = [
        experiment_results[s]["total_time"]
        for s in supports
    ]

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(
        labels,
        total_times,
        marker="o"
    )
    ax1.set_title(
        "Distributed Execution Time"
    )
    ax1.set_ylabel(
        "Time (seconds)"
    )

    for i, value in enumerate(total_times):
        ax1.text(
            i,
            value,
            f"{value:.1f}s",
            ha="center",
            va="bottom"
        )

    total_itemsets = [
        experiment_results[s]["total_itemsets"]
        for s in supports
    ]

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(
        labels,
        total_itemsets
    )
    ax2.set_title(
        "Frequent Itemsets Discovered"
    )
    ax2.set_ylabel(
        "Itemsets"
    )

    lowest_support = supports[-1]
    levels = experiment_results[
        lowest_support
    ]["levels"]

    level_ids = sorted(levels.keys())

    candidates = [
        levels[k]["candidates"]
        for k in level_ids
    ]

    frequent = [
        levels[k]["frequent"]
        for k in level_ids
    ]

    ax3 = fig.add_subplot(2, 2, 3)
    x = range(len(level_ids))

    ax3.bar(
        [value - 0.2 for value in x],
        candidates,
        width=0.4,
        label="Candidates"
    )

    ax3.bar(
        [value + 0.2 for value in x],
        frequent,
        width=0.4,
        label="Frequent"
    )

    ax3.set_xticks(
        list(x)
    )

    ax3.set_xticklabels(
        [f"Level {k}" for k in level_ids]
    )

    ax3.set_yscale("log")
    ax3.set_title(
        f"Pruning at {lowest_support * 100:.1f}% Support"
    )
    ax3.legend()

    level_times = [
        levels[k]["time"]
        for k in level_ids
    ]

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(
        [f"Level {k}" for k in level_ids],
        level_times
    )

    ax4.set_title(
        "Execution Time per Level"
    )
    ax4.set_ylabel(
        "Time (seconds)"
    )

    fig.suptitle(
        "Quantity-Aware A-Priori",
        fontsize=18
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_son_dashboard(
    experiment_results,
    output_path
):
    supports = sorted(
        experiment_results.keys(),
        reverse=True
    )

    labels = [
        f"{support * 100:.1f}%"
        for support in supports
    ]

    fig = plt.figure(
        figsize=(18, 12)
    )

    total_times = [
        experiment_results[s]["total_time"]
        for s in supports
    ]

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(
        labels,
        total_times,
        marker="o"
    )

    ax1.set_title(
        "SON Execution Time"
    )
    ax1.set_ylabel(
        "Time (seconds)"
    )

    itemsets = [
        experiment_results[s]["total_itemsets"]
        for s in supports
    ]

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.bar(
        labels,
        itemsets
    )

    ax2.set_title(
        "Frequent Itemsets Discovered"
    )

    lowest_support = supports[-1]

    global_candidates = (
        experiment_results[
            lowest_support
        ]["global_candidates"]
    )

    confirmed = (
        experiment_results[
            lowest_support
        ]["total_itemsets"]
    )

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(
        [
            "Global Candidates",
            "Frequent Itemsets"
        ],
        [
            global_candidates,
            confirmed
        ]
    )

    ax3.set_yscale("log")
    ax3.set_title(
        f"Candidate Pruning at "
        f"{lowest_support * 100:.1f}%"
    )

    phase1 = [
        experiment_results[s]["phase1_time"]
        for s in supports
    ]

    phase2 = [
        experiment_results[s]["phase2_time"]
        for s in supports
    ]

    ax4 = fig.add_subplot(2, 2, 4)

    ax4.bar(
        labels,
        phase1,
        label="Phase 1"
    )

    ax4.bar(
        labels,
        phase2,
        bottom=phase1,
        label="Phase 2"
    )

    ax4.set_title(
        "Execution Time by Phase"
    )
    ax4.set_ylabel(
        "Time (seconds)"
    )
    ax4.legend()

    fig.suptitle(
        "Quantity-Aware SON",
        fontsize=18
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_quantity_analysis(
    itemsets,
    item_dict,
    output_path
):
    df = build_quantity_dataframe(
        itemsets,
        item_dict
    )

    if df.empty:
        print(
            "No multi-itemsets available "
            "for the quantity analysis."
        )
        return

    fig = plt.figure(
        figsize=(16, 7)
    )

    ax1 = fig.add_subplot(1, 2, 1)

    sns.scatterplot(
        data=df,
        x="Support",
        y="Avg_Log_Qty",
        size="Max_Qty",
        hue="Length",
        sizes=(50, 500),
        alpha=0.7,
        ax=ax1
    )

    ax1.set_title(
        "Frequency vs. Average Co-Purchased Volume"
    )

    ax1.set_xlabel(
        "Support Count"
    )

    ax1.set_ylabel(
        "Average Log Quantity"
    )

    top_volume = (
        df.nlargest(
            3,
            "Avg_Log_Qty"
        )
    )

    for _, row in top_volume.iterrows():
        label = row["Itemset"]

        if len(label) > 30:
            label = label[:30] + "..."

        ax1.text(
            row["Support"],
            row["Avg_Log_Qty"],
            label,
            fontsize=8
        )

    ax2 = fig.add_subplot(1, 2, 2)

    sns.histplot(
        data=df,
        x="Max_Qty",
        bins=30,
        kde=True,
        ax=ax2
    )

    ax2.set_title(
        "Maximum Quantity Distribution"
    )

    ax2.set_xlabel(
        "Maximum Quantity in a Single Transaction"
    )

    ax2.set_ylabel(
        "Number of Patterns"
    )

    fig.suptitle(
        "Quantity-Aware Pattern Analysis",
        fontsize=18
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def save_experiment_statistics(
    output_dir,
    apriori_results,
    son_results
):
    apriori_rows = []

    for support, stats in sorted(
        apriori_results.items(),
        reverse=True
    ):
        apriori_rows.append({
            "support": support,
            "total_time_seconds": stats["total_time"],
            "total_itemsets": stats["total_itemsets"]
        })

    save_csv(
        apriori_rows,
        os.path.join(
            output_dir,
            "apriori_summary.csv"
        )
    )

    apriori_level_rows = []

    for support, stats in sorted(
        apriori_results.items(),
        reverse=True
    ):
        for level, values in sorted(
            stats["levels"].items()
        ):
            apriori_level_rows.append({
                "support": support,
                "level": level,
                "candidates": values["candidates"],
                "frequent": values["frequent"],
                "time_seconds": values["time"]
            })

    save_csv(
        apriori_level_rows,
        os.path.join(
            output_dir,
            "apriori_levels.csv"
        )
    )

    son_rows = []

    for support, stats in sorted(
        son_results.items(),
        reverse=True
    ):
        son_rows.append({
            "support": support,
            "total_time_seconds": stats["total_time"],
            "phase1_time_seconds": stats["phase1_time"],
            "phase2_time_seconds": stats["phase2_time"],
            "global_candidates": stats["global_candidates"],
            "total_itemsets": stats["total_itemsets"]
        })

    save_csv(
        son_rows,
        os.path.join(
            output_dir,
            "son_summary.csv"
        )
    )


def main():
    args = parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    start_time = time.time()

    spark = create_spark_session(
        args.cores
    )

    try:
        pdf = load_dataset(
            input_path=args.input,
            dataset_id=args.dataset_id
        )

        local_baskets, item_dict = (
            prepare_quantity_aware_baskets(
                spark,
                pdf
            )
        )

        baskets_rdd = (
            spark.sparkContext
            .parallelize(
                local_baskets,
                numSlices=args.partitions
            )
            .cache()
        )

        total_baskets = baskets_rdd.count()

        print("\nFinal Spark dataset configuration")
        print(f"Baskets: {total_baskets}")
        print(f"Partitions: {args.partitions}")
        print(f"Execution threads: {args.cores}")

        save_json(
            {
                "cores": args.cores,
                "partitions": args.partitions,
                "supports": args.supports,
                "total_baskets": total_baskets,
                "input_dataset": args.input
            },
            os.path.join(
                args.output_dir,
                "experiment_config.json"
            )
        )

        apriori_results = {}
        son_results = {}

        apriori_itemsets_by_support = {}
        son_itemsets_by_support = {}

        for support in args.supports:
            itemsets, stats = (
                distributed_apriori_qa(
                    spark,
                    baskets_rdd,
                    total_baskets,
                    support
                )
            )

            apriori_results[support] = stats
            apriori_itemsets_by_support[support] = itemsets

        for support in args.supports:
            itemsets, stats = (
                quantity_aware_son(
                    spark,
                    baskets_rdd,
                    total_baskets,
                    support
                )
            )

            son_results[support] = stats
            son_itemsets_by_support[support] = itemsets

        save_experiment_statistics(
            args.output_dir,
            apriori_results,
            son_results
        )

        lowest_support = min(
            args.supports
        )

        apriori_df = build_quantity_dataframe(
            apriori_itemsets_by_support[
                lowest_support
            ],
            item_dict
        )

        son_df = build_quantity_dataframe(
            son_itemsets_by_support[
                lowest_support
            ],
            item_dict
        )

        if not apriori_df.empty:
            save_top_itemsets(
                apriori_df,
                os.path.join(
                    args.output_dir,
                    "apriori_top_itemsets.csv"
                ),
                args.top_n
            )

        if not son_df.empty:
            save_top_itemsets(
                son_df,
                os.path.join(
                    args.output_dir,
                    "son_top_itemsets.csv"
                ),
                args.top_n
            )

        plot_apriori_dashboard(
            apriori_results,
            os.path.join(
                args.output_dir,
                "apriori_dashboard.png"
            )
        )

        plot_son_dashboard(
            son_results,
            os.path.join(
                args.output_dir,
                "son_dashboard.png"
            )
        )

        plot_quantity_analysis(
            apriori_itemsets_by_support[
                lowest_support
            ],
            item_dict,
            os.path.join(
                args.output_dir,
                "apriori_quantity_analysis.png"
            )
        )

        plot_quantity_analysis(
            son_itemsets_by_support[
                lowest_support
            ],
            item_dict,
            os.path.join(
                args.output_dir,
                "son_quantity_analysis.png"
            )
        )

        print("\n" + "=" * 70)
        print("Experiment completed")
        print("=" * 70)
        print(
            f"Results saved to: "
            f"{os.path.abspath(args.output_dir)}"
        )
        print(
            f"Total wall-clock time: "
            f"{time.time() - start_time:.2f} seconds"
        )

        baskets_rdd.unpersist()

    finally:
        spark.stop()


if __name__ == "__main__":
    main()