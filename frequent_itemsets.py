import argparse
import time
import json
import itertools
import os
from collections import defaultdict

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.fpm import FPGrowth


# ==========================================
# DISTRIBUTED A-PRIORI
# ==========================================

def generate_candidates(prev_frequent_itemsets, k):
    candidates = set()
    prev_itemsets_list = list(prev_frequent_itemsets.keys())

    for i in range(len(prev_itemsets_list)):
        for j in range(i + 1, len(prev_itemsets_list)):
            union_set = prev_itemsets_list[i] | prev_itemsets_list[j]

            if len(union_set) == k:
                is_valid = True

                for subset in itertools.combinations(union_set, k - 1):
                    if frozenset(subset) not in prev_frequent_itemsets:
                        is_valid = False
                        break

                if is_valid:
                    candidates.add(union_set)

    return candidates


def distributed_apriori(
    spark,
    baskets_rdd,
    total_baskets,
    min_support_ratio
):
    start_time = time.time()

    min_support_count = max(
        1,
        int(total_baskets * min_support_ratio)
    )

    all_frequent_itemsets = {}

    algorithm_stats = {
        "total_time": 0,
        "total_itemsets": 0,
        "levels": {}
    }

    # ------------------------------------------
    # LEVEL 1
    # ------------------------------------------

    level_start_time = time.time()

    l1_counts_rdd = (
        baskets_rdd
        .flatMap(
            lambda basket: [
                (frozenset([item]), 1)
                for item in basket
            ]
        )
        .reduceByKey(lambda a, b: a + b)
        .cache()
    )

    total_unique_items = l1_counts_rdd.count()

    current_frequent = dict(
        l1_counts_rdd
        .filter(lambda x: x[1] >= min_support_count)
        .collect()
    )

    all_frequent_itemsets.update(current_frequent)

    algorithm_stats["levels"][1] = {
        "candidates_count": total_unique_items,
        "frequent_count": len(current_frequent),
        "execution_time": time.time() - level_start_time
    }

    l1_counts_rdd.unpersist()

    # ------------------------------------------
    # LEVEL K > 1
    # ------------------------------------------

    k = 2

    while current_frequent:
        level_start_time = time.time()

        candidates = generate_candidates(
            current_frequent,
            k
        )

        if not candidates:
            break

        bc_candidates = spark.sparkContext.broadcast(
            candidates
        )

        def count_candidates(basket):
            local_candidates = bc_candidates.value

            for candidate in local_candidates:
                if candidate.issubset(basket):
                    yield candidate, 1

        k_counts_rdd = (
            baskets_rdd
            .flatMap(count_candidates)
            .reduceByKey(lambda a, b: a + b)
            .filter(lambda x: x[1] >= min_support_count)
        )

        current_frequent = dict(
            k_counts_rdd.collect()
        )

        all_frequent_itemsets.update(
            current_frequent
        )

        algorithm_stats["levels"][k] = {
            "candidates_count": len(candidates),
            "frequent_count": len(current_frequent),
            "execution_time": time.time() - level_start_time
        }

        bc_candidates.unpersist()

        k += 1

    algorithm_stats["total_time"] = (
        time.time() - start_time
    )

    algorithm_stats["total_itemsets"] = (
        len(all_frequent_itemsets)
    )

    return algorithm_stats


# ==========================================
# FP-GROWTH
# ==========================================

def run_fpgrowth_experiment(
    df_baskets,
    support_thresholds
):
    experiment_results = {}

    # Warm-up run
    fp_warmup = FPGrowth(
        itemsCol="items",
        predictionCol="prediction",
        minSupport=0.1
    )

    fp_warmup.fit(
        df_baskets
    ).freqItemsets.collect()

    for min_sup in support_thresholds:

        fp = FPGrowth(
            itemsCol="items",
            predictionCol="prediction",
            minSupport=min_sup
        )

        t0 = time.time()

        model = fp.fit(df_baskets)

        t1 = time.time()

        freq_df = (
            model.freqItemsets
            .withColumn(
                "length",
                F.size(F.col("items"))
            )
        )

        itemsets = freq_df.collect()

        t2 = time.time()

        experiment_results[min_sup] = {
            "build_time": t1 - t0,
            "mine_time": t2 - t1,
            "total_time": t2 - t0,
            "count": len(itemsets),
            "max_length": (
                max(
                    row["length"]
                    for row in itemsets
                )
                if itemsets
                else 0
            )
        }

    return experiment_results


# ==========================================
# SON ALGORITHM
# ==========================================

def get_local_frequent_itemsets(
    baskets,
    min_support_count
):
    frequent_itemsets = []

    item_counts = defaultdict(int)

    # Count single items
    for basket in baskets:
        for item in basket:
            item_counts[
                frozenset([item])
            ] += 1

    current_frequent = {
        itemset: count
        for itemset, count in item_counts.items()
        if count >= min_support_count
    }

    frequent_itemsets.extend(
        current_frequent.keys()
    )

    k = 2

    while current_frequent:

        candidates = set()

        prev_itemsets = list(
            current_frequent.keys()
        )

        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):

                union_set = (
                    prev_itemsets[i]
                    | prev_itemsets[j]
                )

                if len(union_set) == k:
                    candidates.add(
                        union_set
                    )

        if not candidates:
            break

        candidate_counts = defaultdict(int)

        for basket in baskets:

            basket_set = set(basket)

            for candidate in candidates:

                if candidate.issubset(
                    basket_set
                ):
                    candidate_counts[
                        candidate
                    ] += 1

        current_frequent = {
            candidate: count
            for candidate, count
            in candidate_counts.items()
            if count >= min_support_count
        }

        frequent_itemsets.extend(
            current_frequent.keys()
        )

        k += 1

    return frequent_itemsets


def run_son_algorithm_with_stats(
    spark_context,
    baskets_rdd,
    total_baskets,
    min_support_ratio
):
    start_time = time.time()

    son_stats = {}

    son_stats["num_partitions"] = (
        baskets_rdd.getNumPartitions()
    )

    global_support_count = max(
        1,
        int(
            total_baskets
            * min_support_ratio
        )
    )

    # ------------------------------------------
    # PHASE 1
    # ------------------------------------------

    p1_start = time.time()

    def first_map_function(
        partition_iterator
    ):
        baskets_chunk = list(
            partition_iterator
        )

        chunk_size = len(
            baskets_chunk
        )

        if chunk_size == 0:
            return iter([])

        local_support_count = max(
            1,
            int(
                chunk_size
                * min_support_ratio
            )
        )

        local_frequent = (
            get_local_frequent_itemsets(
                baskets_chunk,
                local_support_count
            )
        )

        return iter(
            [
                (itemset, 1)
                for itemset
                in local_frequent
            ]
        )

    phase1_mapped = (
        baskets_rdd
        .mapPartitions(
            first_map_function
        )
    )

    candidates_rdd = (
        phase1_mapped
        .reduceByKey(
            lambda a, b: 1
        )
        .keys()
    )

    global_candidates = (
        candidates_rdd.collect()
    )

    son_stats["phase1_time"] = (
        time.time() - p1_start
    )

    son_stats["candidates_generated"] = (
        len(global_candidates)
    )

    # ------------------------------------------
    # PHASE 2
    # ------------------------------------------

    p2_start = time.time()

    broadcast_candidates = (
        spark_context.broadcast(
            global_candidates
        )
    )

    def second_map_function(
        partition_iterator
    ):
        candidates = (
            broadcast_candidates.value
        )

        baskets_chunk = list(
            partition_iterator
        )

        candidate_counts = defaultdict(
            int
        )

        for basket in baskets_chunk:

            basket_set = set(basket)

            for candidate in candidates:

                if candidate.issubset(
                    basket_set
                ):
                    candidate_counts[
                        candidate
                    ] += 1

        return iter(
            candidate_counts.items()
        )

    phase2_mapped = (
        baskets_rdd
        .mapPartitions(
            second_map_function
        )
    )

    global_frequent_itemsets = (
        phase2_mapped
        .reduceByKey(
            lambda count1, count2:
            count1 + count2
        )
        .filter(
            lambda kv:
            kv[1] >= global_support_count
        )
        .collect()
    )

    son_stats["phase2_time"] = (
        time.time() - p2_start
    )

    son_stats["final_frequent_itemsets"] = (
        len(global_frequent_itemsets)
    )

    son_stats["total_time"] = (
        time.time() - start_time
    )

    broadcast_candidates.unpersist()

    return son_stats


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Frequent Itemset Mining "
            "using Apache Spark on a remote server"
        )
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=(
            "Path to the input Parquet file "
            "on the remote server"
        )
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=(
            "Directory where experiment results "
            "will be stored on the remote server"
        )
    )

    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help=(
            "Optional Spark master URL. "
            "If omitted, Spark uses the server configuration."
        )
    )

    parser.add_argument(
        "--partitions",
        type=int,
        default=None,
        help=(
            "Optional number of partitions for the baskets RDD."
        )
    )

    args = parser.parse_args()

    # ------------------------------------------
    # Validate paths
    # ------------------------------------------

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(
            f"Input path does not exist: "
            f"{args.input_path}"
        )

    os.makedirs(
        args.output_path,
        exist_ok=True
    )

    # ------------------------------------------
    # Initialize Spark
    # ------------------------------------------

    builder = (
        SparkSession.builder
        .appName(
            "FrequentItemsetMining_UniVR"
        )
    )

    if args.master:
        builder = builder.master(
            args.master
        )

    spark = builder.getOrCreate()

    sc = spark.sparkContext

    sc.setLogLevel("WARN")

    print("=" * 60)
    print("FREQUENT ITEMSET MINING")
    print("=" * 60)

    print(
        f"Input path: {args.input_path}"
    )

    print(
        f"Output path: {args.output_path}"
    )

    print(
        f"Spark master: {sc.master}"
    )

    print(
        f"Spark application ID: "
        f"{sc.applicationId}"
    )

    # ------------------------------------------
    # Read input
    # ------------------------------------------

    print("\nReading baskets...")

    df_baskets = (
        spark.read
        .parquet(args.input_path)
        .cache()
    )

    total_baskets = (
        df_baskets.count()
    )

    print(
        f"Total baskets: {total_baskets}"
    )

    # ------------------------------------------
    # Prepare RDD
    # ------------------------------------------

    baskets_rdd = (
        df_baskets.rdd
        .map(
            lambda row:
            set(row["items"])
        )
        .cache()
    )

    if args.partitions:
        baskets_rdd = (
            baskets_rdd
            .repartition(
                args.partitions
            )
            .cache()
        )

    print(
        f"RDD partitions: "
        f"{baskets_rdd.getNumPartitions()}"
    )

    # ------------------------------------------
    # Experiment configuration
    # ------------------------------------------

    support_thresholds = [
        0.03,
        0.025,
        0.02
    ]

    all_results = {
        "apriori": {},
        "fpgrowth": {},
        "son": {},
        "metadata": {
            "total_baskets": total_baskets,
            "thresholds": support_thresholds,
            "spark_master": sc.master,
            "spark_application_id": (
                sc.applicationId
            ),
            "num_partitions": (
                baskets_rdd.getNumPartitions()
            )
        }
    }

    # ------------------------------------------
    # DISTRIBUTED A-PRIORI
    # ------------------------------------------

    print(
        "\n--- RUNNING DISTRIBUTED A-PRIORI ---"
    )

    for sup in support_thresholds:

        print(
            f"Apriori Support: {sup}"
        )

        stats = distributed_apriori(
            spark,
            baskets_rdd,
            total_baskets,
            sup
        )

        all_results["apriori"][
            str(sup)
        ] = stats

    # ------------------------------------------
    # FP-GROWTH
    # ------------------------------------------

    print(
        "\n--- RUNNING FP-GROWTH ---"
    )

    fp_results = (
        run_fpgrowth_experiment(
            df_baskets,
            support_thresholds
        )
    )

    for k, v in fp_results.items():

        all_results["fpgrowth"][
            str(k)
        ] = v

    # ------------------------------------------
    # SON
    # ------------------------------------------

    print(
        "\n--- RUNNING SON ALGORITHM ---"
    )

    for sup in support_thresholds:

        print(
            f"SON Support: {sup}"
        )

        stats = (
            run_son_algorithm_with_stats(
                sc,
                baskets_rdd,
                total_baskets,
                sup
            )
        )

        all_results["son"][
            str(sup)
        ] = stats

    # ------------------------------------------
    # Save results
    # ------------------------------------------

    timestamp = int(time.time())

    output_file = os.path.join(
        args.output_path,
        f"experiment_metrics_{timestamp}.json"
    )

    print(
        f"\nSaving results to: "
        f"{output_file}"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    print(
        "\nExperiment completed successfully."
    )

    print(
        f"Results saved to: {output_file}"
    )

    # ------------------------------------------
    # Cleanup
    # ------------------------------------------

    baskets_rdd.unpersist()
    df_baskets.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()