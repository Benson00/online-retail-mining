import argparse
import time
import json
import itertools
import os
from collections import defaultdict

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.fpm import FPGrowth



# DISTRIBUTED A-PRIORI


def generate_candidates(prev_frequent_itemsets, k):
    """
    Generate candidate itemsets of size k from the previous level's frequent itemsets.
    """
    candidates = set()
    prev_itemsets_list = list(prev_frequent_itemsets.keys())

    # Loop through each frequent itemset using its index 
    # and compare it with every following itemset to avoid duplicate pairs.
    for i in range(len(prev_itemsets_list)):
        for j in range(i + 1, len(prev_itemsets_list)):
            
            # Combine the two itemsets
            union_set = prev_itemsets_list[i] | prev_itemsets_list[j]
            
            # Check whether the union contains exactly k different items 
            # then control if all subsets of size k-1 are frequent
            if len(union_set) == k:
                is_valid = True
                # Generate every possible subset of the candidate containing k-1 items
                for subset in itertools.combinations(union_set, k - 1):
                    # Check whether the current subset was frequent in the previous level
                    if frozenset(subset) not in prev_frequent_itemsets:
                        is_valid = False
                        break

                if is_valid:
                    candidates.add(union_set)

    return candidates


def distributed_apriori(spark, baskets_rdd, total_baskets, min_support_ratio):
    """
    Define the main Distributed Apriori function.

    It receives the Spark context, the RDD containing the baskets, the total number of baskets, and the minimum support ratio.
    """

    start_time = time.time()

    # Convert the minimum support ratio into the minimum number of baskets
    # e.g. if min_support_ratio = 0.02 and total_baskets = 1000, then min_support_count = 20
    min_support_count = max(1, int(total_baskets * min_support_ratio))
    
    # dictionary to store all frequent itemsets found across all levels
    all_frequent_itemsets = {}
    # dictionary to store statistics about the algorithm
    algorithm_stats = {
        "total_time": 0,
        "total_itemsets": 0,
        "levels": {}
    }

    # LEVEL 1 - frequent itemsets containing only one item
    level_start_time = time.time()
    l1_counts_rdd = (
        baskets_rdd # RDD containing the baskets
        .flatMap(lambda basket: [(frozenset([item]), 1) for item in basket]) # Convert every item in every basket into a pair (single-itemset, 1).
        .reduceByKey(lambda a, b: a + b) # Sum the values associated with the same itemset
        .cache() # Store this RDD in memory
    )

    # Count the total number of different individual items
    total_unique_items = l1_counts_rdd.count() 
    
    current_frequent = dict(
        l1_counts_rdd
        .filter(lambda x: x[1] >= min_support_count) # Keep only the items whose support count is at least the minimum support count
        .collect() # Collect the results into a dictionary where the keys are the frequent itemsets and the values are their support counts
    )

    # Add all frequent itemsets from Level 1 to the dictionary containing all results
    all_frequent_itemsets.update(current_frequent)

    algorithm_stats["levels"][1] = {
        "candidates_count": total_unique_items,
        "frequent_count": len(current_frequent),
        "execution_time": time.time() - level_start_time
    }

    # Remove the Level 1 RDD from memory because it is no longer needed
    l1_counts_rdd.unpersist()

    # LEVEL K > 1
    k = 2
    while current_frequent:
        level_start_time = time.time()
        # Generate candidate itemsets of size k using the current frequent itemsets 
        candidates = generate_candidates(current_frequent, k)

        if not candidates:
            break
        # Send the candidate set to all Spark workers using a broadcast variable
        bc_candidates = spark.sparkContext.broadcast(candidates)

        def count_candidates(basket):
            """function that checks which candidates are contained in one basket"""
            local_candidates = bc_candidates.value
            for candidate in local_candidates:
                if candidate.issubset(basket):
                    yield candidate, 1 # If the candidate is contained in the basket -> output: (candidate, 1).

        
        k_counts_rdd = (
            baskets_rdd
            .flatMap(count_candidates) # Apply count_candidates to every basket
            .reduceByKey(lambda a, b: a + b) # count the number of baskets containing each candidate itemset
            .filter(lambda x: x[1] >= min_support_count) # filter out candidates that do not meet the minimum support
        )

        # collect the results into a dictionary where the keys are the frequent itemsets and the values are their support counts
        current_frequent = dict(k_counts_rdd.collect())

        # Add all frequent itemsets from Level k to the dictionary containing all results
        all_frequent_itemsets.update(current_frequent)

        algorithm_stats["levels"][k] = {
            "candidates_count": len(candidates),
            "frequent_count": len(current_frequent),
            "execution_time": time.time() - level_start_time
        }

        # Remove the broadcast variable from memory because this level is finished
        bc_candidates.unpersist()
        
        k += 1

    algorithm_stats["total_time"] = time.time() - start_time
    algorithm_stats["total_itemsets"] = len(all_frequent_itemsets)
    return algorithm_stats



# FP-GROWTH

def run_fpgrowth_experiment(df_baskets, support_thresholds):
    """
    Run FP-Growth experiments for different minimum support thresholds.
    
    For each threshold, it measures model-building time, itemset extraction time, total execution time, number of frequent itemsets, and maximum itemset length.
    """

    # dictionary to store the results for each support threshold.
    experiment_results = {}
    

    ###

    # Warm-up run
    # Run FP-Growth once before the real experiments 
    # to reduce the impact of Spark startup, initialization, and first-execution overhead.
    # The results are discarded.
    
    fp_warmup = FPGrowth(
        itemsCol="items",
        predictionCol="prediction",
        minSupport=0.1
    )

    fp_warmup.fit(
        df_baskets
    ).freqItemsets.collect()

    ###

    for min_sup in support_thresholds:

        # Create a new FP-Growth estimator using the current support threshold.
        fp = FPGrowth(
            itemsCol="items",
            predictionCol="prediction",
            minSupport=min_sup
        )
        
        t0 = time.time()

        # Run FP-Growth on the baskets and build the FP-Growth model.
        model = fp.fit(df_baskets)
        

        t1 = time.time()

        # Get the frequent itemsets generated by the model.
        # Add a new column called "length" containing the number of items
        # in each frequent itemset.
        freq_df = (
            model.freqItemsets
            .withColumn(
                "length",
                F.size(F.col("items"))
            )
        )

        # Execute the Spark computation and bring all frequent itemsets 
        # from the workers to the driver as a local Python list.
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


# SON ALGORITHM

def get_local_frequent_itemsets(baskets, min_support_count):
    
    frequent_itemsets = []
    
    # count how many baskets contain each single item
    item_counts = defaultdict(int)

    # count occurrences of each item in the baskets
    for basket in baskets:
        for item in basket:
            item_counts[frozenset([item])] += 1

    # keep only those items that meet the minimum support count
    current_frequent = {
        itemset: count
        for itemset, count in item_counts.items()
        if count >= min_support_count
    }
    
    
    frequent_itemsets.extend(current_frequent.keys())
    k = 2

    # Continue while there are frequent itemsets of the previous size
    while current_frequent:
        candidates = set()
        prev_itemsets = list(current_frequent.keys())

        # generate candidate itemsets of size k by combining pairs of frequent itemsets of size k-1
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                union_set = prev_itemsets[i] | prev_itemsets[j]

                if len(union_set) == k:
                    candidates.add(union_set)

        if not candidates:
            break

        candidate_counts = defaultdict(int)

        # Scan all baskets in the current partition
        for basket in baskets:
            basket_set = set(basket)
            # Check every candidate against the current basket and if it is contained, increment its count
            for candidate in candidates:
                if candidate.issubset(basket_set):
                    candidate_counts[candidate] += 1

        # Filter candidates that satisfy the local support threshold
        current_frequent = {
            candidate: count
            for candidate, count in candidate_counts.items()
            if count >= min_support_count
        }

        # Add the new frequent itemsets to the result
        frequent_itemsets.extend(current_frequent.keys())
        # next itemset size
        k += 1

    return frequent_itemsets


def run_son_algorithm_with_stats(spark_context, baskets_rdd, total_baskets, min_support_ratio):
    
    start_time = time.time()
    son_stats = {}
    son_stats["num_partitions"] = baskets_rdd.getNumPartitions()

    # global minimum support count (at least 1)
    global_support_count = max(
        1,
        int(
            total_baskets * min_support_ratio
        )
    )

    # PHASE 1 - LOCAL FREQUENT ITEMSET MINING

    p1_start = time.time()

    def first_map_function(partition_iterator):
        """
        first map function for the SON algorithm. It receives an iterator over the baskets in the current partition and returns an iterator over the local frequent itemsets found in this partition.
        """

        # Load all baskets belonging to this Spark partition
        baskets_chunk = list(partition_iterator)
        # baskets in this partition
        chunk_size = len(baskets_chunk)
        # if the partition is empty, return an empty iterator
        if chunk_size == 0:
            return iter([])

        # local minimum support count (at least 1)
        local_support_count = max(
            1,
            int(
                chunk_size * min_support_ratio
            )
        )

        # get the local frequent itemsets in this partition using the local support count
        local_frequent = get_local_frequent_itemsets(
            baskets_chunk,
            local_support_count
        )

        # return iterator over the local frequent itemsets, each paired with a count of 1
        return iter([
            (itemset, 1)
            for itemset in local_frequent
        ])

    # Process each partition independently to find local frequent itemsets
    phase1_mapped = baskets_rdd.mapPartitions(
        first_map_function
    )

    # Merge equal itemsets coming from different partitions. 
    # If an itemset is frequent in at least one partition
    # it becomes a global candidate.
    candidates_rdd = (
        phase1_mapped
        .reduceByKey(lambda a, b: 1)
        .keys()
    )

    global_candidates = candidates_rdd.collect()

    son_stats["phase1_time"] = time.time() - p1_start

    son_stats["candidates_generated"] = len(
        global_candidates
    )

    # PHASE 2

    p2_start = time.time()

    # Broadcast the candidate itemsets to all Spark workers, avoid sending the same data multiple times to each worker.
    broadcast_candidates = spark_context.broadcast(
        global_candidates
    )

    def second_map_function(partition_iterator):
        """
        second map function for the SON algorithm. It receives an iterator over the baskets in the current partition and returns an iterator over the candidate itemsets and their counts.
        """
        candidates = broadcast_candidates.value
        baskets_chunk = list(partition_iterator)
        candidate_counts = defaultdict(int)

        # for each basket in the current partition, check which candidates are contained in it and increment their counts
        for basket in baskets_chunk:
            basket_set = set(basket)

            for candidate in candidates:
                if candidate.issubset(basket_set):
                    candidate_counts[candidate] += 1

        return iter(candidate_counts.items())

    # Process each partition independently to count the occurrences of each candidate itemset in the baskets of that partition
    phase2_mapped = baskets_rdd.mapPartitions(
        second_map_function
    )

    # Merge the counts of the same candidate itemsets coming from different partitions and filter out those that do not meet the global minimum support count
    global_frequent_itemsets = (
        phase2_mapped
        .reduceByKey(
            lambda count1, count2: count1 + count2
        )
        .filter(
            lambda kv: kv[1] >= global_support_count
        )
        .collect()
    )

    son_stats["phase2_time"] = time.time() - p2_start

    son_stats["final_frequent_itemsets"] = len(
        global_frequent_itemsets
    )

    son_stats["total_time"] = time.time() - start_time

    # Release the broadcast variable from Spark's memory 
    broadcast_candidates.unpersist()

    return son_stats


# MAIN EXECUTION

# 1) Parse the arguments
# 2) Initialize Spark
# 3) Read input data
# 4) Run the experiments
# 5) Save the results
# 6) Clean up resources

def main():

    # 1) Parse command-line arguments

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


    # Validate paths

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(
            f"Input path does not exist: "
            f"{args.input_path}"
        )

    os.makedirs(
        args.output_path,
        exist_ok=True
    )

    # 2) Initialize Spark

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

    print(f"Input path: {args.input_path}")

    print(f"Output path: {args.output_path}")

    print(f"Spark master: {sc.master}")

    print(f"Spark application ID: {sc.applicationId}")


    # 3) Read input

    print("\nReading baskets...")

    df_baskets = (
        spark.read.parquet(args.input_path).cache()
    )

    total_baskets = (
        df_baskets.count()
    )

    print(
        f"Total baskets: {total_baskets}"
    )

    # Prepare RDD

    # create the rdd of baskets, where each basket is represented as a set of items
    baskets_rdd = (
        df_baskets.rdd 
        .map(       
            lambda row:
            set(row["items"])
        ) 
        .cache()
    )

    # If the user specifies a number of partitions, repartition the RDD with that number of partitions and cache it in memory. 
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

    # 4)Experiment configuration

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

    # DISTRIBUTED A-PRIORI

    print("\n--- RUNNING DISTRIBUTED A-PRIORI ---")

    for sup in support_thresholds:


        print(f"Apriori Support: {sup}")

        stats = distributed_apriori(
            spark,
            baskets_rdd,
            total_baskets,
            sup
        )

        all_results["apriori"][str(sup)] = stats

    # FP-GROWTH

    print("\n--- RUNNING FP-GROWTH ---")

    fp_results = (
        run_fpgrowth_experiment(
            df_baskets,
            support_thresholds
        )
    )

    for k, v in fp_results.items():
        all_results["fpgrowth"][str(k)] = v

    # SON

    print("\n--- RUNNING SON ALGORITHM ---")

    for sup in support_thresholds:

        print(f"SON Support: {sup}")

        stats = (
            run_son_algorithm_with_stats(
                sc,
                baskets_rdd,
                total_baskets,
                sup
            )
        )

        all_results["son"][str(sup)] = stats

    # 5) Save results

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

    print("\nExperiment completed successfully.")

    print(f"Results saved to: {output_file}")

    # 6) Cleanup

    baskets_rdd.unpersist()
    df_baskets.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()