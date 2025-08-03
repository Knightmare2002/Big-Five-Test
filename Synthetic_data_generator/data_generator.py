import pandas as pd
import numpy as np

# === Synthetic data generator with variability ===
def create_synthetic_case(index_start, n_samples, cluster, label, O_range, C_range, E_range, A_range, N_range):
    data = []
    meta = ["white", 30, 1, "male", "right", "synthetic", "IT"]

    for i in range(n_samples):
        O_mean = np.random.uniform(*O_range)
        C_mean = np.random.uniform(*C_range)
        E_mean = np.random.uniform(*E_range)
        A_mean = np.random.uniform(*A_range)
        N_mean = np.random.uniform(*N_range)

        trait = lambda mean: np.clip(np.random.normal(mean, 0.4, 10), 1, 5).round()

        row = [index_start + i] + meta
        row += trait(E_mean).tolist()
        row += trait(N_mean).tolist()
        row += trait(A_mean).tolist()
        row += trait(C_mean).tolist()
        row += trait(O_mean).tolist()
        row += [cluster, label]
        data.append(row)
    return data

# === Generate samples ===
synthetic_cases_expanded = []
index_counter = 30000

# Balanced neutral
synthetic_cases_expanded += create_synthetic_case(index_counter, 200, 3, "Balanced",
                                                  (2.5,3.5),(2.5,3.5),(2.5,3.5),(2.5,3.5),(2.5,3.5))
index_counter += 200

# Internalizer
synthetic_cases_expanded += create_synthetic_case(index_counter, 200, 2, "Internalizer",
                                                  (1,2.5),(2.5,3.5),(1,2.5),(2.5,3.5),(4,5))
index_counter += 200

# Balanced stable
synthetic_cases_expanded += create_synthetic_case(index_counter, 200, 3, "Balanced",
                                                  (3.5,4.5),(3.5,4.5),(3.5,4.5),(3.5,4.5),(1,2))
index_counter += 200

# Reserved: O high, C low
synthetic_cases_expanded += create_synthetic_case(index_counter, 200, 0, "Reserved",
                                                  (4,5),(1,2),(2.5,3.5),(2.5,3.5),(2.5,3.5))
index_counter += 200

# Striver
synthetic_cases_expanded += create_synthetic_case(index_counter, 200, 1, "Striver",
                                                  (3.5,5),(3.5,5),(3.5,5),(3.5,5),(2.5,3.5))
index_counter += 200

# High E avoid Reserved/Internalizer
synthetic_cases_expanded += create_synthetic_case(index_counter, 200, 1, "Striver",
                                                  (2.5,3.5),(2.5,3.5),(4,5),(2.5,3.5),(2.5,3.5))
index_counter += 200

# High E + low N Balanced
synthetic_cases_expanded += create_synthetic_case(index_counter, 200, 3, "Balanced",
                                                  (2.5,3.5),(2.5,3.5),(4,5),(3,4),(1.5,2.5))
index_counter += 200

# New case: Very Low Extraversion + Normal Other Traits → Reserved
synthetic_cases_expanded += create_synthetic_case(index_counter, 200, 0, "Reserved",
                                                  (2.5,3.5),(2.5,3.5),(1,2),(2.5,3.5),(2.5,3.5))
index_counter += 200

# === Columns ===
columns = ["index","race","age","engnat","gender","hand","source","country"] + \
          [f"E{i}" for i in range(1,11)] + \
          [f"N{i}" for i in range(1,11)] + \
          [f"A{i}" for i in range(1,11)] + \
          [f"C{i}" for i in range(1,11)] + \
          [f"O{i}" for i in range(1,11)] + \
          ["Cluster","Psych_Label"]

# === Build DataFrame ===
synthetic_df_expanded = pd.DataFrame(synthetic_cases_expanded, columns=columns)

# === Save updated dataset ===
output_path_expanded = "synthetic_big5_edge_cases_expanded.csv"
synthetic_df_expanded.to_csv(output_path_expanded, index=False)
output_path_expanded
