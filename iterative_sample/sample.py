import numpy as np
import random
import json
from tqdm import tqdm

MIN_STRENGH = 0.5
MAX_STRENGH = 5.0
SAMPLE_MULTIPLIER = 100

np.random.seed(42)
COSINE_SIMILARITY_THRESHOLD = 0.999


def generate_weights_dirichlet(
    prior_dist,
    train_groups,
    minimum_number,
    num_samples,
    enable_bound,
    temperature,
    maximum_usage,
    num_selected,
):
    final_samples = []

    prior_dist = prior_dist / np.sum(prior_dist)

    if enable_bound:
        number_bound = []
        for i in range(len(prior_dist)):
            number_bound.append([0.0, min(prior_dist[i] * maximum_usage, 1.0)])
    else:
        number_bound = None

    if temperature < 1.0:
        prior_dist = prior_dist**temperature
        prior_dist = prior_dist / np.sum(prior_dist)
        print("\n\nWith temperature: ", prior_dist)

    print("\n\nThe domain usage bound (maximum domain weight): ")
    for i in range(len(prior_dist)):
        print(f"{train_groups[i]}: {number_bound[i][1]}")

    for _i in range(num_samples * SAMPLE_MULTIPLIER):
        if MIN_STRENGH == MAX_STRENGH:
            samples = np.random.dirichlet(prior_dist * MIN_STRENGH, 1)
        else:
            samples = []
            min_strength_log = np.log10(MIN_STRENGH)
            max_strength_log = np.log10(MAX_STRENGH)
            for strength in np.logspace(min_strength_log, max_strength_log, 15):
                samples_per_strength = np.random.dirichlet(prior_dist * strength, 1)
                samples.append(samples_per_strength)
            samples = random.choice(samples)

        if number_bound is not None:
            for j in range(len(samples[0])):
                if samples[0][j] > number_bound[j][1]:
                    over = samples[0][j] - number_bound[j][1]
                    other_sum = 1 - samples[0][j]
                    multi_factor = 1 + (over / other_sum)
                    samples[0] = samples[0] * multi_factor
                    samples[0][j] = number_bound[j][1]

        samples = np.where(samples < minimum_number, 0.0, samples)
        samples = samples / np.sum(samples, axis=1).reshape(-1, 1)
        samples = np.round(samples / minimum_number) * minimum_number

        sample = samples[0]
        sample = sample / np.sum(sample)
        final_samples.append(sample)

    final_samples = sort_and_deduplicate(np.array(final_samples))
    print("The number of samples after deduplication (L1): ", len(final_samples))

    if len(final_samples) < num_selected:
        print(
            f"Warning: Only {len(final_samples)} samples left, less than requested {num_selected}."
        )
        selected_samples = final_samples
    else:
        selected_samples = random.sample(final_samples, num_selected)

    print("The number of selected samples: ", len(selected_samples))
    selected_samples = np.stack(selected_samples, axis=0)
    return selected_samples


def sort_and_deduplicate(data, threshold=1e-5):
    arr = np.array(data)
    sorted_indices = np.lexsort(arr.T)
    sorted_arr = arr[sorted_indices]
    result = [sorted_arr[0]]

    for i in range(1, len(sorted_arr)):
        diff = np.sum(np.abs(sorted_arr[i] - result[-1]))
        if diff > threshold:
            result.append(sorted_arr[i])

    return result


def filter_by_cosine_similarity(data, threshold=0.99):
    if len(data) == 0:
        return []

    data = np.array(data)
    norms = np.linalg.norm(data, axis=1)

    selected_vectors = []
    selected_norms = []

    for i in tqdm(range(len(data))):
        candidate = data[i]
        cand_norm = norms[i]

        if cand_norm == 0:
            continue

        if not selected_vectors:
            selected_vectors.append(candidate)
            selected_norms.append(cand_norm)
            continue

        selected_matrix = np.stack(selected_vectors)
        dot_products = np.dot(selected_matrix, candidate)
        sims = dot_products / (np.array(selected_norms) * cand_norm)

        if np.any(sims > threshold):
            continue

        selected_vectors.append(candidate)
        selected_norms.append(cand_norm)

    return selected_vectors


if __name__ == "__main__":
    prior_dist = np.array([1278, 284, 69, 100, 51, 141, 633])
    train_groups = [
        "general_target",
        "math_very_high",
        "math_high",
        "math_medium",
        "code_very_high",
        "code_high",
        "code_medium",
    ]
    minimum_number = 2e-4
    num_samples = 5_120
    num_selected = 200_000
    enable_bound = True
    temperature = 0.7
    maximum_usage = 4.0

    selected_samples = generate_weights_dirichlet(
        prior_dist,
        train_groups,
        minimum_number,
        num_samples,
        enable_bound,
        temperature,
        maximum_usage,
        num_selected,
    )
    print("original weight: ", prior_dist / np.sum(prior_dist))
    print("average weight: ", np.mean(selected_samples, axis=0))

    data = {
        f"mix_{idx}": {key: round(value, 6) for key, value in zip(train_groups, selected_samples[idx])}
        for idx in range(len(selected_samples))
    }

    output_json_path = "PATH/TO/20w_massive_samples.json"
    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)
