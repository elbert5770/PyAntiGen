import pandas as pd
import os


def load_data(data_name, data_path):
    # Load experimental data from data folder
    data_path = os.path.join(data_path, data_name)
    df = pd.read_csv(data_path)  # utf-8-sig strips BOM if present
    return df



def load_experiment1_data(data_path):
    experiment_path = os.path.join(data_path, 'Example_experiment1.csv')
    df = pd.read_csv(experiment_path)
    return df

def load_experiment2_data(data_path):
    experiment_path = os.path.join(data_path, 'Example_experiment2.csv')
    df = pd.read_csv(experiment_path)
    return df


# def generate_sampled_data(result, data_path):
    
#     time_points = np.asarray(result['time'])
#     b_comp1 = np.asarray(result['[B_Comp1]'])
#     rng = np.random.default_rng()
#     noise_scale = 0.25  # 3% relative noise
#     noisy_b = b_comp1 + rng.normal(0, noise_scale, size=b_comp1.shape)
    
#     noisy_b = abs(noisy_b)
    
#     # Sample every 2 time units (use point closest to each 0, 2, 4, ...)
#     sample_times = np.arange(0, time_points.max() + 1e-9, 4)
#     indices = np.argmin(np.abs(time_points - sample_times[:, np.newaxis]), axis=1)
#     sampled_time = time_points[indices]
#     sampled_b = noisy_b[indices]
#     experiment_path = os.path.join(data_path, 'Example_experiment2.csv')
#     pd.DataFrame({'time': sampled_time, 'B': sampled_b}).to_csv(experiment_path, index=False)
#     return sampled_time, sampled_b
