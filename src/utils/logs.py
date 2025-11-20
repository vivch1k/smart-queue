import numpy as np
import pandas as pd


def save_logs(path, num_zones, time, load, service, people):
    time = np.array(time)[None, :]
    all_data = np.concatenate((time, load, service, people), axis=0).T
    pref = ["load", "service", "people"]
    columns = ["time"] + [f"{p}_{i+1}" for p in pref for i in range(num_zones)]
    all_data_df = pd.DataFrame(all_data, columns=columns)
    all_data_df.to_csv(path, index=False)
