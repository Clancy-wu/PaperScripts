# Extract measurements

# X-axis：TVS-AAN co-fluctuation amplitude
# Y-axis：TVS expression amplitude
## sqrt(mean(similarity(t)^2))

#############################
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
def run(f, this_iter):
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(f, this_iter), total=len(this_iter)))
    return results
#############################
import os
import re
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import zscore, pearsonr

def extract_measure(file):
    data = np.load(file)
    ## basic info
    sub_name = re.findall(r'(sub-.*)_TVS', os.path.basename(file))[0]
    ## X and Y axis
    aan_pc1 = data['aan_pc1']; similar = data['similar_state3']
    tvs_aan_amplitude = np.sqrt(np.mean(np.power(zscore(aan_pc1)*zscore(similar), 2)))
    tvs_express_amplitude = np.sqrt(np.mean(np.power(similar, 2)))

    return sub_name, tvs_aan_amplitude, tvs_express_amplitude

def create_df(dataset_dir):
    dataset_results = run(extract_measure, sorted(glob(f'{dataset_dir}/sub-*_TVS-AAN_TimeData.npz')))
    dataset_df = pd.DataFrame(dataset_results, columns=['subject', 'tvs_aan_amplitude', 'tvs_express_amplitude'])
    return dataset_df

create_df('MultipleDiseases/CFS').to_csv('MultipleDiseases/CFS.csv', index=None)
create_df('MultipleDiseases/SWD').to_csv('MultipleDiseases/SWD.csv', index=None)
create_df('MultipleDiseases/Health').to_csv('MultipleDiseases/Health.csv', index=None)
create_df('MultipleDiseases/SD').to_csv('MultipleDiseases/SD.csv', index=None)

# finished. 