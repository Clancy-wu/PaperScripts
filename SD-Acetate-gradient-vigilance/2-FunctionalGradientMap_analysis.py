import os
import re
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from brainspace.gradient import GradientMaps

# =========================================================
# Parameters
# =========================================================
matrix_dir = "NetFunc"
network_file = "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_desc-400Parcels7Networks_dseg.tsv"
output_dir = "gradient_results"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# Read FC matrices
# =========================================================
files = sorted(glob(os.path.join(matrix_dir, "PearCorZ_sub-*_*.txt")))
records = []
for file in files:
    basename = os.path.basename(file)
    match = re.search(r"PearCorZ_(sub-\d+)_(01|02)\.txt",  basename)
    subject = match.group(1)
    run = match.group(2)
    time = "Before" if run == "01" else "After"
    fc = np.loadtxt(file) # fz value
    np.fill_diagonal(fc, 0)
    records.append({"subject": subject, "time": time, "file": file, "fc": fc })

before_matrices = [x["fc"] for x in records if x["time"] == "Before"]
reference_fc = np.mean(before_matrices, axis=0)
np.fill_diagonal(reference_fc, 0)

# =========================================================
# Choose n_components
# =========================================================
spectrum_model = GradientMaps(n_components=20, approach="dm", kernel="normalized_angle", random_state=0)
spectrum_model.fit(reference_fc, sparsity=0.90)
lambdas = spectrum_model.lambdas_
lambda_df = pd.DataFrame({
    "component": np.arange(1, len(lambdas) + 1),
    "lambda": lambdas
})
lambda_df["relative_lambda"] = ( lambda_df["lambda"] / lambda_df["lambda"].sum())
lambda_df["cumulative_relative_lambda"] = (lambda_df["relative_lambda"].cumsum())
lambda_df.to_csv(os.path.join(output_dir, "reference_gradient_lambda_spectrum.csv"), index=False)

# =========================================================
# Set n_components = 6
# =========================================================
n_components = 6
reference_model = GradientMaps(n_components=n_components, approach="dm", kernel="normalized_angle", random_state=1234)
reference_model.fit(reference_fc, sparsity=0.9)
reference_gradients = reference_model.gradients_
reference_lambdas = reference_model.lambdas_
np.savetxt(os.path.join(output_dir, "reference_gradients.csv"), reference_gradients, delimiter=",")

pd.DataFrame({"gradient": [ f"G{i + 1}" for i in range(n_components) ],
                "eigenvalue": reference_lambdas
}).to_csv( os.path.join( output_dir, "reference_eigenvalues.csv"), index=False)

# =========================================================
# Compute individual gradients and align to reference
# =========================================================
gradient_rows = []
for item in records:
    model = GradientMaps(n_components=n_components, approach="dm", kernel="normalized_angle", alignment="procrustes", random_state=1234 )
    model.fit(item["fc"], sparsity=0.9, reference=reference_gradients)
    aligned_gradient = model.aligned_
    for roi in range(400):
        row = {
            "subject": item["subject"],
            "time": item["time"],
            "roi": roi + 1
        }
        for gradient_index in range(n_components):
            row[f"G{gradient_index + 1}"] = ( aligned_gradient[roi, gradient_index] )
        gradient_rows.append(row)

gradient_df = pd.DataFrame(gradient_rows)
gradient_df.to_csv(os.path.join(output_dir, "individual_aligned_gradients.csv"), index=False)



