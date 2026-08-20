import pandas as pd
from nilearn import image
import numpy as np
from nimare.extract import fetch_neurosynth
from nimare.decode.continuous import CorrelationDecoder
from nimare.meta.cbma import MKDAChi2

# =========================
# 1. Fetch NeuroSynth dataset
# =========================
all_files = fetch_neurosynth(data_dir='/home/clancy/.nimare/neurosynth', return_type='studyset', target='mni152_2mm', overwrite=False)
dset = all_files[0].to_dataset()

# =========================
# 2. Define decoder
# =========================
decoder = CorrelationDecoder(
    frequency_threshold=0.001,
    meta_estimator=MKDAChi2,
    target_image="z_desc-association",
    n_cores=4
)

# =========================
# 3. Fit decoder
# =========================
decoder.fit(dset)
# =========================
# 4. Decode your brain map
# =========================
decoded_df = decoder.transform("TVS_246roi_brain_map.nii.gz")

# =========================
# 5. Save results
# =========================

decoded_df = decoded_df.sort_values("r", ascending=False)
decoded_df.to_csv(
    "TVS_nimare_neurosynth_decoding_results.csv",
    index=True
)


arousal_attention_keywords = [
    # core terms
    "arousal",
    "alertness",
    "vigilance",
    "attention",
    "attentional",
    
    # attention subdomains
    "sustained attention",
    "selective attention",
    "visual attention",
    "spatial attention",
    "orienting",
    "orienting attention",
    "executive attention",
    "attentional control",
    
    # control and task state
    "cognitive control",
    "executive control",
    "inhibition",
    "response inhibition",
    "conflict",
    "task switching",
    
    # wakefulness / sleep-related arousal
    "wakefulness",
    "wake",
    "sleep",
    "sleep deprivation",
    "fatigue",
    "drowsiness",
    
    # salience / alerting network
    "salience",
    "salient",
    "alerting",
    
    # sensory-attentional systems
    "visual",
    "visuospatial",
    "sensorimotor",
    
    # autonomic/interoceptive arousal
    "autonomic",
    "interoception",
    "interoceptive",
    "heart rate",
]

def filter_features_by_keywords(columns, keywords):
    matched = []
    
    for col in columns:
        col_lower = col.lower()
        
        for kw in keywords:
            if kw.lower() in col_lower:
                matched.append(col)
                break
    
    matched = sorted(set(matched))
    return matched


cols = dset.annotations.columns.tolist()

matched_features = filter_features_by_keywords(
    columns=cols,
    keywords=arousal_attention_keywords
)

print("Number of matched features:", len(matched_features))

for f in matched_features:
    print(f)

decoded_df = decoded_df.reset_index()
filtered_df = decoded_df[decoded_df['feature'].isin(matched_features)]
filtered_df.to_csv('arousal_attention_selected_features.csv', index=None)

from scipy.stats import pearsonr
neurotransmitter = pd.read_csv('atlas/annotations_mni152_parc_246_BN.csv')
neurotransmitter_result = neurotransmitter[['source', 'desc']].copy()
tvs = pd.read_csv("all_dynamic_brain_states/state_3_mean.txt", header=None)

for i in np.arange(neurotransmitter.shape[0]):
    i_cor = pearsonr( np.float16(neurotransmitter.iloc[i, 2:249].values), 
                                                   np.float16(tvs[0].values) )
    neurotransmitter_result.loc[i, 'rValue'] = i_cor.statistic
    neurotransmitter_result.loc[i, 'pValue'] = i_cor.pvalue

neuro_annotation = pd.read_csv('atlas/Neurotransmitters_content(OnlyNeural_43).csv')
df_final = pd.merge(neurotransmitter_result, neuro_annotation, how='left', on=['source', 'desc'])
df_final = df_final.sort_values('rValue')
df_final.to_csv('TVS_neurotransmitter_annotation.csv', index=None)
