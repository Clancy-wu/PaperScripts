#!/bin/bash

## fMRI preprocessing with fMRIPrep
docker run -ti --rm \
    -v /home/clancy/ssd/SleepDisfunction:/work \
    -u $(id -u):$(id -g) \
    -v /home/clancy/TemplateFlow:/opt/templateflow \
    -e TEMPLATEFLOW_HOME=/opt/templateflow \
    nipreps/fmriprep:24.1.1 \
    /work/BIDS/ /work/fmriprep/ participant --participant-label PRE \
	--skip_bids_validation \
	--ignore fieldmaps \
	-w /work/fmriprep_work \
	--nthreads 1 --omp-nthreads 1 \
	--output-spaces MNI152NLin2009cAsym:res-2 \
	--bold2t1w-dof 12 --force-bbr \
	--skull-strip-t1w force \
	--fs-license-file /work/license.txt \
	--output-layout bids \
	--cifti-output 91k \
	--resource-monitor \
	--notrack \
	--stop-on-first-crash

## fMRI postpocessing with xcpd
docker run -ti --rm \
    -v /home/clancy/ssd/SleepDisfunction:/work \
    -u $(id -u):$(id -g) \
    -v /home/clancy/TemplateFlow:/opt/templateflow \
    -e TEMPLATEFLOW_HOME=/opt/templateflow \
	pennlinc/xcp_d:0.10.0 \
	/work/fmriprep/ /work/xcpd_24wmcfs03/ participant --participant-label PRE \
	--mode linc \
	--nthreads 2 --omp-nthreads 2 --mem-gb 10 \
	--task-id rest \
	--input-type fmriprep \
	--file-format nifti \
	--output-type censored \
	--dummy-scans auto \
	--despike y \
	-p /work/24wmcfs03P.yml \
	--smoothing 6 \
	--combine_runs n \
	--lower-bpf 0.01 \
	--upper-bpf 0.10 \
	--fd-thresh 0.3 \
	--min-time 0 \
	--linc_qc n \
	--skip-parcellation \
	-w /work/xcpd_work \
	--fs-license-file /work/license.txt \
	--resource-monitor \
	--stop-on-first-crash 
	


