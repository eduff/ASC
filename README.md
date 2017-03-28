# Additive Signal Change Analysis tool

## Setup environment
```bash
>> source <dhcpdir>/setup.py
```

## Usage
```bash
usage: asc.py [-h] -d DIR --inds1 INDS1 --inds2 INDS2 [-o OUT_FNAME]
              [--pcorrs PCORRS] [--errdist_perms ERRDIST_PERMS]
              [--min_corr_diff MIN_CORR_DIFF] [--prefix PREFIX] [--pctl PCTL]
              [--subj_order SUBJ_ORDER] [--exclude_conns EXCLUDE_CONNS]
```

## FMRI Additive Signal Analysis

optional arguments:
```bash
  -h, --help            show this help message and exit
```
required arguments:
```bash
  -d DIR, --dir DIR     dual_regression dir
  --inds1 INDS1         index file 1
  --inds2 INDS2         index file 2
```
optional arguments:
```bash
  -o OUT_FNAME, --out_fname OUT_FNAME
                        output file [asc.pdf]
  --pcorrs PCORRS       use partial correlation
  --errdist_perms ERRDIST_PERMS
                        permutations for monte carlo
  --min_corr_diff MIN_CORR_DIFF
                        minimum correlation change
  --prefix PREFIX       dr prefix
  --pctl PCTL           percentile of connections to show (FDR)
  --subj_order SUBJ_ORDER
                        subject_order
  --exclude_conns EXCLUDE_CONNS
                        exclude connections
```


## Basic - Apply to Dual Regression dir
Example:

