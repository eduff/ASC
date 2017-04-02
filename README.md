# Additive Signal Change Analysis tool

*Version 0.2.0* --- 2017-03-30

[Repository][REPOSITORY] --- [Changelog][CHANGELOG] --- [License][LICENSE]

----------

Eugene Duff

[FMRIB Centre][FMRIB], University of Oxford

[Please contact me][myWebsite] if you are interested in running ASC.

## Setup environment
```bash
>> source <ascdir>/setup.py
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

### Dependencies ###

*ASC * uses [Python 3](https://www.python.org) and [FSL][]

Python dependencies are the following, to be included in [FSLpy]

##### Python #####
 + [NumPy](http://www.numpy.org): The fundamental package for scientific computing with Python.
 + [SciPy](http://www.scipy.org): A Python-based ecosystem of open-source software for mathematics, science, and engineering.
 + [matplotlib](http://matplotlib.org): A python 2D plotting library.
 + [NiBabel](http://nipy.org/nibabel/): Provides read / write access to some common neuroimaging file formats.
 + [Spectrum](https://github.com/cokelaer/spectrum) Spectrum Analysis Tools.
 + [MNE](http://martinos.org/mne/): Temporary visualisation tools.

----------

[REPOSITORY]: https://git.fmrib.ox.ac.uk/eduff/ampconn "ASC Repository
[CHANGELOG]: https://git.fmrib.ox.ac.uk/eduff/ampconn/blob/release/CHANGELOG.md "ASC Changelog"
[LICENSE]: https://git.fmrib.ox.ac.uk/eduff/ampconn/blob/release/LICENSE "ASC License file"

[myWebsite]: http://www.ndcn.ox.ac.uk/team/eugene-duff  "FMRIB Profile of Eugene Duff"
[FMRIB]: http://www.ndcn.ox.ac.uk/divisions/fmrib/ "FMRIB Website"
[FSL]: http://fsl.fmrib.ox.ac.uk "FSL Wiki"

----------

Written in [Markdown][] using [Dillinger][].


