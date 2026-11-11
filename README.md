# Extracting Features from Random Subseries: A Hybrid Pipeline for Time Series Classification and Extrinsic Regression

Code and results for the paper "Extracting Features from Random Subseries: A Hybrid
Pipeline for Time Series Classification and Extrinsic Regression" from the
International Workshop on Advanced Analytics and Learning on Temporal Data (AALTD)
2023, which introduces the RIST pipeline.

The paper is available from
[Springer](https://doi.org/10.1007/978-3-031-49896-1_8) and as a
[PDF on the workshop website](https://ecml-aaltd.github.io/aaltd2023/papers/AALTD_2023_Hybrid_Pipeline.pdf).

## Contents

| Path | Description |
|---|---|
| [results/classification](results/classification) | Accuracy for each classifier on the 112 UCR datasets, one column per resample |
| [results/regression](results/regression) | RMSE for each regressor on the 55 regression datasets used in the paper, one column per resample |
| [set_rist_classifier.py](set_rist_classifier.py), [set_rist_regressor.py](set_rist_regressor.py) | The estimator configurations used in the experiments |
| [run_classification_experiments.py](run_classification_experiments.py), [run_regression_experiments.py](run_regression_experiments.py) | Scripts to run a single experiment |
| [rist_pipeline.ipynb](rist_pipeline.ipynb) | Example of using the RIST classifier and regressor directly |
| [static_publication_reqs.txt](static_publication_reqs.txt) | Dependency versions used at the time of publication |

## Datasets

The 112 UCR archive datasets are available at
[timeseriesclassification.com](http://www.timeseriesclassification.com/dataset.php).
The full archive is also on Zenodo as the
[UCR Time Series Classification Archive 2018](https://zenodo.org/records/11198697).

The 63 regression datasets are available from the
[TSER archive expansion webpage](https://tsml-eval.readthedocs.io/en/latest/publications/2023/tser_archive_expansion/tser_archive_expansion.html)
and on Zenodo as the
[TSML Extended Time Series Extrinsic Regression Archive 2024](https://zenodo.org/records/11236865).
The paper uses 55 of these datasets.

## Running the experiments

There are two ways to run the code: with the latest versions, or with the versions
recorded at the time of publication. Results from the latest versions may differ from
the published results files, as the algorithms have changed since publication.

### Latest versions

Clone this repository and install the latest `tsml-eval` with all optional estimator
dependencies:

    pip install -r requirements.txt

This is the same as:

    pip install tsml-eval[all_extras,deep_learning]

Not all estimator dependencies are installed by default with `pip install tsml-eval`.
For a smaller install, you can install these individually as required. RIST requires
`statsmodels`.

Run [run_classification_experiments.py](run_classification_experiments.py) or
[run_regression_experiments.py](run_regression_experiments.py) from the repository
root with the following arguments:

1. Path to the data directory
2. Path to the results directory
3. The name of the model to run (see [set_rist_classifier.py](set_rist_classifier.py)
   or [set_rist_regressor.py](set_rist_regressor.py), i.e. RIST, RDST, DrCIF)
4. The name of the problem to run
5. The resample number to run (0 is base train/test split)

i.e. to run the ItalyPowerDemand classification problem using RIST on the base
train/test split:

    python run_classification_experiments.py data/ results/ RIST ItalyPowerDemand 0

Running either script without arguments runs ROCKET on a small example dataset
included with `tsml-eval` and writes to `test_output/`.

The code in this repository was last checked in November 2026 with `tsml-eval` 0.7.0,
`aeon` 1.5.0 and `tsml` 0.10.0 on Python 3.12. It is not actively maintained against
new releases.

### As published

The dependency versions recorded at the time of publication include `aeon` 0.3.0 and
`tsml` 0.2.1. The code in this repository has since been updated for newer versions,
so to run the code from that time use the `tsml-eval` repository at the last commit
which used these versions. This requires Python 3.8 to 3.11. Results from this setup
are close to the published results files but may not match them exactly.

    git clone https://github.com/time-series-machine-learning/tsml-eval.git
    cd tsml-eval
    git checkout 1efd1b46
    pip install -r tsml_eval/publications/y2023/rist_pipeline/requirements.txt
    pip install --no-deps .
    python tsml_eval/publications/y2023/rist_pipeline/run_classification_experiments.py data/ results/ RIST ItalyPowerDemand 0

The requirements file is the same as
[static_publication_reqs.txt](static_publication_reqs.txt) in this repository. It
was generated on Windows, so `pywin32`, `tensorflow-intel`, `ansicon` and `jinxed`
should be removed on other operating systems.

At the time of publication, RIST required the `pycatch22` and `pyfftw` packages. To
install these, run:

    pip install pycatch22 pyfftw

This can be unstable on some setups. If you cannot install these packages, they can
be disabled in the publication version by editing the classifier parameters (this
will change the results produced) i.e.

    RISTClassifier(use_pycatch22=False, use_pyfftw=False)

Current versions of `aeon` do not use these packages for RIST by default, and the
`use_pyfftw` parameter has been removed.

## Using the classifiers and regressors

The estimators used in the experiments are available in the
[aeon](https://github.com/aeon-toolkit/aeon) and
[tsml](https://github.com/time-series-machine-learning/tsml-py) Python packages and
follow the `scikit-learn` interface. See [rist_pipeline.ipynb](rist_pipeline.ipynb)
for an example, which can be run online using Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/time-series-machine-learning/rist-pipeline-2023/main?filepath=rist_pipeline.ipynb)

## Citation

If you use this work, please cite the paper:

```bibtex
@inproceedings{middlehurst2023extracting,
  title={Extracting Features from Random Subseries: A Hybrid Pipeline for Time Series Classification and Extrinsic Regression},
  author={Middlehurst, Matthew and Bagnall, Anthony},
  booktitle={International Workshop on Advanced Analytics and Learning on Temporal Data},
  series={Lecture Notes in Computer Science},
  volume={14343},
  pages={113--126},
  year={2023},
  publisher={Springer},
  doi={10.1007/978-3-031-49896-1_8}
}
```

## Contact

For questions about the paper or this repository, contact Matthew Middlehurst at
[m.b.middlehurst@bradford.ac.uk](mailto:m.b.middlehurst@bradford.ac.uk) or on the
[aeon Discord](https://discord.gg/D6rzqHGKRJ) (`matthewmiddlehurst`).
