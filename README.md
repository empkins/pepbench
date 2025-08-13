# PEPbench - The python package for automated pre-ejection period (PEP) extraction algorithms.

[![PyPI](https://img.shields.io/pypi/v/pepbench)](https://pypi.org/project/pepbench/)
![GitHub](https://img.shields.io/github/license/empkins/pepbench)
[![Documentation Status](https://readthedocs.org/projects/pepbench/badge/?version=latest)](https://pepbench.readthedocs.io/en/latest/?badge=latest)
[![Test and Lint](https://github.com/empkins/pepbench/actions/workflows/test-and-lint.yml/badge.svg)](https://github.com/empkins/pepbench/actions/workflows/test-and-lint.yml)
[![codecov](https://codecov.io/gh/empkins/pepbench/branch/main/graph/badge.svg?token=IK0QBHQKCO)](https://codecov.io/gh/empkins/pepbench)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pepbench)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/empkins/pepbench)

_pepbench_ presents a framework for the automated extraction of the pre-ejection period (PEP) from
electrocardiogram (ECG) and impedance cardiography (ICG) signals. The package includes a variety of 
algorithms for PEP extraction, as well as tools for the evaluation of these algorithms.


- 💻 3 Q-peak and 10 B-point Detection [Algorithms](https://pepbench.readthedocs.io/en/latest/modules/index.html) from the literature
- 📚 Extensive [documentation](https://pepbench.readthedocs.io/en/latest/)
- 📝 Build to be [easily extendable](https://pepbench.readthedocs.io/en/latest/source/user_guide/create_own_algorithm.html)
- 📁 2 manually annotated [reference datasets](https://pepbench.readthedocs.io/en/latest/source/user_guide/datasets.html) for evaluation  
- 📊 [Evaluation tools](https://pepbench.readthedocs.io/en/latest/source/user_guide/evaluation.html) for PEP extraction algorithms

**Documentation:** [pepbench.readthedocs.io](https://pepbench.readthedocs.io/en/latest/README.html)


## Installation

First install a supported Python version (3.10 or higher) and then install the package using `pip`.

```bash
pip install pepbench
```

### Installing from GitHub

If you want to install the latest version from GitHub, you can use the following command:

```bash
pip install "git+https://github.com/empkins/pepbench.git"
```

If you run into problems, clone the repository and install the package locally.

```bash
git clone https://github.com/empkins/pepbench.git
cd pepbench
pip install .
```

Note: We don't guarantee that the latest version on GitHub is stable.

## Usage Recommendation

`pepbench` is designed to be used in the following ways:

1.  **Usage as a full end-to-end pipeline**:  
    We provide configurable pipelines to extract the PEP per-heartbeat from ECG and ICG signals. The exact 
    configuration of the pipeline (i.e., which algorithm combinations are used) depend on the dataset and 
    can be adjusted to the specific use case. A systematic evaluation of different algorithm combinations
    is subject to the paper "PEPbench – Open, Reproducible, and Systematic Benchmarking of Automated 
    Pre-Ejection Period Extraction Algorithms".

    In this case, we recommend to cite the paper and the package as follows:

    > PEP extraction was performed using the `pepbench` Python library pipeline (version {insert version you used}) 
    > as described in the paper [[1]] with the Q-peak extraction proposed by Martinez et al. [[2]] and the B-point 
    > extraction algorithm proposed by Drost et al. [[3]].

    ```
    [1] <pepbench citation>
    [2] Martinez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based ECG delineator
    evaluation on standard databases. IEEE Transactions on Biomedical Engineering, 51(4), 570-581.
    https://doi.org/10.1109/TBME.2003.821031
    [3] Drost, L., Finke, J. B., Port, J., & Schächinger, H. (2022). Comparison of TWA and PEP as indices of a2- and
    ß-adrenergic activation. Psychopharmacology. https://doi.org/10.1007/s00213-022-06114-8
    ```

2.  **Usage of individual algorithms**:
    If you are only interested in a specific algorithm, you can use the individual algorithms provided in the package. 
    If you are using individual algorithms in this way, we recommend citing the original papers the algorithms were 
    proposed in and `pepbench` as a software library.  You can find the best references for each algorithm in the documentation of the respective algorithm.
    

    > B-points were extracted using the{name of algorithm} algorithm [[1]] as implemented in the `pepbench` Python 
    library [[2]] (version {insert version you used}).
    
    ```
    [1] <algorithm citation>
    [2] <pepbench citation>
    ```



## Contributing

**We want to hear from you (and we want your algorithms)!**

👍 We are always happy to receive feedback and contributions.
If you run into any issues or have any questions, please open an [issue on GitHub](https://github.com/empkins/pepbench/issues)
or start a [discussions](https://github.com/empkins/pepbench/discussions) thread.

📚 If you are using *pepbench* in your research or project, we would love to hear about it and link your work here!

💻 And most importantly, we want your algorithms!
If you have an algorithm that you think would be a good fit for _pepbench_, open an issue, and we can discuss how to integrate it.
We are happy to help you with the integration process.
Even if you are not confident in your Python skills, we can discuss ways to get your algorithm into _pepbench_.


## License

_pepbench_ (and _biopsykit_, which contains the core algorithm implementations) are published under a 
[MIT license](https://opensource.org/license/mit/). This is a permissive license, which allows you to use the code in 
nearly any way you want, as long as you include the original license in you modified version.


## For Developers

Install Python >=3.10 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Then run the commands below to install [poethepoet](https://poethepoet.natn.io), get the latest source,
and install the dependencies:

```bash
git clone https://github.com/empkins/pepbench.git
uv tool install poethepoet
cd pepbench
uv sync --all-extras --dev
```

All dependencies are specified in the main `pyproject.toml` when running `uv sync`.

To run any of the tools required for the development workflow, use the provided 
[poethepoet](https://github.com/nat-n/poethepoet) commands:

```bash
uv run poe
...
CONFIGURED TASKS
  format            Format all files with black.
  lint              Lint all files with ruff.
  check             Check all potential format and linting issues.
  test              Run Pytest with coverage.
  docs              Build the html docs using Sphinx.
  conf_jupyter      Register the pepbench environment as a Jupyter kernel for testing.
  version           Bump version in all relevant places.

```

### Format and Linting

To ensure consistent code structure this project uses black and ruff to automatically check (and fix) the code format.

```
poe format  # runs ruff format and ruff lint with the autofix flag
poe lint # runs ruff without autofix (will show issues that can not automatically be fixed)
```

If you want to check if all code follows the code guidelines, run `poe ci_check`.
This can be useful in the CI context.


### Tests

All tests are located in the `tests` folder and can be executed by using `poe test`.


## Funding and Support

This work was developed within the *Empkins* collaborative research center (SFB 1483) funded by the Deutsche 
Forschungsgemeinschaft (DFG, German Research Foundation) - Project-ID 442419336, EmpkinS.

<p align="center">
<img src="./docs/_static/logo/logo_empkins.svg" height="400">
</p>