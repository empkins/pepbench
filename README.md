# PEPbench - The python package for automated pre-ejection period (PEP) extraction algorithms.

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

_pepbench_ can easily be installed via pip:

```bash
pip install pepbench
```

### Installing from GitHub

If you want to install the latest version from GitHub, you can use the following command:

```bash
pip install "git+https://github.com/empkins/pepbench.git"
```

Note: We don't guarantee that the latest version on GitHub is stable.


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
Then run the commands below to install [poethepoet](`https://poethepoet.natn.io`), get the latest source,
and install the dependencies:

```bash
git clone https://github.com/empkins/pepbench.git
uv tool install poethepoet
uv sync --all-extras --dev
```

All dependencies are specified in the main `pyproject.toml` when running `uv sync`.

To run any of the tools required for the development workflow, use the poe commands:

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