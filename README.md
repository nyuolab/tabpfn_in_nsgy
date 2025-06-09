# TabPFN in Neurosurgery Demo

This repository details the experiments used in our paper to analyze the visualize the performance of TabPFN for neurosurgical outcome prediction with the NSQIP dataset. This work was conducted with PriorLabs-TabPFN.

If using this repository, please cite the TabPFN paper (https://www.nature.com/articles/s41586-024-08328-6), the TabPFN extensions repository (https://github.com/PriorLabs/tabpfn-extensions), and our paper.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

You can create a new conda environment with the required packages by running:

### Using a requirements.txt file

```bash
conda create --name tabpfn_in_nsgy --file requirements.txt
```

Then, activate the environment:

```bash
conda activate tabpfn_in_nsgy
```

### Alternatively, using an environment.yml file

If you prefer to use an environment file, create an `environment.yml` file with your dependencies, then run:

```bash
conda env create -f environment.yml
```

Immediately activate the environment:

```bash
conda activate tabpfn_in_nsgy
```

This will set up the conda environment with all necessary packages for the project.

## Usage

Please follow along with the example given in the tests. We plan to update this repository to make it more user-friendly for non-experts in the near future.

## License

This project is licensed under the PriorLabs TabPFN License. 

## Acknowledgements

This will be updated with the paper's final citation once released.
