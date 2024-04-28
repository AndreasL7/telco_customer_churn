# DSA_ML Project Template
**Tidy up and structure your data science project workspace!**

## Description
Create project directories, commonly files and setup, virtual environment, and docker images, all by simply calling the `make` command

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Development Commands](#development-commands)
- [Contributing](#contributing)
- [Directory Structure](#directory-structure)
- [License](#license)

## Installation

1. **Clone the Repository**

   ```
   git clone <repository-url>
   ```

2. **For Windows Users**

   a. Download and install Git. Follow the instruction [here](https://www.educative.io/answers/how-to-install-git-bash-in-windows "Educative.io")

   b. Run Powershell terminal and run the command below (Scoop command-line installer)
    - `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
    - `Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression`
    - `[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12`
    - `iwr -useb get.scoop.sh -outfile 'install.ps1'`
    - `.\install.ps1 -RunAsAdmin`
    - `scoop install make`
   
   c. Open Git BASH

3. **Setup Project Directories**

   On Mac/Linux terminal or Windows Git BASH,
   ```bash
   cd <repository-name>
   make setup-directories
   ```

4. **Setup Common Files**
   
   ```bash
   make setup_initfiles
   ```

5. **Setup the Virtual Environment**

   a. Before setting up the virtual environment, ensure you have `python3` or `conda` installed on your machine.
   
   b. Ensure you can run `python3 --version` or `conda --version` on your Linux/Mac terminal or Windows Git BASH.
   
   c. If command is not found, ensure you look for your Python installation folder, copy the path directory and update your PATH environment variable. Restart terminal or Git BASH accordingly.

   ```bash
   make setup_venv
   ```

   This will create a new virtual environment using conda (if installed on local machine) with all necessary packages listed in `environment.yml` file. If conda is not found, it will default to Python venv and install all necessary packages listed in `requirements.txt`.

6. **Docker (Optional)**

   If you intend to run the application inside a Docker container, ensure you have Docker installed and running.

## Usage

- **Running the App Locally**

   ```bash
   make run
   ```

- **Working with Docker**

  - Build the Docker image:

    ```bash
    make docker_build
    ```

  - Deploy the Docker container:

    ```bash
    make docker_deploy
    ```

  - To stop and remove the Docker container:

    ```bash
    make docker_stop
    ```

## Development Commands

Here are some of the main `Makefile` commands you might use during development:

- `make help`: Display a list of available commands.
- `make setup_directories`: Set up the directory structure for the project.
- `make setup_initfiles`: Set up the project initial files.
- `make setup_venv`: Set up the Python virtual environment and install dependencies.
- `make test`: Run tests using pytest.
- `make docker_build`: Build a Docker image for the project.
- `make docker_deploy`: Deploy the application using Docker.
- `make ci`: Run a full CI/CD sequence, which includes tests, Docker image building, and Docker deployment.
- `make docker_stop`: Stop and remove the Docker container.
- `make clean`: Clean up the development environment (remove virtual environment and Docker image).
- `make setup_readme`: Set up a README.md file

## Directory Structure

```
proj/
│
├── data/                  # Data directory for storing all project data
│   ├── raw/               # Raw data, unmodified from its original state
│   ├── processed/         # Data that has been processed and ready for analysis
│   ├── external/          # External data, like third-party datasets or exports
│   ├── final/             # Final stage data after all processing have been completed
│   ├── interim/           # Temporary data
│   ├── static/            # Additional static unmodified data
│
├── notebooks/             # Jupyter notebooks directory
│   ├── exploratory/       # Notebooks for initial data exploration and experimentation
│       └── nb_descriptive.ipynb/       # Notebooks for descriptive analysis
│   └── report/            # Final notebooks used for reporting and presentation
│
├── .venv/                 # Python virtual environment (not committed to version control)
|
├── utils/                 # Utility folder
│   ├── helpers.py/        # Python file containing helper methods
|
├── config/                # Configuration files folder
│   ├── config.yaml/       # Configuration files
|
├── logfiles/              # Log files
|
├── models/                # ML models
|
├── src/                   # Source code directory
│   ├── app.py/            # Main python file
│
├── tests/                 # Tests directory for unit tests, integration tests, etc.
│
├── requirements.txt       # Python dependencies
│
├── .github/workflows/     # GitHub Actions workflows
│
├── .gitignore             # Specifies intentionally untracked files to ignore by Git
│
├── README.md              # Project description, usage, and other details
│
└── Makefile               # Contains automation commands for the project setup and management
```

Directory Descriptions:

- **data/**: This is the primary directory where all the data related to the project resides. Data is further categorized into raw, processed, and external to maintain clarity and separation of concerns.
  
- **notebooks/**: Contains Jupyter notebooks used throughout the project. `exploratory/` contains initial data exploration and experimentation notebooks, while `report/` has the finalized notebooks for presentation and reporting purposes.

- **utils/**: The source code of the Helpers file reside here. This includes all the reusable methods defined separately from the main file.

- **config/**: The configuration files of the project reside here.

- **logfiles/**: The logging files of the project reside here.

- **models/**: The machine learning models generated reside here.
  
- **src/**: The source code of the project resides here. This might include modules, scripts, and other necessary code files.

- **tests/**: This directory is dedicated to testing. It contains test scripts, fixtures, and other testing-related files to ensure the codebase's functionality and robustness.

- **Makefile**: This is a simple way to manage project tasks. It provides a set of commands for setting up the environment, running tests, building Docker images, and more. It also serves as a form of documentation for the project. If you are running on Windows, you can refer to the `Makefile` commands and run the equivalent commands in the command prompt.

---

## Alternative Layouts

- [Eric Ma's Layout](https://gist.github.com/ericmjl/27e50331f24db3e8f957d1fe7bbbe510)
- [Matt Harrison's Layout](https://github.com/mattharrison/sample_nb_code_project/tree/main)
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

---

This template is inspired by Matt Harrison's work with some modification to my personal workflow preference.