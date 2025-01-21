# Chemical Data Generation
Repository for doctoral research on machine learning for data generation, with applications in chemical data generation and classification.

<!---
Notes on terminology:
Instead of lab-generated spectra - EXPERIMENTAL spectra
Instead of synthetic spectra - IN-SILICO spectra
Instead of machine - INSTRUMENT
--->
<!---
GitHub resources:
https://www.gitkraken.com/learn/git/git-flow
https://nvie.com/posts/a-successful-git-branching-model/
--->

## Steps to run this repo:
1. Set up virtual environment using ```python3 -m venv your_env_name```. Activate new virtual env with ```.\your_env_name\Scripts\activate``` for Windows or ```source your_env_name/bin/activate``` for macOS/Linux. 

If using micromamba:
  - Create with ```micromamba create -n data_gen_venv python=3.10```.
  - Activate with ```micromamba activate your_env_name```.
2. Install packages using ```pip install -r requirements.txt```. Verify correct installation with ```pip list```.
3. Add virtual environment to ```.gitignore``` as it is unnecessary for GitHub to track.
4. Create new kernel using ```python -m ipykernel install --user --name=your_kernel_name```.
5. If necessary, update git config with username and email using ```git config --global user.name "Your Name"``` and ```git config --global user.email "youremail@example.com"```.

## Notation:
Mathematical notation rules applied throughout this project are taken from [this](https://wookai.github.io/paper-tips-and-tricks/math.html) article.
![Notation rules](images/notation_rules.png)
