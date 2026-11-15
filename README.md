\# Predictive Maintenance ML System



An end-to-end Machine Learning project that predicts potential machine failures using machine sensor data.



\## Project Overview



This project uses the AI4I 2020 Predictive Maintenance dataset to predict machine failures based on operational parameters.



\## Machine Learning Models



\- Logistic Regression

\- Random Forest

\- XGBoost

\- Hyperparameter tuning using RandomizedSearchCV

\- Threshold optimization using F2 score



\## Tech Stack



\- Python

\- Pandas, NumPy

\- Scikit-learn

\- XGBoost

\- Matplotlib

\- FastAPI

\- Pytest



\## Project Structure



```text

predictive-maintenance/

├── api/

├── data/

│   └── raw/

├── models/

├── outputs/

├── src/

│   ├── train.py

│   ├── advanced\_training.py

│   └── eda.py

├── tests/

├── requirements.txt

└── README.md

```



\## Installation



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

cd predictive-maintenance



python -m venv .venv

```



Activate environment on Windows:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Train Models



```bash

python src/train.py

python src/advanced\_training.py

```



\## Run API



```bash

uvicorn api.main:app --reload

```



Open API documentation:



http://127.0.0.1:8000/docs



\## Run Tests



```bash

python -m pytest tests/test\_api.py -v

```



\## Project Goal



Build a reliable predictive maintenance system that can identify potential machine failures before they occur.

