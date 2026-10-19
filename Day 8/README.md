# Rainfall EDA Project

A Python project covering 40 Exploratory Data Analysis (EDA) tasks for a rainfall dataset.

## Project files

- `rainfall_eda.py` — code for Q1–Q40
- `rainfall.csv` — your dataset (add this file before running)
- `requirements.txt` — required Python libraries
- `outputs/` — charts and CSV reports generated when the script runs

## Requirements

- Python 3.10 or newer recommended
- Visual Studio Code
- Python extension for VS Code

## Setup in VS Code

1. Open this folder in VS Code.
2. Open the terminal: **Terminal → New Terminal**.
3. Create a virtual environment:

   **Windows**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt and run:
   ```bat
   .venv\Scripts\activate.bat
   ```

4. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Put your dataset in this folder and name it `rainfall.csv`.
6. Run:
   ```bash
   python rainfall_eda.py
   ```

The script looks for the CSV beside the Python file. It saves charts and reports in `outputs/`.

## Expected column names

The script automatically recognizes common variants of:
- rainfall: `rainfall`, `rain`, `precipitation`
- temperature: `temperature`, `temp`
- humidity: `humidity`, `relative_humidity`
- weather condition: `weather_condition`, `weather`, `condition`

Temperature, humidity, and weather-condition questions are skipped if those columns are absent. If the rainfall column has another name, add it to the alias list in `rainfall_eda.py`.

## GitHub

Do not upload your `.venv` folder. If your dataset is private or its license does not permit redistribution, do not upload `rainfall.csv`; instead, explain in this README how to obtain it.
