# EEG + GSR + PSY Merger and Machine-Learning Interface

## 1. Project overview

This Streamlit application processes raw EEG and GSR CSV files, optionally combines them with PSY (task/label) CSV files, and creates question-level datasets. It also provides an interface to train and compare selected machine-learning models.

The application matches records using **Student_ID** and **QuestionKey**. It supports data for student IDs 1 to 38.

## 2. Software requirements

- Python 3.10 or 3.11 recommended
- pip (Python package installer)
- A web browser
- The project Python file: `app.py`
- The raw EEG and GSR CSV files; PSY CSV files are optional

## 3. Install the required Python packages

Open Command Prompt or the VS Code terminal in the folder containing `app.py`.

First, (optional but recommended) create and activate a virtual environment on Windows:

Install the packages required by the application one by one:

1. Install NumPy:
   ```bash
   python -m pip install numpy
   ```
2. Install Pandas:
   ```bash
   python -m pip install pandas
   ```
3. Install Streamlit (for the web-based UI):
   ```bash
   python -m pip install streamlit
   ```
4. Install SciPy (for GSR peak detection):
   ```bash
   python -m pip install scipy
   ```
5. Install Scikit-learn (for preprocessing, model training and evaluation):
   ```bash
   python -m pip install scikit-learn
   ```
6. Install LightGBM (optional; needed only if you select the LightGBM model):
   ```bash
   python -m pip install lightgbm
   ```
7. Install XGBoost (optional; needed only if you select the XGBoost model):
   ```bash
   python -m pip install xgboost
   ```


## 4. Project folder setup

Keep the application file in one folder. For example:

```text
cognitive-load/
│
├── app.py
└── README.md
```

The CSV data files can be stored elsewhere on your computer because the application lets you upload them through the browser.

## 5. How to open and run the application

### Option A: Using Command Prompt

1. Save `app.py` in a folder, for example `cognitive-load`.
2. Open that folder in File Explorer.
3. Click the folder address bar, type `cmd`, and press Enter. Command Prompt opens in that folder.
4. If you created a virtual environment, activate it:

   ```bash
   .venv\Scripts\activate
   ```

5. Install the required packages by following Section 3 (only needed once in that environment).
6. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

7. Streamlit will display a local address, usually `http://localhost:8501`. Open that address in your web browser if it does not open automatically.
8. Keep the Command Prompt window running while you use the application. To stop it, return to that window and press `Ctrl + C`.

### Option B: Using VS Code

1. Open Visual Studio Code.
2. Select **File → Open Folder** and choose the folder containing `app.py`.
3. Open **Terminal → New Terminal**.
4. Run the installation commands from Section 3 in the terminal.
5. Run:

   ```bash
   python -m streamlit run app.py
   ```

6. Open the local Streamlit address shown in the terminal.

**Important:** Do not run this application by double-clicking `app.py`. It must be started with the Streamlit command above.

## 6. How to use the application interface (UI)

The UI is a browser-based interface created with Streamlit. The Python program runs locally on your computer, and Streamlit displays the controls and results in your browser.

### Step 1 — Choose the input mode

At the top of the page, choose one of the two modes:

- **Single file per modality:** upload one combined/raw CSV file for EEG, one for GSR, and optionally one for PSY.
- **Multiple student files per modality:** upload multiple student CSV files for EEG and GSR, and optionally multiple PSY files.

For multiple files, the application combines the uploaded student files. If a file does not contain a `Student_ID` column, its filename must begin with the student number, for example `1_EEG.csv`, `12-GSR.csv`, or `3_PSY.csv`.

### Step 2 — Upload EEG files

Use the **Raw EEG CSV** uploader (or **Raw EEG CSV files** in multiple-file mode).

The raw EEG data must contain:
- `QuestionKey`
- `TimeStamp` (or an accepted timestamp alias)
- `UnixTime`
- The Delta, Theta, Alpha, Beta and Gamma columns for channels `TP9`, `AF7`, `AF8` and `TP10`

The application calculates question-level EEG features, including band means, band variances, theta/alpha and theta/beta ratios, power spectral entropy, and bandpower slope.

### Step 3 — Upload GSR files

Use the **Raw GSR CSV** uploader.

The GSR data must contain:
- `UnixTime`
- `QuestionKey`
- `Timestamp`
- `GSR Conductance CAL`

The application calculates question-level GSR features such as mean skin conductance level, SCR peak rate, number of SCR peaks, rise amplitude change, and recovery amplitude change.

### Step 4 — Upload PSY files (optional)

Use the **PSY CSV** uploader if you want to include task information or labels. PSY files must have a student identifier (or a student number in the filename) and `QuestionKey`.

For model training, the merged PSY data should include at least one recognized target column, such as `Category`, `Cognitive_Load`, `Difficulty`, or `verdict`. If PSY data is not uploaded, the application can still create the EEG-GSR physiology dataset, but it may not have a target label for supervised model training.

### Step 5 — Process and merge

Click **Process and merge**.

The application:
1. Reads the uploaded CSV files.
2. Cleans column names and standardizes student IDs and question keys.
3. Extracts question-level EEG and GSR features.
4. Matches EEG and GSR records using `Student_ID` and `QuestionKey`.
5. If PSY files are supplied, matches PSY information to the EEG-GSR records using the same keys.
6. Creates a student coverage report and displays the merged output.


### Step 6 — Review and download outputs

After successful processing, the page displays:
- Number of final rows
- Number of students in the final dataset
- A preview of the final dataset
- A student coverage table (inside the **Student coverage** expandable section)

Use the download buttons to save the available CSV outputs:
- `EEG_GSR_PSY_Final.csv` — final merged dataset
- `EEG_Question_Level.csv` — processed EEG features
- `GSR_Question_Level.csv` — processed GSR features
- `EEG_GSR_Merged.csv` — matched EEG-GSR physiology data
- `EEG_Questions_Without_GSR.csv` — EEG records without a matching GSR record
- `GSR_Questions_Without_EEG.csv` — GSR records without a matching EEG record
- `EEG_GSR_Questions_Without_PSY.csv` — EEG-GSR records without a matching PSY record
- `Student_Coverage_Report.csv` — question-level coverage counts for student IDs 1–38

### Step 7 — Train and compare machine-learning models

After the merged dataset is displayed:

1. Under **Compare machine-learning models**, choose a target label from the available target columns.
2. Select one or more models. The default selection is Logistic Regression, SVM and Random Forest.
3. Choose the evaluation split:
   - **By student:** keeps students in the test set separate from students in the training set. This is useful for evaluating performance on students not used for training.
   - **Random rows:** randomly splits records; the same student may appear in both training and test sets.
4. Click **Train and evaluate selected models**.
5. Review the Precision, Recall, F1-score and Accuracy results.
6. Expand **Download test predictions** to download each selected model's actual and predicted labels.

LightGBM and XGBoost are optional packages. Install them as described in Section 3 before selecting those models.

## 7. Important input checks

- Upload CSV files, not Excel files.
- Use matching student IDs and `QuestionKey` values across modalities.
- If `Student_ID` is missing, start the filename with the student number (1–38), followed by an underscore, hyphen or space.
- Each processed modality must have no duplicate `Student_ID` + `QuestionKey` records. The application checks for duplicates and reports an error if it finds them.
- EEG and GSR are required. PSY is optional.
- The final merged dataset may contain fewer rows than the original files because the merge keeps only matching keys.
- Keep a backup of the original raw data.
- Do not upload private or identifiable participant data to a public GitHub repository. Keep raw datasets private and follow the dataset's usage conditions.

## 8. Troubleshooting

**`streamlit` is not recognized**
```bash
python -m pip install streamlit
python -m streamlit run app.py
```

**A package is missing**
Install the package in the same Python environment used to run the app:
```bash
python -m pip install numpy pandas scipy scikit-learn streamlit
```

**LightGBM or XGBoost installation error**
```bash
python -m pip install lightgbm xgboost
```

**No matching student-question rows were found**
Check that the `Student_ID` and `QuestionKey` values match across EEG, GSR and PSY files.

**Student ID cannot be identified**
Add a `Student_ID` column or rename the file using a format such as `1_EEG.csv`.

**Application does not open**
Check the terminal for an error message, confirm that the packages are installed in the active environment, and try:
```bash
python -m streamlit run app.py
```

## 9. Main Python libraries and their purpose

| Library | Purpose in this application |
|---|---|
| NumPy | Numerical calculations and handling invalid numeric values |
| Pandas | Reading, cleaning, grouping, merging and exporting CSV data |
| Streamlit | Building the browser-based user interface |
| SciPy | Detecting peaks in the GSR signal |
| Scikit-learn | Preprocessing, model training, data splitting and evaluation |
| LightGBM (optional) | LightGBM classification model |
| XGBoost (optional) | XGBoost classification model |

## 10. Summary

This application provides one interface for uploading raw EEG, GSR and optional PSY CSV files, generating question-level features, merging the modalities, checking student coverage, downloading processed datasets, and comparing selected machine-learning classifiers. The user operates the application through a web browser, while processing and model training run in the local Python environment.
