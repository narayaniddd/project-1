import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


# ================================================================
# SETTINGS
# ================================================================

EXPECTED_STUDENT_IDS = set(range(1, 39))

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

BAND_COLUMNS = {
    band: [f"{band}_{channel}" for channel in EEG_CHANNELS]
    for band in BANDS
}

EEG_SOURCE_COLUMNS = [
    column
    for columns in BAND_COLUMNS.values()
    for column in columns
]

EEG_FEATURES = (
    [f"{band}_Mean" for band in BANDS]
    + [f"{band}_Variance" for band in BANDS]
    + [
        "Theta_Alpha_Ratio",
        "Theta_Beta_Ratio",
        "Power_Spectral_Entropy",
        "Bandpower_Slope",
    ]
)

GSR_REQUIRED_COLUMNS = [
    "UnixTime",
    "QuestionKey",
    "Timestamp",
    "GSR Conductance CAL",
]

KEYS = ["Student_ID", "QuestionKey"]

# Random Forest remains. Soft-voting Ensemble is removed.
MODEL_OPTIONS = [
    "Logistic Regression",
    "SVM",
    "Random Forest",
    "LightGBM",
    "XGBoost",
]


# ================================================================
# COMMON HELPERS
# ================================================================

def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def normalize_student_id(series):
    values = pd.to_numeric(series, errors="coerce")
    values = values.where(np.isfinite(values), np.nan)
    return values.round().astype("Int64")


def normalize_question_key(series):
    result = series.astype("string").str.strip()
    return result.replace("", pd.NA)


def extract_student_id(filename):
    """Extract IDs from filenames such as 1_EEG.csv or 12-GSR.csv."""
    if not filename:
        return None

    stem = Path(filename).stem
    match = re.match(r"^\s*(\d+)(?:[_\-\s]|$)", stem)

    if match:
        student_id = int(match.group(1))
        if student_id in EXPECTED_STUDENT_IDS:
            return student_id

    return None


def prepare_keys(df, source_name, filename=None):
    """Standardize student IDs and question keys."""
    df = clean_columns(df)

    student_aliases = ["StudentID", "Student ID", "student_id"]
    question_aliases = ["Key", "key", "Question Key", "question_key"]

    for alias in student_aliases:
        if "Student_ID" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "Student_ID"})

    for alias in question_aliases:
        if "QuestionKey" not in df.columns and alias in df.columns:
            df = df.rename(columns={alias: "QuestionKey"})

    file_student_id = extract_student_id(filename)

    if "Student_ID" not in df.columns:
        if file_student_id is None:
            raise ValueError(
                f"{source_name}: no Student_ID column and no ID in filename. "
                "Use a filename like 1_EEG.csv or 12-GSR.csv."
            )
        df["Student_ID"] = file_student_id
    else:
        df["Student_ID"] = normalize_student_id(df["Student_ID"])

        if file_student_id is not None:
            existing_ids = set(df["Student_ID"].dropna().astype(int))

            if existing_ids and existing_ids != {file_student_id}:
                raise ValueError(
                    f"{source_name}: filename ID ({file_student_id}) does "
                    "not match the Student_ID column."
                )

            df["Student_ID"] = df["Student_ID"].fillna(file_student_id)

    if "QuestionKey" not in df.columns:
        raise ValueError(
            f"{source_name}: QuestionKey is missing. Accepted alternatives: "
            "Key or Question Key."
        )

    df["Student_ID"] = normalize_student_id(df["Student_ID"])
    df["QuestionKey"] = normalize_question_key(df["QuestionKey"])
    df = df.dropna(subset=KEYS).copy()

    if df.empty:
        raise ValueError(f"{source_name}: no usable key rows.")

    df["Student_ID"] = df["Student_ID"].astype("int64")

    invalid_ids = sorted(set(df["Student_ID"]) - EXPECTED_STUDENT_IDS)
    if invalid_ids:
        raise ValueError(
            f"{source_name}: student IDs must be between 1 and 38. "
            f"Invalid IDs: {invalid_ids}"
        )

    return df


def validate_unique_keys(df, name):
    duplicate_mask = df.duplicated(KEYS, keep=False)

    if duplicate_mask.any():
        examples = df.loc[duplicate_mask, KEYS].head(10)
        raise ValueError(
            f"{name} contains duplicate Student_ID + QuestionKey records. "
            f"Examples:\n{examples.to_string(index=False)}"
        )


def read_uploaded_csv(uploaded_file):
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, low_memory=False)


# ================================================================
# RAW EEG PROCESSING
# ================================================================

def derive_eeg_sample_features(chunk):
    chunk = chunk.copy()

    for column in EEG_SOURCE_COLUMNS:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk[column] = chunk[column].replace([np.inf, -np.inf], np.nan)

    band_values = pd.DataFrame(
        {
            band: chunk[columns].mean(axis=1)
            for band, columns in BAND_COLUMNS.items()
        },
        index=chunk.index,
    )

    for band in BANDS:
        chunk[f"{band}_Mean"] = band_values[band]
        chunk[f"{band}_Variance"] = chunk[BAND_COLUMNS[band]].var(
            axis=1,
            ddof=1,
        )

    epsilon = 1e-12

    chunk["Theta_Alpha_Ratio"] = (
        band_values["Theta"]
        / band_values["Alpha"].abs().clip(lower=epsilon)
    )

    chunk["Theta_Beta_Ratio"] = (
        band_values["Theta"]
        / band_values["Beta"].abs().clip(lower=epsilon)
    )

    positive_power = band_values.clip(lower=0)
    total_power = positive_power.sum(axis=1).replace(0, np.nan)
    probabilities = positive_power.div(total_power, axis=0)

    chunk["Power_Spectral_Entropy"] = -(
        probabilities
        * np.log(probabilities.clip(lower=epsilon))
    ).sum(axis=1) / np.log(len(BANDS))

    x = np.arange(len(BANDS), dtype=float)
    values = band_values.to_numpy(dtype=float)
    slopes = np.full(len(chunk), np.nan)

    for i, row in enumerate(values):
        if np.isfinite(row).all():
            slopes[i] = np.polyfit(x, row, 1)[0]

    chunk["Bandpower_Slope"] = slopes

    for column in EEG_FEATURES:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk[column] = chunk[column].replace([np.inf, -np.inf], np.nan)

    return chunk


def process_eeg_dataframe(df, source_name, filename=None):
    """Aggregate raw EEG into one row per student and question."""
    df = clean_columns(df)
    file_student_id = extract_student_id(filename)

    if "Student_ID" not in df.columns:
        if file_student_id is None:
            raise ValueError(
                f"{source_name}: add Student_ID or start the filename "
                "with the student number, e.g. 1_EEG.csv."
            )
        df["Student_ID"] = file_student_id
    else:
        df["Student_ID"] = normalize_student_id(df["Student_ID"])

        if file_student_id is not None:
            existing_ids = set(df["Student_ID"].dropna().astype(int))

            if existing_ids and existing_ids != {file_student_id}:
                raise ValueError(
                    f"{source_name}: filename ID does not match "
                    "the Student_ID column."
                )

            df["Student_ID"] = df["Student_ID"].fillna(file_student_id)

    if "TimeStamp" not in df.columns:
        for alias in ["Timestamp", "timestamp", "datetime"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "TimeStamp"})
                break

    if "QuestionKey" not in df.columns:
        for alias in ["Key", "key", "Question Key", "question_key"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "QuestionKey"})
                break

    required = [
        "QuestionKey",
        "TimeStamp",
        "UnixTime",
        *EEG_SOURCE_COLUMNS,
    ]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{source_name}: raw EEG file is missing columns: {missing}"
        )

    df["Student_ID"] = normalize_student_id(df["Student_ID"])
    df["QuestionKey"] = normalize_question_key(df["QuestionKey"])
    df["TimeStamp"] = pd.to_datetime(
        df["TimeStamp"], errors="coerce", utc=True
    )
    df["UnixTime"] = pd.to_numeric(df["UnixTime"], errors="coerce")

    df = df.dropna(
        subset=["Student_ID", "QuestionKey", "TimeStamp"]
    ).copy()

    if df.empty:
        raise ValueError(f"{source_name}: no valid EEG samples found.")

    df["Student_ID"] = df["Student_ID"].astype("int64")

    invalid_ids = sorted(set(df["Student_ID"]) - EXPECTED_STUDENT_IDS)
    if invalid_ids:
        raise ValueError(
            f"{source_name}: student IDs outside 1–38: {invalid_ids}"
        )

    df = derive_eeg_sample_features(df)

    aggregate = {
        "EEG_Sample_Count": ("Student_ID", "size"),
        "EEG_Start_Time": ("TimeStamp", "min"),
        "EEG_End_Time": ("TimeStamp", "max"),
        "EEG_UnixTime_Start": ("UnixTime", "min"),
        "EEG_UnixTime_End": ("UnixTime", "max"),
    }

    for column in EEG_FEATURES:
        aggregate[column] = (column, "mean")

    result = df.groupby(KEYS, as_index=False, sort=False).agg(**aggregate)
    validate_unique_keys(result, f"Processed EEG: {source_name}")
    return result


def process_eeg_files(files):
    results = []

    for uploaded_file in files:
        with st.spinner(f"Processing EEG: {uploaded_file.name}"):
            raw = read_uploaded_csv(uploaded_file)
            results.append(
                process_eeg_dataframe(
                    raw,
                    uploaded_file.name,
                    uploaded_file.name,
                )
            )

    if not results:
        raise ValueError("Upload at least one raw EEG CSV file.")

    eeg = pd.concat(results, ignore_index=True)
    validate_unique_keys(eeg, "Combined EEG data")
    return eeg


# ================================================================
# RAW GSR PROCESSING
# ================================================================

def calculate_gsr_features(group):
    group = group.sort_values("Timestamp").copy()

    signal = pd.to_numeric(
        group["GSR Conductance CAL"],
        errors="coerce",
    ).to_numpy(dtype=float)

    timestamps = pd.to_datetime(
        group["Timestamp"], errors="coerce", utc=True
    )

    valid = np.isfinite(signal) & timestamps.notna().to_numpy()
    signal = signal[valid]
    timestamps = timestamps[valid]

    if len(signal) == 0:
        return None

    duration_seconds = (
        timestamps.max() - timestamps.min()
    ).total_seconds()

    if duration_seconds <= 0:
        unix_values = pd.to_numeric(
            group["UnixTime"], errors="coerce"
        ).to_numpy(dtype=float)
        unix_values = unix_values[np.isfinite(unix_values)]

        if len(unix_values) >= 2:
            duration_seconds = unix_values.max() - unix_values.min()

    duration_minutes = (
        duration_seconds / 60
        if duration_seconds > 0
        else np.nan
    )

    peaks, _ = find_peaks(signal, prominence=0.01, distance=10)

    rise_changes = [
        signal[p] - signal[p - 1]
        for p in peaks
        if p > 0
    ]

    recovery_changes = [
        signal[p] - signal[p + 1]
        for p in peaks
        if p < len(signal) - 1
    ]

    return {
        "Mean_SCL": float(np.mean(signal)),
        "SCR_Peaks_Per_Minute": (
            len(peaks) / duration_minutes
            if len(peaks) > 0
            and np.isfinite(duration_minutes)
            and duration_minutes > 0
            else np.nan
        ),
        "Number_of_SCRs": int(len(peaks)),
        "Rise_Amplitude_Change": (
            float(np.mean(rise_changes)) if rise_changes else np.nan
        ),
        "Recovery_Amplitude_Change": (
            float(np.mean(recovery_changes))
            if recovery_changes
            else np.nan
        ),
    }


def process_gsr_dataframe(df, source_name, filename=None):
    df = clean_columns(df)

    if "Timestamp" not in df.columns:
        for alias in ["TimeStamp", "timestamp", "datetime"]:
            if alias in df.columns:
                df = df.rename(columns={alias: "Timestamp"})
                break

    df = prepare_keys(df, source_name, filename)

    missing = [
        column
        for column in GSR_REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{source_name}: GSR file is missing columns: {missing}"
        )

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], errors="coerce", utc=True
    )
    df["UnixTime"] = pd.to_numeric(df["UnixTime"], errors="coerce")
    df["GSR Conductance CAL"] = pd.to_numeric(
        df["GSR Conductance CAL"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=["QuestionKey", "Timestamp", "GSR Conductance CAL"]
    ).copy()

    if df.empty:
        raise ValueError(f"{source_name}: no valid GSR samples found.")

    rows = []

    for (student_id, question_key), group in df.groupby(KEYS, sort=False):
        features = calculate_gsr_features(group)

        if features is not None:
            rows.append({
                "Student_ID": int(student_id),
                "QuestionKey": question_key,
                **features,
            })

    result = pd.DataFrame(rows)

    if not result.empty:
        validate_unique_keys(result, f"Processed GSR: {source_name}")

    return result


def process_gsr_files(files):
    results = []

    for uploaded_file in files:
        with st.spinner(f"Processing GSR: {uploaded_file.name}"):
            raw = read_uploaded_csv(uploaded_file)
            results.append(
                process_gsr_dataframe(
                    raw,
                    uploaded_file.name,
                    uploaded_file.name,
                )
            )

    if not results:
        raise ValueError("Upload at least one raw GSR CSV file.")

    gsr = pd.concat(results, ignore_index=True)

    if not gsr.empty:
        validate_unique_keys(gsr, "Combined GSR data")

    return gsr


# ================================================================
# OPTIONAL PSY PROCESSING
# ================================================================

def process_psy_files(files):
    if not files:
        return None

    frames = []

    for uploaded_file in files:
        raw = read_uploaded_csv(uploaded_file)
        frames.append(
            prepare_keys(
                raw,
                uploaded_file.name,
                uploaded_file.name,
            )
        )

    psy = pd.concat(frames, ignore_index=True, sort=False)
    psy = psy.drop_duplicates(subset=KEYS, keep="first").copy()
    validate_unique_keys(psy, "Combined PSY data")

    recognized_columns = [
        "Category",
        "Cognitive_Load",
        "Difficulty",
        "verdict",
        "ResponseTime",
    ]

    if not any(column in psy.columns for column in recognized_columns):
        raise ValueError(
            "No recognized target column found in PSY files. Expected "
            "Category, Cognitive_Load, Difficulty, verdict, or ResponseTime."
        )

    return psy


# ================================================================
# MERGE AND REPORTS
# ================================================================

def merge_data(eeg, gsr, psy=None):
    validate_unique_keys(eeg, "EEG data")

    if not gsr.empty:
        validate_unique_keys(gsr, "GSR data")

    eeg_only = (
        eeg[KEYS]
        .merge(gsr[KEYS], on=KEYS, how="left", indicator=True)
        .query("_merge == 'left_only'")[KEYS]
        .reset_index(drop=True)
    )

    gsr_only = (
        gsr[KEYS]
        .merge(eeg[KEYS], on=KEYS, how="left", indicator=True)
        .query("_merge == 'left_only'")[KEYS]
        .reset_index(drop=True)
    )

    physiology = eeg.merge(
        gsr,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    physiology_without_psy = pd.DataFrame(columns=KEYS)
    final = physiology.copy()

    if psy is not None:
        collisions = [
            column
            for column in psy.columns
            if column not in KEYS and column in final.columns
        ]

        psy = psy.rename(
            columns={column: f"PSY_{column}" for column in collisions}
        )

        physiology_without_psy = (
            physiology[KEYS]
            .merge(psy[KEYS], on=KEYS, how="left", indicator=True)
            .query("_merge == 'left_only'")[KEYS]
            .reset_index(drop=True)
        )

        final = physiology.merge(
            psy,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )

    validate_unique_keys(final, "Final merged data")

    return (
        final,
        physiology,
        eeg_only,
        gsr_only,
        physiology_without_psy,
    )


def make_coverage(eeg, gsr, physiology, final):
    coverage = pd.DataFrame(
        {"Student_ID": sorted(EXPECTED_STUDENT_IDS)}
    ).set_index("Student_ID")

    definitions = [
        (eeg, "EEG_Question_Count"),
        (gsr, "GSR_Question_Count"),
        (physiology, "EEG_GSR_Matched_Count"),
        (final, "Final_Question_Count"),
    ]

    for df, column in definitions:
        if df.empty:
            counts = pd.Series(dtype="int64", name=column)
        else:
            counts = (
                df.groupby("Student_ID")["QuestionKey"]
                .nunique()
                .rename(column)
            )
        coverage = coverage.join(counts)

    count_columns = [
        "EEG_Question_Count",
        "GSR_Question_Count",
        "EEG_GSR_Matched_Count",
        "Final_Question_Count",
    ]

    coverage[count_columns] = coverage[count_columns].fillna(0).astype(int)
    coverage["EEG_Usable"] = coverage["EEG_Question_Count"] > 0
    coverage["GSR_Usable"] = coverage["GSR_Question_Count"] > 0
    coverage["EEG_GSR_Usable"] = coverage["EEG_GSR_Matched_Count"] > 0
    coverage["Final_Included"] = coverage["Final_Question_Count"] > 0

    return coverage.reset_index()


def download_csv(label, df, filename):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


# ================================================================
# MODEL BUILDERS
# ================================================================

def build_model(model_name):
    if model_name == "Logistic Regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=0.1,
            )),
        ])

    if model_name == "SVM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=0.5,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=42,
            )),
        ])

    if model_name == "Random Forest":
        # Conservative settings are intended to limit overfitting.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("random_forest", RandomForestClassifier(
                n_estimators=500,
                max_depth=6,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )),
        ])

    if model_name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM is not installed. Run: pip install lightgbm"
            ) from exc

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("lightgbm", LGBMClassifier(
                objective="multiclass",
                n_estimators=200,
                learning_rate=0.03,
                num_leaves=7,
                max_depth=3,
                min_child_samples=15,
                class_weight="balanced",
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )),
        ])

    raise ValueError(f"Unsupported model: {model_name}")


def make_xgboost_model(y_train):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. Run: pip install xgboost"
        ) from exc

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    n_classes = len(label_encoder.classes_)

    common_params = {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    if n_classes == 2:
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common_params,
        )
    else:
        classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            **common_params,
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgboost", classifier),
    ])

    return model, label_encoder, y_encoded


# ================================================================
# SINGLE TRAIN / TEST SPLIT
# ================================================================

def make_student_split(X, y, groups):
    """
    Make one fixed student-level split.

    The test fraction is fixed in code at 25%; it is not a user control.
    Tries several seeds to find a split with all classes represented
    in the training data.
    """
    all_classes = set(y.unique())

    for seed in range(42, 142):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=seed,
        )

        train_idx, test_idx = next(splitter.split(X, y, groups))

        if set(y.iloc[train_idx].unique()) == all_classes:
            return train_idx, test_idx

    raise ValueError(
        "Could not make a student-level split with every class represented "
        "in training. A target class may occur in too few students. Add "
        "labeled data or consider combining rare classes if appropriate."
    )


def train_and_evaluate_models(
    data,
    target_column,
    selected_models,
    split_type,
):
    """Train selected models and return the requested five result columns."""
    working = data.copy()

    if target_column not in working.columns:
        raise ValueError(f"Target '{target_column}' was not found.")

    working[target_column] = (
        working[target_column].astype("string").str.strip()
    )
    working = working.dropna(subset=[target_column])
    working = working[working[target_column] != ""]

    excluded = {
        "Student_ID",
        "QuestionKey",
        "Category",
        "Cognitive_Load",
        "Difficulty",
        "verdict",
        "ResponseTime",
        "PSY_Category",
        "PSY_Cognitive_Load",
        "PSY_Difficulty",
        "PSY_verdict",
        "Timestamp",
        "TimeStamp",
        "UnixTime",
        "EEG_Start_Time",
        "EEG_End_Time",
        "EEG_UnixTime_Start",
        "EEG_UnixTime_End",
    }

    feature_columns = [
        column
        for column in working.select_dtypes(include="number").columns
        if column not in excluded and column != target_column
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns are available.")

    X = working[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]
    working = working.loc[X.index]
    y = working[target_column].astype(str)

    if y.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")

    if split_type == "By student":
        if "Student_ID" not in working.columns:
            raise ValueError("Student_ID is required for a by-student split.")

        groups = working["Student_ID"].astype(str)

        if groups.nunique() < 2:
            raise ValueError(
                "A by-student split requires at least two different students."
            )

        train_idx, test_idx = make_student_split(X, y, groups)

    else:
        class_counts = y.value_counts()
        test_rows = int(np.ceil(len(y) * 0.25))
        can_stratify = (
            class_counts.min() >= 2
            and test_rows >= y.nunique()
            and len(y) - test_rows >= y.nunique()
        )

        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.25,
            random_state=42,
            stratify=y if can_stratify else None,
        )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if y_train.nunique() < 2:
        raise ValueError(
            "The training split has fewer than two target classes."
        )

    all_classes = set(y.unique())
    training_classes = set(y_train.unique())

    if training_classes != all_classes:
        raise ValueError(
            "Some target classes are missing from the training split: "
            f"{sorted(all_classes - training_classes)}"
        )

    result_rows = []
    predictions_by_model = {}

    for model_name in selected_models:
        label_encoder = None

        if model_name == "XGBoost":
            model, label_encoder, y_train_fit = make_xgboost_model(y_train)
        else:
            model = build_model(model_name)
            y_train_fit = y_train

        model.fit(X_train, y_train_fit)
        predictions = np.asarray(model.predict(X_test))

        if model_name == "XGBoost":
            predictions = label_encoder.inverse_transform(
                predictions.astype(int)
            )

        actual = y_test.to_numpy()

        result_rows.append({
            "Model": model_name,
            "Precision": precision_score(
                actual,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "Recall": recall_score(
                actual,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "F1-score": f1_score(
                actual,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "Accuracy": accuracy_score(actual, predictions),
        })

        prediction_frame = pd.DataFrame({
            "Actual": actual,
            "Predicted": predictions,
        })

        if "Student_ID" in working.columns:
            prediction_frame.insert(
                0,
                "Student_ID",
                working.iloc[test_idx]["Student_ID"].to_numpy(),
            )

        if "QuestionKey" in working.columns:
            prediction_frame.insert(
                1 if "Student_ID" in prediction_frame.columns else 0,
                "QuestionKey",
                working.iloc[test_idx]["QuestionKey"].to_numpy(),
            )

        predictions_by_model[model_name] = prediction_frame

    results = pd.DataFrame(
        result_rows,
        columns=["Model", "Precision", "Recall", "F1-score", "Accuracy"],
    )

    return results, predictions_by_model


# ================================================================
# STREAMLIT INTERFACE
# ================================================================

st.set_page_config(
    page_title="EEG + GSR + PSY Merger and ML",
    layout="wide",
)

st.title("Raw EEG + GSR + PSY Question-Level Merger")

st.write(
    "Upload raw EEG and GSR CSV files. Student IDs are read from the "
    "Student_ID column when available; otherwise use filenames such as "
    "`1_EEG.csv` or `12-GSR.csv`."
)

input_mode = st.radio(
    "Input mode",
    [
        "Single file per modality",
        "Multiple student files per modality",
    ],
    horizontal=True,
)

if input_mode == "Single file per modality":
    eeg_upload = st.file_uploader(
        "Raw EEG CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="eeg_single",
    )
    gsr_upload = st.file_uploader(
        "Raw GSR CSV",
        type=["csv"],
        accept_multiple_files=False,
        key="gsr_single",
    )
    psy_upload = st.file_uploader(
        "PSY CSV (optional)",
        type=["csv"],
        accept_multiple_files=False,
        key="psy_single",
    )

    eeg_files = [eeg_upload] if eeg_upload else []
    gsr_files = [gsr_upload] if gsr_upload else []
    psy_files = [psy_upload] if psy_upload else []
else:
    eeg_files = st.file_uploader(
        "Raw EEG CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="eeg_multiple",
        help="If Student_ID is absent, start the filename with the ID.",
    )
    gsr_files = st.file_uploader(
        "Raw GSR CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="gsr_multiple",
        help="If Student_ID is absent, start the filename with the ID.",
    )
    psy_files = st.file_uploader(
        "PSY CSV files (optional)",
        type=["csv"],
        accept_multiple_files=True,
        key="psy_multiple",
        help="If Student_ID is absent, start the filename with the ID.",
    )


if st.button("Process and merge", type="primary"):
    if not eeg_files:
        st.error("Upload at least one raw EEG CSV.")
    elif not gsr_files:
        st.error("Upload at least one raw GSR CSV.")
    else:
        try:
            with st.spinner("Processing and merging files..."):
                eeg = process_eeg_files(eeg_files)
                gsr = process_gsr_files(gsr_files)
                psy = process_psy_files(psy_files)

                (
                    final,
                    physiology,
                    eeg_only,
                    gsr_only,
                    physiology_without_psy,
                ) = merge_data(eeg, gsr, psy)

                coverage = make_coverage(eeg, gsr, physiology, final)

            st.session_state["final_data"] = final
            st.session_state["eeg_data"] = eeg
            st.session_state["gsr_data"] = gsr
            st.session_state["physiology_data"] = physiology
            st.session_state["eeg_only"] = eeg_only
            st.session_state["gsr_only"] = gsr_only
            st.session_state["physiology_without_psy"] = (
                physiology_without_psy
            )
            st.session_state["coverage"] = coverage

            if final.empty:
                st.warning(
                    "No matching student-question rows were found. "
                    "Check Student_ID and QuestionKey values."
                )
            else:
                st.success("Processing and merging completed.")

        except Exception as exc:
            st.error(f"Processing failed: {exc}")


# ================================================================
# DISPLAY DATA AND MODEL RESULTS
# ================================================================

if "final_data" in st.session_state:
    final = st.session_state["final_data"]
    eeg = st.session_state["eeg_data"]
    gsr = st.session_state["gsr_data"]
    physiology = st.session_state["physiology_data"]
    eeg_only = st.session_state["eeg_only"]
    gsr_only = st.session_state["gsr_only"]
    physiology_without_psy = st.session_state[
        "physiology_without_psy"
    ]
    coverage = st.session_state["coverage"]

    col1, col2 = st.columns(2)
    col1.metric("Final rows", f"{len(final):,}")
    col2.metric(
        "Students in final dataset",
        final["Student_ID"].nunique() if not final.empty else 0,
    )

    st.subheader("Final dataset")
    st.dataframe(final.head(100), use_container_width=True)

    st.subheader("Download output files")
    download_csv("Download final dataset", final, "EEG_GSR_PSY_Final.csv")
    download_csv("Download question-level EEG", eeg, "EEG_Question_Level.csv")
    download_csv("Download question-level GSR", gsr, "GSR_Question_Level.csv")
    download_csv(
        "Download EEG-GSR merged data",
        physiology,
        "EEG_GSR_Merged.csv",
    )
    download_csv(
        "Download EEG questions without GSR",
        eeg_only,
        "EEG_Questions_Without_GSR.csv",
    )
    download_csv(
        "Download GSR questions without EEG",
        gsr_only,
        "GSR_Questions_Without_EEG.csv",
    )
    download_csv(
        "Download EEG-GSR questions without PSY",
        physiology_without_psy,
        "EEG_GSR_Questions_Without_PSY.csv",
    )
    download_csv(
        "Download student coverage report",
        coverage,
        "Student_Coverage_Report.csv",
    )

    with st.expander("Student coverage"):
        st.dataframe(coverage, use_container_width=True)

    st.subheader("Compare machine-learning models")

    possible_targets = [
        "Category",
        "Cognitive_Load",
        "Difficulty",
        "verdict",
        "PSY_Category",
        "PSY_Cognitive_Load",
        "PSY_Difficulty",
        "PSY_verdict",
    ]

    available_targets = [
        column for column in possible_targets if column in final.columns
    ]

    if final.empty:
        st.info("No merged data is available for training.")
    elif not available_targets:
        st.info(
            "No recognized target column was found. PSY data should include "
            "Category, Cognitive_Load, Difficulty, or verdict."
        )
    else:
        target_column = st.selectbox(
            "Choose target label",
            available_targets,
        )

        selected_models = st.multiselect(
            "Choose models",
            MODEL_OPTIONS,
            default=[
                "Logistic Regression",
                "SVM",
                "Random Forest",
            ],
        )

        split_type = st.radio(
            "Evaluation split",
            ["By student", "Random rows"],
            horizontal=True,
            help=(
                "By student keeps test students separate from training "
                "students. Random rows may put the same student in both sets."
            ),
        )

        if st.button("Train and evaluate selected models"):
            if not selected_models:
                st.warning("Select at least one model.")
            else:
                try:
                    with st.spinner("Training selected models..."):
                        results, predictions_by_model = (
                            train_and_evaluate_models(
                                data=final,
                                target_column=target_column,
                                selected_models=selected_models,
                                split_type=split_type,
                            )
                        )

                    st.subheader("Model performance")
                    st.dataframe(
                        results.style.format({
                            "Precision": "{:.3f}",
                            "Recall": "{:.3f}",
                            "F1-score": "{:.3f}",
                            "Accuracy": "{:.3f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                    with st.expander("Download test predictions"):
                        for model_name, prediction_df in (
                            predictions_by_model.items()
                        ):
                            safe_name = (
                                model_name.lower()
                                .replace(" ", "_")
                                .replace("-", "_")
                            )

                            download_csv(
                                f"Download {model_name} predictions",
                                prediction_df,
                                f"{safe_name}_predictions.csv",
                            )

                except ImportError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Model training failed: {exc}")