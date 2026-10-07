
from pathlib import Path
import json
import uuid
import warnings
import re
import os
import joblib
import sqlite3
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi import Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "dataset"
CACHE_DIR = MODEL_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "signup.db")


# Database

def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT UNIQUE,
            name TEXT,
            email TEXT,
            mobile TEXT,
            password TEXT
        )
        """
    )
    con.commit()
    con.close()


init_db()


MODEL_PATH = MODEL_DIR / "catboost_model.joblib"
ENCODER_PATH = MODEL_DIR / "categorical_encoder.joblib"
CATCOLS_PATH = MODEL_DIR / "categorical_columns.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.joblib"
RAW_DATA_PATH = DATA_DIR / "project_risk_raw_dataset.csv"
LIME_CACHE_PATH = CACHE_DIR / "lime_background.joblib"
UPLOAD_CACHE_DIR = CACHE_DIR / "uploads"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# The notebook maps the original four risk labels into two model classes.
RISK_LABELS = {
    0: "Low / Medium Risk",
    1: "High / Critical Risk",
}

app = FastAPI(title="Project Risk Prediction & LIME XAI", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.globals["get_flashed_messages"] = lambda **kwargs: []

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
categorical_cols = joblib.load(CATCOLS_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# LIME is intentionally imported lazily so the API can still expose a useful
# startup error if the optional XAI dependency has not been installed.
lime_explainer = None


def _build_lime_explainer():
    """Recreate the notebook's LIME background from the supplied raw dataset.

    The notebook builds LIME with X_train_balanced, where SMOTENC is applied
    after the same ordinal encoding and stratified split used for training.
    """
    global lime_explainer
    if lime_explainer is not None:
        return lime_explainer

    try:
        from imblearn.over_sampling import SMOTENC
        from lime.lime_tabular import LimeTabularExplainer
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "LIME dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    if LIME_CACHE_PATH.exists():
        cache = joblib.load(LIME_CACHE_PATH)
        background = cache["background"]
    else:
        raw = pd.read_csv(RAW_DATA_PATH)
        work = raw.drop(columns=["Project_ID", "Tech_Environment_Stability"], errors="ignore")
        X = work.drop(columns=["Risk_Level"], errors="ignore").copy()
        y = work["Risk_Level"].map({"Critical": 1, "High": 1, "Medium": 0, "Low": 0})

        missing = [c for c in feature_columns if c not in X.columns]
        if missing:
            raise RuntimeError(f"Raw dataset is missing LIME features: {missing}")
        X = X[feature_columns]
        X[categorical_cols] = _ordinal_encode_compat(X[categorical_cols])

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        categorical_indices = [X_train.columns.get_loc(c) for c in categorical_cols]
        smote = SMOTENC(categorical_features=categorical_indices, random_state=42)
        X_train_balanced, _ = smote.fit_resample(X_train, y_train)
        background = X_train_balanced.values
        joblib.dump({"background": background}, LIME_CACHE_PATH)

    categorical_indices = [feature_columns.index(c) for c in categorical_cols]
    categorical_names = {
        i: [str(v) for v in encoder.categories_[j]]
        for j, i in enumerate(categorical_indices)
    }

    lime_explainer = LimeTabularExplainer(
        training_data=np.asarray(background),
        feature_names=feature_columns,
        categorical_features=categorical_indices,
        categorical_names=categorical_names,
        class_names=[RISK_LABELS[0], RISK_LABELS[1]],
        mode="classification",
        random_state=42,
        discretize_continuous=True,
    )
    return lime_explainer


def _read_upload(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        from io import BytesIO
        return pd.read_excel(BytesIO(content))
    if suffix == ".xls":
        from io import BytesIO
        return pd.read_excel(BytesIO(content))
    if suffix == ".csv":
        from io import BytesIO
        return pd.read_csv(BytesIO(content))
    raise HTTPException(status_code=400, detail="Upload an Excel file (.xlsx/.xls/.xlsm) or CSV file.")


def _ordinal_encode_compat(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the saved OrdinalEncoder mapping with unknown_category=-1.

    The notebook used OrdinalEncoder(handle_unknown="use_encoded_value",
    unknown_value=-1). Reading an old serialized sklearn encoder with a newer
    sklearn can fail inside transform(), so we reproduce the stored mapping
    directly from encoder.categories_.
    """
    out = frame.copy()
    for j, col in enumerate(categorical_cols):
        categories = [str(v) for v in encoder.categories_[j]]
        mapping = {value: idx for idx, value in enumerate(categories)}
        out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(float)
    return out


def _prepare_features(df: pd.DataFrame):
    work = df.copy()

    # These columns exist in the source dataset but are not model inputs.
    work = work.drop(columns=["Project_ID", "Risk_Level", "Tech_Environment_Stability"], errors="ignore")

    missing = [c for c in feature_columns if c not in work.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "The uploaded file is missing required model columns.",
                "missing_columns": missing,
            },
        )

    # Ignore extra columns and preserve the exact model feature order.
    work = work[feature_columns].copy()

    # Match the notebook's OrdinalEncoder semantics without calling its
    # version-sensitive transform() method. This also makes the app robust
    # when a newer scikit-learn reads the serialized 1.0.2 encoder.
    work[categorical_cols] = _ordinal_encode_compat(work[categorical_cols])
    numeric_cols = [c for c in feature_columns if c not in categorical_cols]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    return work


def _prediction_rows(raw_df: pd.DataFrame, X: pd.DataFrame):
    pred = model.predict(X).ravel().astype(int)
    proba = model.predict_proba(X)
    rows = []
    for i, p in enumerate(pred):
        confidence = float(np.max(proba[i]))
        rows.append({
            "row_index": i,
            "prediction_code": int(p),
            "prediction": RISK_LABELS[int(p)],
            "confidence": round(confidence * 100, 2),
            "probability_low_medium": round(float(proba[i][0]) * 100, 2),
            "probability_high_critical": round(float(proba[i][1]) * 100, 2),
        })
    return rows








@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "CatBoost", "features": len(feature_columns)}


@app.post("/api/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    original_name = file.filename or "upload.xlsx"
    raw_df = _read_upload(original_name, content)
    X = _prepare_features(raw_df)
    rows = _prediction_rows(raw_df, X)

    # Keep the uploaded source available for row-specific LIME explanations
    # after the UI navigates from index.html to result.html.
    token = uuid.uuid4().hex
    suffix = Path(original_name).suffix.lower() or ".xlsx"
    upload_path = UPLOAD_CACHE_DIR / f"{token}{suffix}"
    upload_path.write_bytes(content)

    # Return a compact preview of original values for the UI table.
    preview_cols = [c for c in ["Project_ID", "Project_Type", "Team_Size", "Project_Budget_USD", "Project_Phase"] if c in raw_df.columns]
    preview = raw_df[preview_cols].copy() if preview_cols else pd.DataFrame(index=raw_df.index)
    preview = preview.fillna("").astype(str).to_dict(orient="records")
    for i, item in enumerate(preview):
        rows[i]["input"] = item

    return JSONResponse({
        "token": token,
        "filename": original_name,
        "row_count": len(X),
        "columns_used": feature_columns,
        "predictions": rows,
    })


@app.post("/api/explain/{row_index}")
async def explain(row_index: int, token: str, num_features: int = 10):
    if row_index < 0:
        raise HTTPException(status_code=400, detail="row_index must be >= 0")
    num_features = max(1, min(int(num_features), 20))

    if not token or any(ch not in "0123456789abcdef" for ch in token.lower()) or len(token) != 32:
        raise HTTPException(status_code=400, detail="Invalid upload token.")
    matches = list(UPLOAD_CACHE_DIR.glob(f"{token}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Uploaded file session has expired or does not exist.")
    upload_path = matches[0]
    raw_df = _read_upload(upload_path.name, upload_path.read_bytes())
    X = _prepare_features(raw_df)
    if row_index >= len(X):
        raise HTTPException(status_code=404, detail="Selected row does not exist.")

    try:
        explainer = _build_lime_explainer()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    instance = X.iloc[row_index].values.astype(float)
    prediction_code = int(model.predict(instance.reshape(1, -1)).ravel()[0])
    probabilities = model.predict_proba(instance.reshape(1, -1))[0]

    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=num_features,
        top_labels=2,
    )

    # Explain the class that was actually predicted, which makes the XAI panel
    # follow the selected prediction in the UI.
    pairs = explanation.as_list(label=prediction_code)
    factors = [
        {
            "feature": str(feature),
            "weight": round(float(weight), 6),
            "direction": "increases risk" if weight > 0 else "decreases risk",
            "impact": round(abs(float(weight)), 6),
        }
        for feature, weight in pairs
    ]

    return {
        "row_index": row_index,
        "prediction": RISK_LABELS[prediction_code],
        "probability_low_medium": round(float(probabilities[0]) * 100, 2),
        "probability_high_critical": round(float(probabilities[1]) * 100, 2),
        "factors": factors,
    }


# Public routes
@app.get("/", response_class=HTMLResponse)
async def index_home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/results", response_class=HTMLResponse)
@app.get("/results.html", response_class=HTMLResponse)
async def results_page(request: Request):
    return templates.TemplateResponse(request=request, name="results.html")

@app.get("/graphs", response_class=HTMLResponse)
async def graphs(request: Request):
    return templates.TemplateResponse(request=request, name="graphs.html")

@app.get("/home", response_class=HTMLResponse)
@app.get("/predict", response_class=HTMLResponse)
@app.get("/predict.html", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html")

@app.get("/login", response_class=HTMLResponse)
async def signin(request: Request):
    return templates.TemplateResponse(request=request, name="signin.html")

@app.get("/logon", response_class=HTMLResponse)
async def signup(request: Request):
    return templates.TemplateResponse(request=request, name="signup.html")


@app.post("/signup", response_class=HTMLResponse)
async def signup_post(
    request: Request,
    user: str = Form(""),
    name: str = Form(""),
    email: str = Form(""),
    mobile: str = Form(""),
    password: str = Form(""),
):
    username_pattern = r"^.{6,}$"
    name_pattern = r"^[A-Za-z ]{3,}$"
    email_pattern = r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$"
    mobile_pattern = r"^[6-9][0-9]{9}$"
    password_pattern = r"^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$"

    if not re.match(username_pattern, user):
        return templates.TemplateResponse("signup.html", {"request": request, "message": "Username must be at least 6 characters."},)
    if not re.match(name_pattern, name):
        return templates.TemplateResponse("signup.html",{"request": request,
                "message": "Full Name must be at least 3 letters, only letters and spaces allowed.",
            },
        )
    if not re.match(email_pattern, email):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "message": "Enter a valid email address."},
        )
    if not re.match(mobile_pattern, mobile):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "message": "Mobile must start with 6-9 and be 10 digits."},
        )
    if not re.match(password_pattern, password):
        return templates.TemplateResponse(
            "signup.html",
            {
                "request": request,
                "message": "Password must be at least 8 characters, with an uppercase letter, a number, and a lowercase letter.",
            },
        )

    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT 1 FROM info WHERE user = ?", (user,))
        if cur.fetchone():
            con.close()
            return templates.TemplateResponse(
                "signup.html",
                {"request": request, "message": "Username already exists. Please choose another."},
            )

        cur.execute(
            "INSERT INTO info (user, name, email, mobile, password) VALUES (?, ?, ?, ?, ?)",
            (user, name, email, mobile, password),
        )
        con.commit()
        con.close()
        return RedirectResponse(url="/login", status_code=303)
    except Exception as exc:
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "message": f"Database error: {exc}"},
        )


@app.post("/signin", response_class=HTMLResponse)
async def signin_post(
    request: Request,
    user: str = Form(""),
    password: str = Form(""),
):
    if user == "admin" and password == "admin":
        response = RedirectResponse(url="/home", status_code=303)
        response.set_cookie(key="session_user", value="admin")
        return response

    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            "SELECT user, password FROM info WHERE user = ? AND password = ?",
            (user, password),
        )
        data = cur.fetchone()
        con.close()

        if data is None:
            return templates.TemplateResponse(
                "signin.html",
                {"request": request, "message": "Invalid username or password."},
            )

        response = RedirectResponse(url="/home", status_code=303)
        response.set_cookie(key="session_user", value=str(data[0]))
        return response
    except Exception as exc:
        return templates.TemplateResponse(
            "signin.html",
            {"request": request, "message": f"Database error: {exc}"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)