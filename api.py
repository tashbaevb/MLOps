from fastapi import FastAPI
import joblib, os, pandas as pd, time, threading, subprocess

app = FastAPI()

MODEL_DIR = "data/models"
DATA_FILE = "data/dataset/synthetic_demand.csv"
last_mtime = None


@app.get("/forecast")
def forecast(weeks: int):
    model, model_name = load_latest_model()
    X = pd.DataFrame([[weeks]], columns=["weeks"])
    prediction = float(model.predict(X)[0])
    return {"week": weeks, "prediction": prediction, "model_used": model_name}


def load_latest_model():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError("No models available.")
    latest = max(files, key=lambda f: os.path.getctime(os.path.join(MODEL_DIR, f)))
    model = joblib.load(os.path.join(MODEL_DIR, latest))
    return model, latest


def monitor_csv_changes():
    global last_mtime
    if not os.path.exists(DATA_FILE):
        print(f"File {DATA_FILE} not found")
        return
    last_mtime = os.path.getmtime(DATA_FILE)
    while True:
        try:
            time.sleep(10)
            mtime = os.path.getmtime(DATA_FILE)
            if mtime != last_mtime:
                print(f"Dataset changed → retrain")
                subprocess.run(["python", "monitor_and_retrain.py"])
                last_mtime = mtime
        except Exception as e:
            print(f"[Watcher] error: {e}")


@app.on_event("startup")
def start_watcher():
    thread = threading.Thread(target=monitor_csv_changes, daemon=True)
    thread.start()
    print("Watcher launched")
