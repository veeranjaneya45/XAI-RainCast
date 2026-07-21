"""
run_pipeline.py — single entrypoint that reproduces the entire project:

    data -> preprocessing -> train XGBoost -> evaluate -> SHAP -> MC Dropout

Run with:  python run_pipeline.py
"""
import time

from src.data_loader import load_raw
from src.preprocessing import prepare_dataset
from src.train_model import train_xgboost, save_model
from src.evaluate import evaluate_classifier
from src.explainability import run_shap_analysis
from src.uncertainty import MCDropoutNet, evaluate_uncertainty


def main():
    t0 = time.time()

    print("\n=== 1/5  Loading data ===")
    raw_df = load_raw()

    print("\n=== 2/5  Preprocessing ===")
    X_train, X_val, X_test, y_train, y_val, y_test, artifacts = prepare_dataset(raw_df)

    print("\n=== 3/5  Training XGBoost ===")
    model = train_xgboost(X_train, y_train, X_val, y_val)
    save_model(model)

    print("\n=== 4/5  Evaluating + SHAP explainability ===")
    metrics, proba = evaluate_classifier(model, X_test, y_test)
    run_shap_analysis(model, X_test)

    print("\n=== 5/5  Training MC Dropout uncertainty network ===")
    net = MCDropoutNet(n_in=X_train.shape[1])
    net.fit(X_train.values.astype(float), y_train.values.astype(float), epochs=40)
    net.save()
    evaluate_uncertainty(net, X_test.values.astype(float), y_test.values.astype(float))

    elapsed = time.time() - t0
    print(f"\n\u2705 Pipeline complete in {elapsed:.1f}s. "
          f"Artifacts saved to models/ and outputs/. Run the dashboard with:\n"
          f"    streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()