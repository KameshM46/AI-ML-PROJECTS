import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "student_performance.csv"
FEATURES = [
    "attendance",
    "study_hours",
    "previous_score",
    "assignment_completion",
    "sleep_hours",
]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_model():
    data = load_data()
    X = data[FEATURES]
    y = data["final_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=150, max_depth=8, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, metrics, data

model, metrics, data = train_model()

st.title("🎓 Student Performance Prediction")
st.write(
    "A beginner-friendly machine learning project that estimates a student's "
    "final score from attendance, study habits, previous performance, and "
    "assignment completion."
)
st.info(
    "Learning/demo project: the included dataset is synthetic (artificially generated), "
    "not real student records. Predictions should not be used for real academic decisions."
)

tab1, tab2, tab3 = st.tabs(["Predict a Student", "Explore Dataset", "About the Model"])

with tab1:
    st.subheader("Enter student details")
    c1, c2 = st.columns(2)
    with c1:
        attendance = st.slider("Attendance (%)", 0, 100, 80)
        study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
        previous_score = st.slider("Previous exam score", 0, 100, 65)
    with c2:
        assignment_completion = st.slider("Assignment completion (%)", 0, 100, 75)
        sleep_hours = st.slider("Sleep hours per day", 0.0, 12.0, 7.0, 0.5)

    if st.button("Predict Performance", type="primary"):
        student = pd.DataFrame([{
            "attendance": attendance,
            "study_hours": study_hours,
            "previous_score": previous_score,
            "assignment_completion": assignment_completion,
            "sleep_hours": sleep_hours,
        }])
        predicted_score = float(np.clip(model.predict(student[FEATURES])[0], 0, 100))
        st.metric("Estimated final score", f"{predicted_score:.1f} / 100")
        if predicted_score < 50:
            st.error("Support indicator: this estimate is below 50. Consider reviewing attendance, study time, and assignment completion.")
        elif predicted_score < 70:
            st.warning("Support indicator: this estimate is in the middle range. Consistent practice may help.")
        else:
            st.success("Support indicator: this estimate is in the higher range. Keep maintaining good study habits.")
        st.caption("This is an estimate from a synthetic training dataset, not a guarantee or a formal student assessment.")

with tab2:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Sample records", len(data))
    b.metric("Average final score", f"{data['final_score'].mean():.1f}")
    c.metric("Features", len(FEATURES))
    st.dataframe(data.head(20), use_container_width=True)
    st.subheader("Feature relationships")
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("How it works")
    st.markdown("""
    1. The app loads the sample CSV dataset.
    2. It uses five input features to estimate the final score.
    3. A **Random Forest Regressor** learns patterns from the training data.
    4. The model is evaluated on a held-out test set.
    """)
    x, y = st.columns(2)
    x.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.2f}")
    y.metric("R² score", f"{metrics['r2']:.3f}")
    st.write("Feature importance")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Metrics are calculated using the bundled synthetic dataset. They do not represent performance on real-world student data.")
