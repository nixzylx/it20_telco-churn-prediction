
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def train_models(X_train, y_train):
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    return rf_model, gb_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred)

    return accuracy, roc_auc, report

def save_model(model, filename):
    joblib.dump(model, filename)
