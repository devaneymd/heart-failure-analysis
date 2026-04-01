import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    import xgboost as xgb
    from mlxtend.plotting import plot_learning_curves
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        recall_score,
        precision_score,
        matthews_corrcoef,
        roc_auc_score,
    )


@app.cell
def _():
    hf_data = pl.read_csv(
        "heart_failure_clinical_records_dataset.csv",
        schema_overrides={"age": pl.Float32},
    )
    return (hf_data,)


@app.cell
def _(hf_data):
    # Separating predictors and target variable
    X, y = (
        hf_data.select(pl.exclude("DEATH_EVENT", "time")),
        hf_data.select(pl.col("DEATH_EVENT")),
    )
    # Creating a training set and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=1
    )

    # Standardizing the continuous variables
    cont_vars = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
    ]
    bin_vars = [
        "anaemia",
        "diabetes",
        "high_blood_pressure",
        "sex",
        "smoking",
        "DEATH_EVENT",
    ]

    stdsc = ColumnTransformer(
        [
            ("standardize", StandardScaler(), cont_vars),
            ("nothing", "passthrough", bin_vars[:-1]),
        ]
    )
    # Only fit the training data, not the testing data
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    return (
        X_test,
        X_test_std,
        X_train,
        X_train_std,
        bin_vars,
        cont_vars,
        y_test,
        y_train,
    )


@app.cell
def _(X_train_std, y_train):
    lr = LogisticRegression(solver="lbfgs", class_weight="balanced", random_state=0)
    scores_lr = cross_val_score(
        estimator=lr,
        X=X_train_std,
        y=y_train.to_numpy().ravel(),
        cv=10,
        scoring="recall",
        n_jobs=-1,
    )
    print(f"CV recall: {np.mean(scores_lr):.3f} +/- {np.std(scores_lr):.3f}")
    return lr, scores_lr


@app.cell
def _(X_train_std, y_train):
    svm = SVC(kernel="rbf", C=1.0, random_state=0, class_weight="balanced")
    scores_svm = cross_val_score(
        estimator=svm,
        X=X_train_std,
        y=y_train.to_numpy().ravel(),
        cv=10,
        scoring="recall",
        n_jobs=-1,
    )
    print(f"CV recall: {np.mean(scores_svm):.3f} +/- {np.std(scores_svm):.3f}")
    return scores_svm, svm


@app.cell
def _(X_train, y_train):
    random_forest = RandomForestClassifier(
        n_estimators=500, random_state=0, class_weight="balanced"
    )
    scores_rf = cross_val_score(
        estimator=random_forest,
        X=X_train,
        y=y_train.to_numpy().ravel(),
        cv=10,
        scoring="recall",
        n_jobs=-1,
    )
    print(f"CV recall: {np.mean(scores_rf):.3f} +/- {np.std(scores_rf):.3f}")
    return random_forest, scores_rf


@app.cell
def _(X_train, y_train):
    neg, pos = np.bincount(y_train.to_numpy().ravel())
    scale = neg / pos
    gb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        random_state=1,
        scale_pos_weight=scale,
    )
    scores_gb = cross_val_score(
        estimator=gb,
        X=X_train,
        y=y_train.to_numpy().flatten(),
        cv=60,
        scoring="recall",
        n_jobs=-1,
    )
    print(f"CV recall: {np.mean(scores_gb):.3f} +/- {np.std(scores_gb):.3f}")
    return gb, scale, scores_gb


@app.cell
def _(scores_gb, scores_lr, scores_rf, scores_svm):
    from great_tables import GT

    full_cv = pl.DataFrame(
        {
            "Full model": ["Logistic regression", "SVM", "Random forest", "XGBoost"],
            "Mean CV recall score": [
                np.mean(scores_lr),
                np.mean(scores_svm),
                np.mean(scores_rf),
                np.mean(scores_gb),
            ],
            "Standard deviation": [
                np.std(scores_lr),
                np.std(scores_svm),
                np.std(scores_rf),
                np.std(scores_gb),
            ],
        }
    )
    (GT(full_cv).fmt_number(columns=[1, 2]).opt_row_striping())
    return (GT,)


@app.cell
def _(X_test_std, X_train_std, lr, y_test, y_train):
    plot_learning_curves(
        X_train_std,
        y_train.to_numpy().flatten(),
        X_test_std,
        y_test.to_numpy().flatten(),
        clf=lr,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("Performance (recall)", fontsize=14)
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_test, X_train, random_forest, y_test, y_train):
    plot_learning_curves(
        X_train,
        y_train.to_numpy().flatten(),
        X_test,
        y_test.to_numpy().flatten(),
        clf=random_forest,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("Performance (recall)", fontsize=14)
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_test_std, X_train_std, svm, y_test, y_train):
    plot_learning_curves(
        X_train_std,
        y_train.to_numpy().flatten(),
        X_test_std,
        y_test.to_numpy().flatten(),
        clf=svm,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("")
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_test, X_train, gb, y_test, y_train):
    plot_learning_curves(
        X_train,
        y_train.to_numpy().flatten(),
        X_test,
        y_test.to_numpy().flatten(),
        clf=gb,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("")
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(X_train, y_train.to_numpy().ravel())
    plt.figure(figsize=(8, 6))
    plt.barh(y=X_train.columns, width=forest.feature_importances_, color="gray")
    plt.xlabel("Importance score", fontsize=12)
    plt.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_train, bin_vars, cont_vars, y_train):
    # standardizing with columntrans isnt necessary before selectfrommodel technically, but it needs to be done first due to pipeline
    lr_pipeline = Pipeline(
        [
            [
                "stdsc",
                ColumnTransformer(
                    [
                        ("standardize", StandardScaler(), cont_vars),
                        ("nothing", "passthrough", bin_vars[:-1]),
                    ]
                ),
            ],
            [
                "sfm",
                SelectFromModel(
                    RandomForestClassifier(n_estimators=500, random_state=0),
                    threshold=0.15,
                ),
            ],
            [
                "lr",
                LogisticRegression(
                    solver="lbfgs", random_state=0, class_weight="balanced"
                ),
            ],
        ]
    )

    scores_lr_red = cross_val_score(
        estimator=lr_pipeline,
        X=X_train,
        y=y_train.to_numpy().ravel(),
        cv=10,
        scoring="recall",
        n_jobs=-1,
    )
    print(
        f"CV reduced recall: {np.mean(scores_lr_red):.3f} +/- {np.std(scores_lr_red):.3f}"
    )
    return lr_pipeline, scores_lr_red


@app.cell
def _(X_train, bin_vars, cont_vars, y_train):
    # standardizing with columntrans isnt necessary before selectfrommodel technically, but it needs to be done first due to pipeline
    svm_pipeline = Pipeline(
        [
            [
                "stdsc",
                ColumnTransformer(
                    [
                        ("standardize", StandardScaler(), cont_vars),
                        ("nothing", "passthrough", bin_vars[:-1]),
                    ]
                ),
            ],
            [
                "sfm",
                SelectFromModel(
                    RandomForestClassifier(n_estimators=500, random_state=0),
                    threshold=0.15,
                ),
            ],
            ["sv", SVC(kernel="rbf", C=1.0, random_state=0, class_weight="balanced")],
        ]
    )

    scores_svm_red = cross_val_score(
        estimator=svm_pipeline,
        X=X_train,
        y=y_train.to_numpy().ravel(),
        cv=10,
        scoring="recall",
        n_jobs=-1,
    )
    print(
        f"CV reduced recall: {np.mean(scores_svm_red):.3f} +/- {np.std(scores_svm_red):.3f}"
    )
    return scores_svm_red, svm_pipeline


@app.cell
def _(X_train, y_train):
    # standardizing with columntrans isnt necessary before selectfrommodel technically, but it needs to be done first due to pipeline
    rf_pipeline = Pipeline(
        [
            [
                "sfm",
                SelectFromModel(
                    RandomForestClassifier(n_estimators=500, random_state=0),
                    threshold=0.15,
                ),
            ],
            [
                "rf",
                RandomForestClassifier(
                    n_estimators=500, random_state=0, class_weight="balanced"
                ),
            ],
        ]
    )

    scores_rf_red = cross_val_score(
        estimator=rf_pipeline,
        X=X_train,
        y=y_train.to_numpy().ravel(),
        cv=10,
        scoring="recall",
        n_jobs=-1,
    )
    print(
        f"CV reduced recall: {np.mean(scores_rf_red):.3f} +/- {np.std(scores_rf_red):.3f}"
    )
    return rf_pipeline, scores_rf_red


@app.cell
def _(X_train, scale, y_train):
    # standardizing with columntrans isnt necessary before selectfrommodel technically, but it needs to be done first due to pipeline
    gb_pipeline = Pipeline(
        [
            [
                "sfm",
                SelectFromModel(
                    RandomForestClassifier(n_estimators=500, random_state=0),
                    threshold=0.15,
                ),
            ],
            [
                "gb",
                xgb.XGBClassifier(
                    n_estimators=1000,
                    learning_rate=0.01,
                    max_depth=4,
                    random_state=1,
                    scale_pos_weight=scale,
                ),
            ],
        ]
    )

    scores_gb_red = cross_val_score(
        estimator=gb_pipeline,
        X=X_train,
        y=y_train.to_numpy().ravel(),
        cv=10,
        scoring="recall",
        n_jobs=-1,
    )
    print(
        f"CV reduced recall: {np.mean(scores_gb_red):.3f} +/- {np.std(scores_gb_red):.3f}"
    )
    return gb_pipeline, scores_gb_red


@app.cell
def _(GT, scores_gb_red, scores_lr_red, scores_rf_red, scores_svm_red):
    red_cv = pl.DataFrame(
        {
            "Reduced model": ["Logistic regression", "SVM", "Random forest", "XGBoost"],
            "Mean CV recall score": [
                np.mean(scores_lr_red),
                np.mean(scores_svm_red),
                np.mean(scores_rf_red),
                np.mean(scores_gb_red),
            ],
            "Standard deviation": [
                np.std(scores_lr_red),
                np.std(scores_svm_red),
                np.std(scores_rf_red),
                np.std(scores_gb_red),
            ],
        }
    )
    (GT(red_cv).fmt_number(columns=[1, 2]).opt_row_striping())
    return


@app.cell
def _(X_test, X_train, lr_pipeline, y_test, y_train):
    plot_learning_curves(
        X_train,
        y_train.to_numpy().flatten(),
        X_test,
        y_test.to_numpy().flatten(),
        clf=lr_pipeline,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("Performance (recall)", fontsize=14)
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_test, X_train, rf_pipeline, y_test, y_train):
    plot_learning_curves(
        X_train,
        y_train.to_numpy().flatten(),
        X_test,
        y_test.to_numpy().flatten(),
        clf=rf_pipeline,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("Performance (recall)", fontsize=14)
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_test, X_train, svm_pipeline, y_test, y_train):
    plot_learning_curves(
        X_train,
        y_train.to_numpy().flatten(),
        X_test,
        y_test.to_numpy().flatten(),
        clf=svm_pipeline,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("Performance (recall)", fontsize=14)
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_test, X_train, gb_pipeline, y_test, y_train):
    plot_learning_curves(
        X_train,
        y_train.to_numpy().flatten(),
        X_test,
        y_test.to_numpy().flatten(),
        clf=gb_pipeline,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("Performance (recall)", fontsize=14)
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_train, rf_pipeline, y_train):
    rf_param_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [None, 5, 10, 15],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4],
        "rf__class_weight": ["balanced"],
    }

    rf_gs = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=rf_param_grid,
        scoring="recall",
        cv=10,
        refit=True,
        n_jobs=-1,
    )
    rf_gs.fit(X_train, y_train.to_numpy().ravel())
    return (rf_gs,)


@app.cell
def _(rf_gs):
    rf_gs.best_params_
    return


@app.cell
def _(X_test, X_train, rf_gs, y_test, y_train):
    plot_learning_curves(
        X_train,
        y_train.to_numpy().flatten(),
        X_test,
        y_test.to_numpy().flatten(),
        clf=rf_gs.best_estimator_,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("Performance (recall)", fontsize=14)
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(X_train, gb_pipeline, y_train):
    gb_param_grid = {
        "gb__n_estimators": [100, 200],
        "gb__max_depth": [None, 1],
        "gb__learning_rate": [0.01, 0.1],
        "gb__subsample": [0.7, 0.9],
        "gb__colsample_bytree": [0.7, 0.9],
        "gb__min_child_weight": [2, 5],
        "gb__gamma": [0, 0.2],
    }

    gb_gs = GridSearchCV(
        estimator=gb_pipeline,
        param_grid=gb_param_grid,
        cv=10,
        refit=True,
        scoring="recall",
        n_jobs=-1,
    )

    gb_gs.fit(X_train, y_train.to_numpy().ravel())
    return (gb_gs,)


@app.cell
def _(gb_gs):
    gb_gs.best_score_
    return


@app.cell
def _(X_test, X_train, gb_gs, y_test, y_train):
    plot_learning_curves(
        X_train,
        y_train.to_numpy().flatten(),
        X_test,
        y_test.to_numpy().flatten(),
        clf=gb_gs.best_estimator_,
        scoring="recall",
        print_model=False,
    )
    plt.tick_params(axis="both", labelsize=12)
    plt.ylabel("Performance (recall)", fontsize=14)
    plt.xlabel("Training set size (%)", fontsize=14)
    plt.show()
    return


@app.cell
def _(gb_gs, lr_pipeline, rf_gs):
    from sklearn.ensemble import VotingClassifier

    vc = VotingClassifier(
        estimators=[
            ("rf", rf_gs.best_estimator_),
            ("lr", lr_pipeline),
            ("gb", gb_gs.best_estimator_),
        ]
    )
    return (vc,)


@app.cell
def _(X_train, vc, y_train):
    scores_vc = cross_val_score(
        estimator=vc,
        X=X_train,
        y=y_train.to_numpy().ravel(),
        cv=10,
        scoring="recall",
        n_jobs=-1,
    )
    print(f"CV recall score: {np.mean(scores_vc):.3f} +/- {np.std(scores_vc):.3f}")
    return


@app.cell
def _(X_test, gb_gs, lr_pipeline, rf_gs, svm_pipeline, y_test):
    # accuracy, recall, precision, roc_auc, f1, matthews_corrcoef
    lr_pred = lr_pipeline.predict(X=X_test)
    svm_pred = svm_pipeline.predict(X=X_test)
    rf_pred = rf_gs.best_estimator_.predict(X=X_test)
    gb_pred = gb_gs.best_estimator_.predict(X=X_test)
    final_scores = pl.DataFrame(
        {
            "Model": ["Logistic regression", "SVM", "Random forest", "XGBoost"],
            "Accuracy": [
                accuracy_score(y_test, lr_pred),
                accuracy_score(y_test, svm_pred),
                accuracy_score(y_test, rf_pred),
                accuracy_score(y_test, gb_pred),
            ],
            "Recall": [
                recall_score(y_test, lr_pred),
                recall_score(y_test, svm_pred),
                recall_score(y_test, rf_pred),
                recall_score(y_test, gb_pred),
            ],
            "Precision": [
                precision_score(y_test, lr_pred),
                precision_score(y_test, svm_pred),
                precision_score(y_test, rf_pred),
                precision_score(y_test, gb_pred),
            ],
            "AUC": [
                roc_auc_score(y_test, lr_pred),
                roc_auc_score(y_test, svm_pred),
                roc_auc_score(y_test, rf_pred),
                roc_auc_score(y_test, gb_pred),
            ],
            "F1": [
                f1_score(y_test, lr_pred),
                f1_score(y_test, svm_pred),
                f1_score(y_test, rf_pred),
                f1_score(y_test, gb_pred),
            ],
            "MCC": [
                matthews_corrcoef(y_test, lr_pred),
                matthews_corrcoef(y_test, svm_pred),
                matthews_corrcoef(y_test, rf_pred),
                matthews_corrcoef(y_test, gb_pred),
            ],
        }
    )
    return (final_scores,)


@app.cell
def _(GT, final_scores):
    (GT(final_scores).fmt_number(columns=[1, 2, 3, 4, 5, 6]).opt_row_striping())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
