import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    from mlxtend.plotting import heatmap


@app.cell
def hf_data():
    # Loading data from csv - changing dtype for 'age' column from i64 to f32 because of likely data entry error (e.g. 60.667)
    hf_data = pl.read_csv(
        "heart_failure_clinical_records_dataset.csv",
        schema_overrides={"age": pl.Float32},
    )
    return (hf_data,)


@app.cell
def _(hf_data):
    hf_data.null_count()
    return


@app.cell
def _(hf_data):
    cont_vars = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "time",
    ]
    bin_vars = [
        "anaemia",
        "diabetes",
        "high_blood_pressure",
        "sex",
        "smoking",
        "DEATH_EVENT",
    ]
    # getting some summary statistics - min, max, mean, median
    hf_data_min = hf_data[cont_vars].min().cast(pl.Float64)
    hf_data_max = hf_data[cont_vars].max().cast(pl.Float64)
    hf_data_mean = hf_data[cont_vars].mean().cast(pl.Float64)
    hf_data_median = hf_data[cont_vars].median().cast(pl.Float64)
    # getting the frequencies for the binary variables
    hf_data_freq = pl.DataFrame()
    for i, col in enumerate(bin_vars):
        counts = hf_data[col].value_counts().sort(col)
        count_series = pl.Series(name=col, values=counts[:, 1]).cast(pl.Float64)
        hf_data_freq.insert_column(index=i, column=count_series)
    return (
        bin_vars,
        cont_vars,
        hf_data_freq,
        hf_data_max,
        hf_data_mean,
        hf_data_median,
        hf_data_min,
    )


@app.cell
def _(cont_vars, hf_data_max, hf_data_mean, hf_data_median, hf_data_min):
    # getting the data ready to be put into table format
    hf_data_stats = pl.concat(
        [hf_data_min, hf_data_max, hf_data_mean, hf_data_median]
    ).transpose()
    hf_data_stats.columns = ["Min", "Max", "Mean", "Median"]
    hf_data_stats = hf_data_stats.with_columns(pl.col("Mean").round(1))
    hf_data_stats.insert_column(index=0, column=pl.Series(values=cont_vars))
    return (hf_data_stats,)


@app.cell
def _(bin_vars, hf_data_freq, hf_data_stats):
    # adding some blank rows so that we can concat the bin and cont vars together
    blank_rows = pl.DataFrame({"column_0": [None] * 7, "column_1": [None] * 7})
    bin = pl.concat(
        [blank_rows.cast(pl.Float64), hf_data_freq.transpose()], how="vertical"
    )
    bin.columns = ["Count 0", "Count 1"]
    blank_rows = pl.DataFrame(
        {
            "": bin_vars,
            "Min": [None] * 6,
            "Max": [None] * 6,
            "Mean": [None] * 6,
            "Median": [None] * 6,
        }
    )
    cont = pl.concat([hf_data_stats, blank_rows], how="vertical")
    return bin, cont


@app.cell
def _(bin, cont):
    from great_tables import GT, style, loc

    table = cont.hstack(bin)
    proper_names = pl.Series(
        "",
        [
            "Age",
            "CPK (mcg/L)",
            "EF",
            "Platelets",
            "SC (mg/dL)",
            "SS (mEQ/L)",
            "Follow-up (days)",
            "Anaemia",
            "Diabetes",
            "HBP",
            "Sex",
            "Smoking",
            "Death",
        ],
    )
    table.replace_column(index=0, column=proper_names)

    (
        GT(table, rowname_col="")
        .sub_missing(table.columns, missing_text="")
        .tab_style(
            style=style.fill(color="#ebeced"),
            locations=loc.body(rows=proper_names[::2].to_list()),
        )
        .fmt_percent(rows=2, scale_values=False, decimals=1)
        .fmt_integer(rows=[3, 6, 7, 8, 9, 10, 11, 12])
        .save("latex/figs/variable_stats_.png")
    )
    return (proper_names,)


@app.cell
def _(hf_data_freq, proper_names):
    # frequencies of each of the binary variables - not included as not really necessary

    x_labels = ["0", "1"]
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(10, 5))
    n = 0
    proper_bin_vars = proper_names.to_list()[7:]
    for rows in range(ax.shape[0]):
        for cols in range(ax.shape[1]):
            ax[rows, cols].bar(x=x_labels, height=hf_data_freq[:, n], color="gray")
            ax[rows, cols].set_title(proper_bin_vars[n], loc="left", size=18)
            ax[rows, cols].tick_params(axis="both", labelsize=12)
            n += 1

    ax[0, 0].set_ylabel("Count", fontsize=14)
    ax[1, 0].set_ylabel("Count", fontsize=14)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(hf_data):
    # calculates the death rates per age group
    bin_edges = np.linspace(40, 100, 7)
    bin_indices = np.digitize(hf_data["age"], bin_edges)
    death_counts = []
    for bin_ in np.unique(bin_indices):
        deaths = hf_data.filter(bin_indices == bin_)["DEATH_EVENT"].sum()
        death_counts.append(deaths / hf_data.filter(bin_indices == bin_).shape[0])
    return (death_counts,)


@app.cell
def _(hf_data):
    # calculates the death rate per sex
    grouped_sex = hf_data.group_by(pl.col("sex"), maintain_order=True)
    male_deaths = (
        grouped_sex.sum()["DEATH_EVENT"][0]
        / hf_data.filter(pl.col("sex") == 1).shape[0]
    )
    female_deaths = (
        grouped_sex.sum()["DEATH_EVENT"][1]
        / hf_data.filter(pl.col("sex") == 0).shape[0]
    )
    return female_deaths, male_deaths


@app.cell
def _(hf_data):
    # calculates the death rate per month of follow up time
    bin_edges_ = np.linspace(0, 300, 11)
    bin_indices_ = np.digitize(hf_data["time"], bin_edges_)
    death_counts_ = []
    for bin__ in np.unique(bin_indices_):
        deaths_ = hf_data.filter(bin_indices_ == bin__)["DEATH_EVENT"].sum()
        death_counts_.append(deaths_ / hf_data.filter(bin_indices_ == bin__).shape[0])
    return (death_counts_,)


@app.cell
def _(hf_data):
    # calculates the death rate per health condition - an example may have more than one of these conditions
    grouped_anaemia = hf_data.group_by(pl.col("anaemia"), maintain_order=True)
    anaemia_deaths = (
        grouped_anaemia.sum().filter(pl.col("anaemia") == 1)["DEATH_EVENT"][0]
        / hf_data.filter(pl.col("anaemia") == 1).shape[0]
    )

    grouped_diabetes = hf_data.group_by(pl.col("diabetes"), maintain_order=True)
    diabetes_deaths = (
        grouped_diabetes.sum().filter(pl.col("diabetes") == 1)["DEATH_EVENT"][0]
        / hf_data.filter(pl.col("diabetes") == 1).shape[0]
    )

    grouped_hbp = hf_data.group_by(pl.col("high_blood_pressure"), maintain_order=True)
    hbp_deaths = (
        grouped_hbp.sum().filter(pl.col("high_blood_pressure") == 1)["DEATH_EVENT"][0]
        / hf_data.filter(pl.col("high_blood_pressure") == 1).shape[0]
    )

    grouped_smoking = hf_data.group_by(pl.col("smoking"), maintain_order=True)
    smoking_deaths = (
        grouped_smoking.sum().filter(pl.col("smoking") == 1)["DEATH_EVENT"][0]
        / hf_data.filter(pl.col("smoking") == 1).shape[0]
    )
    return anaemia_deaths, diabetes_deaths, hbp_deaths, smoking_deaths


@app.cell
def _(
    anaemia_deaths,
    death_counts,
    death_counts_,
    diabetes_deaths,
    female_deaths,
    hbp_deaths,
    male_deaths,
    smoking_deaths,
):
    fig_, ax_ = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    ax_[0, 0].bar(
        x=["[40, 50)", "[50, 60)", "[60, 70)", "[70, 80)", "[80, 90)", "[90, 100)"],
        height=death_counts,
        color="gray",
    )
    ax_[0, 0].set_xlabel("Age group", fontsize=16)
    ax_[0, 0].set_ylabel("Death rate", fontsize=16)
    ax_[0, 0].tick_params(axis="y", labelsize=14)
    ax_[0, 0].tick_params(axis="x", rotation=45, labelsize=12)

    p2 = ax_[0, 1].bar(
        x=["Male", "Female"], height=[male_deaths, female_deaths], color="gray"
    )
    ax_[0, 1].set_xlabel("Sex", fontsize=16)
    ax_[0, 1].bar_label(p2, label_type="center", c="white", fmt="%.3f", fontsize=14)
    ax_[0, 1].tick_params(axis="both", labelsize=14)

    ax_[1, 0].bar(
        x=[
            "[0, 30)",
            "[30, 60)",
            "[60, 90)",
            "[90, 120)",
            "[120, 150)",
            "[150, 180)",
            "[180, 210)",
            "[210, 240)",
            "[240, 270)",
            "[270, 300)",
        ],
        height=death_counts_,
        color="gray",
    )
    ax_[1, 0].tick_params(axis="x", rotation=60, labelsize=12)
    ax_[1, 0].tick_params(axis="y", labelsize=14)
    ax_[1, 0].set_xlabel("Follow-up time (days)", fontsize=16)
    ax_[1, 0].set_ylabel("Death rate", fontsize=16)

    p4 = ax_[1, 1].bar(
        x=["Anaemia", "Diabetes", "HBP", "Smoking"],
        height=[anaemia_deaths, diabetes_deaths, hbp_deaths, smoking_deaths],
        color="gray",
    )
    ax_[1, 1].set_xlabel("Condition", fontsize=16)
    ax_[1, 1].bar_label(p4, label_type="center", c="white", fmt="%.3f", fontsize=14)
    ax_[1, 1].tick_params(axis="both", labelsize=14)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(hf_data, proper_names):
    # Correlation matrix
    cm = np.corrcoef(hf_data.transpose())
    hm = heatmap(
        cm,
        row_names=proper_names.to_list(),
        column_names=proper_names.to_list(),
        figsize=(12, 8),
        cmap=plt.cm.Blues,
    )
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
