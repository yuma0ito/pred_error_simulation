import json
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Config:
    def __init__(self, config_file_path):
        with open(config_file_path, mode="r", encoding="utf-8") as f:
            json_data = json.load(f)

        self.n_item = json_data["n_item"]
        self.n_examinee = json_data["n_examinee"]
        self.min_scores = json_data["min_scores"]
        self.max_scores = json_data["max_scores"]


def read_data(score_file_path):
    df = pd.read_csv(score_file_path)
    examinees = df["examinee"].astype(str).tolist()
    scores = df.iloc[:, 1:].fillna(-1).values.tolist()
    return examinees, scores


def read_is_missing(missing_file_path):
    df = pd.read_csv(missing_file_path)
    examinees = df["examinee"].tolist()
    is_missing = df.iloc[:, 1:].applymap(lambda x: x == 1).values.tolist()
    return examinees, is_missing


def save_result(
    output_path,
    args,
    n_item,
    n_examinee,
    n_score,
    n_warmup,
    examinees,
    fit,
    theta_mean,
    theta_std,
    alpha_mean,
    alpha_std,
    beta_mean,
    beta_std,
    error_pattern=None,
    scores_with_error=None,
):
    summary = fit.summary()["R_hat"]
    r_hat = summary[
        summary.index.to_series().str.contains(r"^(theta|alpha|beta)(\[\d+(,\d+)*\])?$")
    ]
    fit_data = az.from_cmdstanpy(fit)

    # examinee_params.csv
    df = pd.DataFrame(
        np.stack((np.array(examinees), theta_mean, theta_std), axis=1),
        columns=["examinee", "theta", "std"],
    )
    df["theta"] = df["theta"].astype(float)
    df["std"] = df["std"].astype(float)
    df.to_csv(
        os.path.join(output_path, "examinee_params.csv"),
        float_format="%.5f",
        index=None,
    )

    # item_params.csv
    df = pd.DataFrame(
        np.concatenate(
            (
                np.arange(1, n_item + 1).reshape(-1, 1),
                alpha_mean.reshape(-1, 1),
                beta_mean,
            ),
            axis=1,
        ),
        columns=["item", "alpha"] + [f"beta{i+1}" for i in range(n_score - 1)],
    )
    df["item"] = df["item"].astype(int)
    df.to_csv(
        os.path.join(output_path, "item_params.csv"),
        float_format="%.5f",
        index=None,
    )

    # item_params_std.csv
    df = pd.DataFrame(
        np.concatenate(
            (
                np.arange(1, n_item + 1).reshape(-1, 1),
                alpha_std.reshape(-1, 1),
                beta_std,
            ),
            axis=1,
        ),
        columns=["item", "alpha"] + [f"beta{i+1}" for i in range(n_score - 1)],
    )
    df["item"] = df["item"].astype(int)
    df.to_csv(
        os.path.join(output_path, "item_params_std.csv"),
        float_format="%.5f",
        index=None,
    )

    if args.estimation == "imputation":
        # error pattern
        df = pd.DataFrame(
            np.concatenate(
                (
                    np.array(examinees).reshape(-1, 1),
                    error_pattern,
                ),
                axis=1,
            ),
            columns=["examinee"] + [f"item{i+1}" for i in range(n_item)],
            dtype=float,
        )
        for i in range(n_item):
            df[f"item{i+1}"] = df[f"item{i+1}"].astype(int)
        df.to_csv(
            os.path.join(output_path, "error_pattern.csv"),
            index=None,
        )

        df = pd.DataFrame(
            np.concatenate(
                (
                    np.array(examinees).reshape(-1, 1),
                    scores_with_error,
                ),
                axis=1,
            ),
            columns=["examinee"] + [f"item{i+1}" for i in range(n_item)],
            dtype=float,
        )
        for i in range(n_item):
            df[f"item{i+1}"] = df[f"item{i+1}"].astype(int)
        df.to_csv(
            os.path.join(output_path, "scores_with_error.csv"),
            index=None,
        )

    # convergence check
    convergence_dir_path = os.path.join(output_path, "convergence_check")
    os.makedirs(convergence_dir_path, exist_ok=False)

    # theta autocorrelation
    autocorr_theta = fit_data.posterior["theta"].sel(
        theta_dim_0=slice(0, min(n_examinee, 10))
    )
    az.plot_autocorr(autocorr_theta)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "autocorrelation_plot_theta.png"))
    plt.close()

    # alpha autocorrelation
    autocorr_alpha = fit_data.posterior["alpha"].sel(
        alpha_dim_0=slice(0, min(n_item, 10))
    )
    az.plot_autocorr(autocorr_alpha)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "autocorrelation_plot_alpha.png"))
    plt.close()

    # beta autocorrelation
    autocorr_beta = fit_data.posterior["beta"].sel(
        beta_dim_0=slice(0, n_item),
        beta_dim_1=slice(0, n_score - 1),
    )
    az.plot_autocorr(autocorr_beta)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "autocorrelation_plot_beta.png"))
    plt.close()

    # trace
    coords = {
        "theta_dim_0": range(min(n_examinee, 10)),
        "alpha_dim_0": range(min(n_item, 10)),
        "beta_dim_0": range(min(n_item, 1)),
        "beta_dim_1": range(min(n_score - 1, 10)),
    }
    az.plot_trace(fit_data, var_names=["theta", "alpha", "beta"], coords=coords)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "trace_plot.png"))
    plt.close()

    # Rhat
    r_hat.to_csv(
        os.path.join(convergence_dir_path, "r_hat.csv"),
        index_label="parameter",
        float_format="%.5f",
    )

    # log
    log_dir_path = os.path.join(output_path, "log")
    os.makedirs(log_dir_path, exist_ok=False)
    fit.save_csvfiles(dir=log_dir_path)

    # metadata file
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)


def save_result2(
    output_path,
    args,
    n_item,
    n_examinee,
    n_score,
    n_warmup,
    examinees,
    fit,
    theta_mean,
    theta_std,
    alpha_mean,
    alpha_std,
    beta_mean,
    beta_std,
    d_mean,
    d_std,
    error_pattern=None,
    scores_with_error=None,
):
    summary = fit.summary()["R_hat"]
    r_hat = summary[
        summary.index.to_series().str.contains(
            r"^(theta|alpha|beta|d)(\[\d+(,\d+)*\])?$"
        )
    ]
    fit_data = az.from_cmdstanpy(fit)

    # examinee_params.csv
    df = pd.DataFrame(
        np.stack((np.array(examinees), theta_mean, theta_std), axis=1),
        columns=["examinee", "theta", "std"],
    )
    df["theta"] = df["theta"].astype(float)
    df["std"] = df["std"].astype(float)
    df.to_csv(
        os.path.join(output_path, "examinee_params.csv"),
        float_format="%.5f",
        index=None,
    )

    # item_params.csv
    df = pd.DataFrame(
        np.concatenate(
            (
                np.arange(1, n_item + 1).reshape(-1, 1),
                alpha_mean.reshape(-1, 1),
                beta_mean.reshape(-1, 1),
                d_mean,
            ),
            axis=1,
        ),
        columns=["item", "alpha", "beta"] + [f"d{i+1}" for i in range(n_score - 1)],
    )
    df["item"] = df["item"].astype(int)
    df.to_csv(
        os.path.join(output_path, "item_params.csv"),
        float_format="%.5f",
        index=None,
    )

    # item_params_std.csv
    df = pd.DataFrame(
        np.concatenate(
            (
                np.arange(1, n_item + 1).reshape(-1, 1),
                alpha_std.reshape(-1, 1),
                beta_std.reshape(-1, 1),
                d_std,
            ),
            axis=1,
        ),
        columns=["item", "alpha", "beta"] + [f"d{i+1}" for i in range(n_score - 1)],
    )
    df["item"] = df["item"].astype(int)
    df.to_csv(
        os.path.join(output_path, "item_params_std.csv"),
        float_format="%.5f",
        index=None,
    )

    if args.estimation == "imputation":
        # error pattern
        df = pd.DataFrame(
            np.concatenate(
                (
                    np.array(examinees).reshape(-1, 1),
                    error_pattern,
                ),
                axis=1,
            ),
            columns=["examinee"] + [f"item{i+1}" for i in range(n_item)],
            dtype=float,
        )
        for i in range(n_item):
            df[f"item{i+1}"] = df[f"item{i+1}"].astype(int)
        df.to_csv(
            os.path.join(output_path, "error_pattern.csv"),
            index=None,
        )

        df = pd.DataFrame(
            np.concatenate(
                (
                    np.array(examinees).reshape(-1, 1),
                    scores_with_error,
                ),
                axis=1,
            ),
            columns=["examinee"] + [f"item{i+1}" for i in range(n_item)],
            dtype=float,
        )
        for i in range(n_item):
            df[f"item{i+1}"] = df[f"item{i+1}"].astype(int)
        df.to_csv(
            os.path.join(output_path, "scores_with_error.csv"),
            index=None,
        )

    # convergence check
    convergence_dir_path = os.path.join(output_path, "convergence_check")
    os.makedirs(convergence_dir_path, exist_ok=False)

    # theta autocorrelation
    autocorr_theta = fit_data.posterior["theta"].sel(
        theta_dim_0=slice(0, min(n_examinee, 10))
    )
    az.plot_autocorr(autocorr_theta)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "autocorrelation_plot_theta.png"))
    plt.close()

    # alpha autocorrelation
    autocorr_alpha = fit_data.posterior["alpha"].sel(
        alpha_dim_0=slice(0, min(n_item, 10))
    )
    az.plot_autocorr(autocorr_alpha)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "autocorrelation_plot_alpha.png"))
    plt.close()

    # beta autocorrelation
    autocorr_beta = fit_data.posterior["beta"].sel(beta_dim_0=slice(0, min(n_item, 10)))
    az.plot_autocorr(autocorr_beta)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "autocorrelation_plot_beta.png"))
    plt.close()

    # d autocorrelation
    autocorr_d = fit_data.posterior["d"].sel(
        d_dim_0=slice(0, n_item),
        d_dim_1=slice(0, n_score - 1),
    )
    az.plot_autocorr(autocorr_d)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "autocorrelation_plot_d.png"))
    plt.close()

    # trace
    coords = {
        "theta_dim_0": range(min(n_examinee, 10)),
        "alpha_dim_0": range(min(n_item, 10)),
        "beta_dim_0": range(min(n_item, 10)),
        "d_dim_0": range(min(n_item, 1)),
        "d_dim_1": range(min(n_score - 1, 10)),
    }
    az.plot_trace(fit_data, var_names=["theta", "alpha", "beta", "d"], coords=coords)
    plt.tight_layout()
    plt.savefig(os.path.join(convergence_dir_path, "trace_plot.png"))
    plt.close()

    # Rhat
    r_hat.to_csv(
        os.path.join(convergence_dir_path, "r_hat.csv"),
        index_label="parameter",
        float_format="%.5f",
    )

    # log
    log_dir_path = os.path.join(output_path, "log")
    os.makedirs(log_dir_path, exist_ok=False)
    fit.save_csvfiles(dir=log_dir_path)

    # metadata file
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, mode="w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)
