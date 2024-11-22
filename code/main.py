import argparse
import logging
import os
import sys
import warnings

import numpy as np
from cmdstanpy import CmdStanModel
from utils import Config, read_data, read_is_missing, save_result

warnings.filterwarnings("ignore")


def main(args):
    # load config
    config_file_path = os.path.join("../data", args.dataset, "config.json")
    config = Config(config_file_path)
    min_scores, max_scores = np.array(config.min_scores), np.array(config.max_scores)
    n_scores = max_scores - min_scores + 1

    # read data
    score_file_path = os.path.join("../data", args.dataset, "score.csv")
    examinees, scores = read_data(score_file_path)
    scores = np.array(scores, dtype=int)

    if args.estimation != "all":
        missing_file_path = os.path.join(
            "../data",
            args.dataset,
            "set",
            args.mask_pattern,
            args.set_dir,
            "is_test.csv",
        )
        _, is_missing = read_is_missing(missing_file_path)
        if args.estimation == "no_imputation":
            scores = np.where(is_missing, np.nan, scores).astype(float)
            error_pattern = None
            scores_with_error = None
        elif args.estimation == "imputation":
            error_pattern_ori = np.round(
                np.random.normal(0, args.sigma, size=scores.shape)
            )
            error_pattern = np.where(
                is_missing,
                np.clip(scores + error_pattern_ori, min_scores, max_scores) - scores,
                np.zeros_like(scores),
            )
            scores_with_error = scores + error_pattern
            scores = scores_with_error.copy()
        else:
            raise ValueError
    else:
        error_pattern = None
        scores_with_error = None

    scores = scores - min_scores + 1

    stan_scores = np.where(np.isnan(scores), -1, scores).astype(int)
    stan_data = {
        "I": config.n_item,
        "J": config.n_examinee,
        "K": n_scores.max(),
        "n_scores": n_scores,
        "y": stan_scores.T,  # (n_item, n_examinee)
    }

    stan_file = "stan/gpcm_decomp.stan" if args.decomp else "stan/gpcm.stan"
    stan_model = CmdStanModel(stan_file=stan_file)

    fit = stan_model.sample(
        data=stan_data,
        chains=args.n_chains,
        iter_sampling=args.n_sampling,
        iter_warmup=args.n_warmup,
        thin=args.thin,
    )

    # get alpha, beta, theta estimation results
    theta_samples = fit.stan_variable("theta")
    alpha_samples = fit.stan_variable("alpha")
    beta_samples = fit.stan_variable("beta")
    if args.decomp:
        d_samples = fit.stan_variable("d")

    theta_mean, theta_std = theta_samples.mean(axis=0), theta_samples.std(axis=0)
    alpha_mean, alpha_std = alpha_samples.mean(axis=0), alpha_samples.std(axis=0)
    beta_mean, beta_std = beta_samples.mean(axis=0), beta_samples.std(axis=0)
    if args.decomp:
        d_mean, d_std = d_samples.mean(axis=0), d_samples.std(axis=0)
    else:
        d_mean, d_std = None, None

    # save result
    output_dir = "output_decomp" if args.decomp else "output"
    if args.estimation == "all":
        output_path = os.path.join("..", output_dir, args.dataset, "all")
    else:
        output_path = os.path.join(
            "..",
            output_dir,
            args.dataset,
            args.mask_pattern,
            args.set_dir,
            args.estimation,
        )
        if args.estimation == "imputation":
            output_path = os.path.join(
                output_path,
                f"error_std={args.sigma:.2f}",
                f"error{str(args.error_num).zfill(3)}",
            )
    os.makedirs(output_path, exist_ok=False)
    save_result(
        is_decomp=args.decomp,
        output_path=output_path,
        args=args,
        n_item=config.n_item,
        n_examinee=config.n_examinee,
        n_score=max(n_scores),
        examinees=examinees,
        fit=fit,
        theta_mean=theta_mean,
        theta_std=theta_std,
        alpha_mean=alpha_mean,
        alpha_std=alpha_std,
        beta_mean=beta_mean,
        beta_std=beta_std,
        d_mean=d_mean,
        d_std=d_std,
        error_pattern=error_pattern,
        scores_with_error=scores_with_error,
    )


if __name__ == "__main__":
    stream_hundler = logging.StreamHandler(sys.stdout)
    stream_hundler.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)8s - %(name)s: %(message)s",
        handlers=[stream_hundler],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "benesse",
            "esjp2019",
            "esjp2019_sum",
            "gsk2021b",
            "gsk2021b_sum",
            "elyza100",
        ],
        help="Dataset name",
    )
    parser.add_argument(
        "--estimation",
        type=str,
        required=True,
        choices=[
            "all",
            "no_imputation",
            "imputation",
        ],
        help="Estimation method",
    )
    parser.add_argument(
        "--mask_pattern",
        type=str,
        required=True,
        choices=["mask_one_item", "mask_half_item", "not_mask_train", "mask_10%_data"],
        help="masking method",
    )
    parser.add_argument(
        "--n_chains", type=int, default=4, help="Number of chains for Stan sampling"
    )
    parser.add_argument(
        "--n_sampling",
        type=int,
        default=1000,
        help="Number of sampling for Stan sampling",
    )
    parser.add_argument(
        "--n_warmup",
        type=int,
        default=1000,
        help="Number of warmup for Stan sampling",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Thin for Stan sampling",
    )
    if parser.parse_known_args()[0].estimation != "all":
        parser.add_argument(
            "--set_dir", type=str, required=True, help="Set directory name"
        )
    if parser.parse_known_args()[0].estimation == "imputation":
        parser.add_argument(
            "--sigma",
            type=float,
            required=True,
            help="Random seed. if you set -1, it will be not fixed.",
        )
        parser.add_argument(
            "--error_num",
            type=str,
            required=True,
            help="error namber",
        )
    parser.add_argument(
        "--decomp",
        action="store_true",
        default=False,
        help="Whether to decompose step parameters",
    )
    args = parser.parse_args()

    main(args)
