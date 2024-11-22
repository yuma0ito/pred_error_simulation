// 参考 https://norimune.net/3138

data {
  int I;  // 問題数
  int J;  // 被験者数
  int K;  // カテゴリ数
  array[I] int n_scores;  // 各問題のカテゴリ数
  array[I, J] int y;  // データ
}

transformed data{
  vector[K] c = cumulative_sum(rep_vector(1,K));
}

parameters {
  vector[J] theta;  // 能力値
  vector<lower=0>[I] alpha;  // 識別力
  array[I] vector[K-1] beta;  // 困難度
}

transformed parameters{
  array[I] vector[K] cumsum_beta;
  for (i in 1:I){
    cumsum_beta[i] = cumulative_sum(append_row(0,beta[i]));
  }
}

model {
  alpha ~ lognormal(0, 0.5);
  for(i in 1:I) beta[i] ~ normal(0, 1);
  theta ~ normal(0, 1);

  for (i in 1:I) {
    for (j in 1:J) {
      if (y[i, j] != -1) {
        vector[K] logits = alpha[i] * (c * theta[j] - cumsum_beta[i]);
        logits[n_scores[i]+1:K] = rep_vector(-pow(10, 18), K - n_scores[i]);
        y[i, j] ~ categorical_logit(logits);
      }
    }
  }
}
