# Segment validation and model choice (seg-v1.0)

Outcomes were never model inputs. Answer key not used.

**Chosen: `kproto-k8`** (best mean rank over stability_ari_mean, silhouette_gower, v_conversion, eps2_deal_value, eps2_days_to_close, v_late_payment, distinct_pair_share, emphasis_robustness; the d02 baseline is the benchmark).

## Scorecard

| candidate_id | model | k | eligible | stability_ari_mean | silhouette_gower | v_conversion | eps2_deal_value | eps2_days_to_close | v_late_payment | distinct_pair_share | emphasis_robustness | min_share | mean_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-kmeans-k8 | kmeans | 8 | False | 0.836 | 0.153 | 0.213 | 0.632 | 0.424 | 0.653 | 1.0 |  | 0.03 |  |
| gmm-k7 | gmm | 7 | True | 0.807 | 0.155 | 0.186 | 0.633 | 0.409 | 0.657 | 0.714 | 0.637 | 0.024 | 2.75 |
| kmeans-k9 | kmeans | 9 | True | 0.836 | 0.157 | 0.219 | 0.633 | 0.422 | 0.656 | 0.972 | 0.786 | 0.023 | 2.0 |
| kproto-k8 | kproto | 8 | True | 0.909 | 0.163 | 0.216 | 0.633 | 0.422 | 0.656 | 1.0 | 0.82 | 0.023 | 1.25 |

## Outcome tests

### baseline-kmeans-k8

| outcome | test | population | n | statistic | p_value | effect_size | effect_measure |
|---|---|---|---|---|---|---|---|
| conversion | chi-square | customers with a closed lead | 57682 | 2621.4 | 0.0 | 0.213 | cramers_v |
| deal_value | Kruskal-Wallis | customers with a deal | 14936 | 9441.0 | 0.0 | 0.632 | epsilon_squared |
| days_to_close | Kruskal-Wallis | customers with a closed lead | 57682 | 24450.5 | 0.0 | 0.424 | epsilon_squared |
| late_payment | chi-square | customers with payments due | 14936 | 6377.6 | 0.0 | 0.653 | cramers_v |

### gmm-k7

| outcome | test | population | n | statistic | p_value | effect_size | effect_measure |
|---|---|---|---|---|---|---|---|
| conversion | chi-square | customers with a closed lead | 57682 | 1997.1 | 0.0 | 0.186 | cramers_v |
| deal_value | Kruskal-Wallis | customers with a deal | 14936 | 9456.8 | 0.0 | 0.633 | epsilon_squared |
| days_to_close | Kruskal-Wallis | customers with a closed lead | 57682 | 23600.4 | 0.0 | 0.409 | epsilon_squared |
| late_payment | chi-square | customers with payments due | 14936 | 6438.0 | 0.0 | 0.657 | cramers_v |

### kmeans-k9

| outcome | test | population | n | statistic | p_value | effect_size | effect_measure |
|---|---|---|---|---|---|---|---|
| conversion | chi-square | customers with a closed lead | 57682 | 2756.7 | 0.0 | 0.219 | cramers_v |
| deal_value | Kruskal-Wallis | customers with a deal | 14936 | 9459.3 | 0.0 | 0.633 | epsilon_squared |
| days_to_close | Kruskal-Wallis | customers with a closed lead | 57682 | 24346.1 | 0.0 | 0.422 | epsilon_squared |
| late_payment | chi-square | customers with payments due | 14936 | 6427.5 | 0.0 | 0.656 | cramers_v |

### kproto-k8

| outcome | test | population | n | statistic | p_value | effect_size | effect_measure |
|---|---|---|---|---|---|---|---|
| conversion | chi-square | customers with a closed lead | 57682 | 2689.0 | 0.0 | 0.216 | cramers_v |
| deal_value | Kruskal-Wallis | customers with a deal | 14936 | 9462.6 | 0.0 | 0.633 | epsilon_squared |
| days_to_close | Kruskal-Wallis | customers with a closed lead | 57682 | 24355.1 | 0.0 | 0.422 | epsilon_squared |
| late_payment | chi-square | customers with payments due | 14936 | 6429.6 | 0.0 | 0.656 | cramers_v |

## Actionably distinct segment pairs (chosen candidate)

Practical margins: {'conversion': 3.0, 'late_payment': 2.0, 'deal_value_ratio': 1.25, 'days_to_close': 7.0}; Bonferroni family alpha 0.01.

| segment_a | segment_b | distinct | differs_on |
|---|---|---|---|
| 0 | 1 | True | conversion, days_to_close |
| 0 | 2 | True | conversion, days_to_close |
| 0 | 3 | True | deal_value, days_to_close |
| 0 | 4 | True | conversion, late_payment, deal_value, days_to_close |
| 0 | 5 | True | late_payment |
| 0 | 6 | True | conversion, late_payment, deal_value, days_to_close |
| 0 | 7 | True | conversion, deal_value, days_to_close |
| 1 | 2 | True | conversion, days_to_close |
| 1 | 3 | True | conversion, days_to_close |
| 1 | 4 | True | conversion, late_payment, deal_value, days_to_close |
| 1 | 5 | True | conversion, deal_value, days_to_close |
| 1 | 6 | True | conversion, deal_value, days_to_close |
| 1 | 7 | True | conversion |
| 2 | 3 | True | conversion, deal_value, days_to_close |
| 2 | 4 | True | late_payment, deal_value, days_to_close |
| 2 | 5 | True | conversion, late_payment, deal_value, days_to_close |
| 2 | 6 | True | conversion, late_payment, deal_value, days_to_close |
| 2 | 7 | True | deal_value, days_to_close |
| 3 | 4 | True | conversion, late_payment, deal_value, days_to_close |
| 3 | 5 | True | late_payment, deal_value, days_to_close |
| 3 | 6 | True | conversion, late_payment, deal_value, days_to_close |
| 3 | 7 | True | conversion, days_to_close |
| 4 | 5 | True | conversion, late_payment, deal_value, days_to_close |
| 4 | 6 | True | conversion, late_payment, deal_value |
| 4 | 7 | True | late_payment, deal_value, days_to_close |
| 5 | 6 | True | conversion, deal_value, days_to_close |
| 5 | 7 | True | conversion, late_payment, deal_value, days_to_close |
| 6 | 7 | True | conversion, late_payment, deal_value, days_to_close |

## Agreement between candidates (ARI)

| index | baseline-kmeans-k8 | gmm-k7 | kmeans-k9 | kproto-k8 |
|---|---|---|---|---|
| baseline-kmeans-k8 | 1.0 | 0.4 | 0.6 | 0.73 |
| gmm-k7 | 0.4 | 1.0 | 0.64 | 0.54 |
| kmeans-k9 | 0.6 | 0.64 | 1.0 | 0.81 |
| kproto-k8 | 0.73 | 0.54 | 0.81 | 1.0 |

## Chosen vs runner-up: where customers go

| kproto-k8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 184 | 15 | 0 | 23 | 1 | 175 | 0 | 18046 |
| 1 | 0 | 22343 | 7 | 0 | 135 | 13 | 593 | 0 | 78 |
| 2 | 0 | 93 | 6678 | 0 | 0 | 3 | 16 | 0 | 43 |
| 3 | 0 | 60 | 0 | 0 | 15678 | 1 | 0 | 0 | 4 |
| 4 | 2 | 6030 | 2 | 9391 | 5 | 0 | 277 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 1 | 5 | 2766 | 4150 | 1 |
| 6 | 2271 | 0 | 0 | 13 | 0 | 0 | 24 | 0 | 0 |
| 7 | 0 | 306 | 1 | 0 | 3 | 11047 | 44 | 0 | 1 |
