import streamlit as st
import pandas as pd
import numpy as np

# Updated global covariance matrix
# This is the predefined covariance matrix used for portfolio risk calculations
assets = ['USA', 'World ex USA', 'EM', 'AGG']
USD_cov = pd.DataFrame(
    data=[
        [0.02350089, 0.022129004029219898, 0.023322444267672212, -0.0011591943593256455],
        [0.022129004029219898, 0.02745649, 0.029032569512265394, -0.0008177948272787588],
        [0.023322444267672212, 0.029032569512265394, 0.04247721, -0.001007035988371974],
        [-0.0011591943593256455, -0.0008177948272787588, -0.001007035988371974, 0.00226576],
    ],
    index=assets,
    columns=assets
)

def portfolio_vol(cov_matrix, weights):
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(portfolio_variance)

# --- CALCULATE CHANGE IN PORTFOLIO VOLATILITY AND TRACKING ERROR FOR VARIOUS US OVERWEIGHTS ---
# This function simulates portfolio changes under incremental US overweight scenarios

def analyze_us_overweight_impact(acwi_weight, us_weight_in_acwi):
    def get_acwi_geo_weights(us_weight):
        rest = 1 - us_weight
        return np.array([us_weight, rest * 0.7, rest * 0.3])

    def tracking_error(cov_matrix, w1, w2):
        diff = w1 - w2
        return np.sqrt(np.dot(diff.T, np.dot(cov_matrix, diff)))

    overweight_steps = np.arange(0, 0.06, 0.005)
    base_acwi_geo_weights = get_acwi_geo_weights(us_weight_in_acwi)
    base_portfolio_weights = pd.Series(
        data=np.append(base_acwi_geo_weights * acwi_weight, 1 - acwi_weight),
        index=assets
    )

    cov = USD_cov.loc[assets, assets]
    base_vol = portfolio_vol(cov, base_portfolio_weights.values)

    results = []
    for ow in overweight_steps:
        new_us_weight = us_weight_in_acwi + ow
        new_acwi_geo_weights = get_acwi_geo_weights(new_us_weight)
        new_portfolio_weights = pd.Series(
            data=np.append(new_acwi_geo_weights * acwi_weight, 1 - acwi_weight),
            index=assets
        )

        new_vol = portfolio_vol(cov, new_portfolio_weights.values)
        vol_change = new_vol - base_vol
        te = tracking_error(cov, new_portfolio_weights.values, base_portfolio_weights.values)

        results.append({
            'US Overweight': ow * 100,
            'Volatility Change': vol_change * 100,
            'Tracking Error': te * 100
        })

    result_df = pd.DataFrame(results).set_index('US Overweight').round(2)
    return result_df

# This function returns the optimal US overweight and the corresponding volatility change and tracking error.
def get_optimal_us_overweight(ow_results, max_tracking_error_threshold, min_volatility_reduction_threshold):
    max_tracking_error_threshold *= 100
    min_volatility_reduction_threshold *= 100
    optimal_us_overweight = ow_results[(ow_results['Tracking Error'] <= max_tracking_error_threshold) & (ow_results['Volatility Change'] <= min_volatility_reduction_threshold)]
    if not optimal_us_overweight.empty:
        optimal_us_overweight = optimal_us_overweight.loc[optimal_us_overweight['Volatility Change'].idxmin()]
    return optimal_us_overweight.name, optimal_us_overweight['Volatility Change'], optimal_us_overweight['Tracking Error']

# This function loops through multiple ACWI and US-in-ACWI weight combinations
# to generate matrices for optimal overweight, volatility changes, and tracking error
def generate_optimal_us_overweight_matrix(max_tracking_error_threshold, min_volatility_reduction_threshold):
    acwi_weights = np.arange(0.5, 0.91, 0.05)
    us_weights_in_acwi = np.arange(0.5, 0.91, 0.05)

    acwi_labels = (acwi_weights * 100).round(0)
    us_labels = (us_weights_in_acwi * 100).round(0)

    optimal_ow_matrix = pd.DataFrame(index=us_labels, columns=acwi_labels)
    vol_change_matrix = pd.DataFrame(index=us_labels, columns=acwi_labels)
    tracking_error_matrix = pd.DataFrame(index=us_labels, columns=acwi_labels)

    for acwi_weight in acwi_weights:
        for us_weight in us_weights_in_acwi:
            try:
                ow_results = analyze_us_overweight_impact(acwi_weight, us_weight)
                optimal_ow, vol_change, te = get_optimal_us_overweight(
                    ow_results, max_tracking_error_threshold, min_volatility_reduction_threshold
                )
                optimal_ow_matrix.at[round(us_weight * 100), round(acwi_weight * 100)] = round(optimal_ow, 2)
                vol_change_matrix.at[round(us_weight * 100), round(acwi_weight * 100)] = round(vol_change, 2)
                tracking_error_matrix.at[round(us_weight * 100), round(acwi_weight * 100)] = round(te, 2)
            except Exception:
                optimal_ow_matrix.at[round(us_weight * 100), round(acwi_weight * 100)] = np.nan
                vol_change_matrix.at[round(us_weight * 100), round(acwi_weight * 100)] = np.nan
                tracking_error_matrix.at[round(us_weight * 100), round(acwi_weight * 100)] = np.nan

    for df in [optimal_ow_matrix, vol_change_matrix, tracking_error_matrix]:
        df.index.name = 'US Weight in ACWI (%)'
        df.columns.name = 'ACWI Weight in GP (%)'

    return optimal_ow_matrix, vol_change_matrix, tracking_error_matrix

# --- Streamlit User Interface ---
st.title("US Overweight Optimization App")

max_te = st.number_input("Max Tracking Error Threshold (%)", min_value=0.0, value=5.0, step=0.1)
min_vol_red = st.number_input("Min Volatility Reduction Threshold (%)", value=0.0, step=0.1, format="%0.2f")

if st.button("Generate Matrices"):
    optimal_ow_matrix, vol_change_matrix, tracking_error_matrix = generate_optimal_us_overweight_matrix(max_te / 100, min_vol_red / 100)

    st.subheader("Optimal US Overweight Matrix (%)")
    st.dataframe(optimal_ow_matrix)

    st.subheader("Volatility Change Matrix (%)")
    st.dataframe(vol_change_matrix)

    st.subheader("Tracking Error Matrix (%)")
    st.dataframe(tracking_error_matrix)
