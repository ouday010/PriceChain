import numpy as np
import pandas as pd
import joblib
from hedera import (
    Client, PrivateKey, TokenCreateTransaction, TokenType,
    TokenMintTransaction, Hbar, AccountId, AccountBalanceQuery,
    AccountInfoQuery
)
from jnius import autoclass  # ADD THIS
import time

# ADD THIS LINE
Duration = autoclass('java.time.Duration')

def predict_price(data):
    data_path = 'data/'
    # Load models
    xgb_model = joblib.load(data_path + 'xgboost_final_model.pkl')
    cat_model = joblib.load(data_path + 'catboost_final_model.pkl')
    rf_model = joblib.load(data_path + 'rf_final_model.pkl')
    preprocessor = joblib.load(data_path + 'preprocessor_final.pkl')
    feature_indices = joblib.load(data_path + 'selected_feature_indices.pkl')
    ensemble_weights = joblib.load(data_path + 'ensemble_weights.pkl')

    input_data = pd.DataFrame([data])

    # Feature engineering
    input_data['bedrooms_area_marla'] = input_data['bedrooms'] * input_data['area_marla']
    input_data['baths_area_marla'] = input_data['baths'] * input_data['area_marla']
    input_data['lat_lon_interaction'] = input_data['latitude'] * input_data['longitude']

    city_centers = {'Lahore': (31.5204, 74.3587), 'Karachi': (24.8607, 67.0011), 'Islamabad': (33.6844, 73.0479)}
    def calculate_distance(row):
        if row['city'] in city_centers:
            lat_c, lon_c = city_centers[row['city']]
            return np.sqrt((row['latitude'] - lat_c)**2 + (row['longitude'] - lon_c)**2)
        return 0.0
    input_data['distance_to_center'] = input_data.apply(calculate_distance, axis=1)

    kmeans = joblib.load(data_path + 'kmeans_model.pkl')
    input_data['location_cluster'] = kmeans.predict(input_data[['latitude', 'longitude']])
    poly = joblib.load(data_path + 'poly_transformer.pkl')
    poly_cols = ['area_marla', 'bedrooms']
    poly_features = poly.transform(input_data[poly_cols])
    poly_names = poly.get_feature_names_out(poly_cols)
    for i, name in enumerate(poly_names):
        input_data[f'poly_{name}'] = poly_features[:, i]

    numerical_cols = [
        'latitude', 'longitude', 'baths', 'area_marla', 'bedrooms',
        'year', 'month', 'day',
        'bedrooms_area_marla', 'baths_area_marla',
        'lat_lon_interaction', 'distance_to_center'
    ]
    numerical_cols.extend([f'poly_{name}' for name in poly_names])
    for col in numerical_cols:
        input_data[col] = input_data[col].replace([np.inf, -np.inf], np.nan).fillna(input_data[col].median())

    X = preprocessor.transform(input_data)
    X_filtered = X[:, feature_indices]

    y_pred_log = (
        ensemble_weights[0] * xgb_model.predict(X_filtered) +
        ensemble_weights[1] * cat_model.predict(X_filtered) +
        ensemble_weights[2] * rf_model.predict(X_filtered)
    )
    predicted_price = np.expm1(y_pred_log)[0]

    # === HEDERA: NO METADATA, NO JVM ERRORS ===
    client = Client.forTestnet()
    client.setRequestTimeout(Duration.ofSeconds(300))  # FIXED LINE

    operator_account_id = AccountId.fromString("0.0.7146824")
    operator_private_key = PrivateKey.fromString("0x8e707f85ac5950ef20ee482842800d957526f2cdba118d09ee1e5fcf930e6c0f")
    client.setOperator(operator_account_id, operator_private_key)

    print(f"   Operator set to: {operator_account_id}")

    # Verify connection
    try:
        balance = AccountBalanceQuery().setAccountId(operator_account_id).execute(client)
        print(f"   Balance: {balance.hbars} HBAR")
        info = AccountInfoQuery().setAccountId(operator_account_id).execute(client)
        print("   Key verification passed")
    except Exception as e:
        print(f"   Connection failed: {e}")
        return predicted_price, None, None

    token_id = None
    nft_serial = None

    try:
        # Create Token with price in name
        token_name = f"Deed-{predicted_price:,.0f}PKR"
        token_tx = TokenCreateTransaction() \
            .setTokenName(token_name) \
            .setTokenSymbol("PDNFT") \
            .setTokenType(TokenType.NON_FUNGIBLE_UNIQUE) \
            .setDecimals(0) \
            .setInitialSupply(0) \
            .setTreasuryAccountId(operator_account_id) \
            .setAdminKey(operator_private_key) \
            .setSupplyKey(operator_private_key) \
            .setMaxTransactionFee(Hbar(20))

        token_tx = token_tx.freezeWith(client).sign(operator_private_key)
        response = token_tx.execute(client)
        receipt = response.getReceipt(client)
        token_id = receipt.tokenId
        print(f"   Token created: {token_id}")

        # MINT NFT — NO METADATA = NO JVM ERRORS
        mint_tx = TokenMintTransaction() \
            .setTokenId(token_id) \
            .setMaxTransactionFee(Hbar(2))

        mint_tx = mint_tx.freezeWith(client).sign(operator_private_key)
        mint_response = mint_tx.execute(client)
        mint_receipt = mint_response.getReceipt(client)
        nft_serial = mint_receipt.serials[0]
        print(f"   Minted NFT with Serial: {nft_serial}")

    except Exception as e:
        print(f"   NFT minting failed: {e}")

    return predicted_price, token_id, nft_serial