from predict import predict_price

input_data = {
    'city': 'Lahore',
    'latitude': 31.5204,
    'longitude': 74.3587,
    'baths': 3,
    'bedrooms': 5,
    'area_marla': 10.0,
    'year': 2025,
    'month': 10,
    'day': 29,
    'locality': 'DHA',
    'purpose': 'For Sale',
    'location': 'DHA Phase 5',
    'property_type': 'House'
}

print("STARTING PREDICTION + NFT MINT...")
price, token_id, nft_serial = predict_price(input_data)

print("\n" + "="*50)
print("FINAL RESULT:")
print(f"Predicted Price: {price:,.2f} PKR" if price else "Price: FAILED")
print(f"NFT Token ID: {token_id}")
print(f"NFT Serial: {nft_serial}")
print("="*50)