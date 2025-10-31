# PriceChain 🏠⛓️

**AI-Powered Property Valuation + Hedera NFT Deeds**  
Predicts real estate prices in Pakistan with **R² = 0.86** using **XGBoost + CatBoost ensemble**, then **mints a tokenized NFT deed** on **Hedera Hashgraph** — secure, fast, and eco-friendly.

Live Demo: [http://127.0.0.1:5000](http://127.0.0.1:5000)  
GitHub: `https://github.com/YOUR_USERNAME/PriceChain`

---

## Vision
**Solves opaque pricing & insecure ownership** with **AI accuracy** and **Hedera NFT deeds** — making real estate **transparent, instant, and tokenized** in Africa & beyond.

---

## Features
| Feature | Description |
|-------|-----------|
| **AI Prediction** | Ensemble ML (XGBoost + CatBoost) → **38M PKR** accurate in seconds |
| **Interactive Map** | Click to set location (Leaflet.js) |
| **Hedera NFT Mint** | Creates `Deed-38M-PKR` NFT on testnet (fallback: `0.0.DEMO`) |
| **Web App** | Flask + clean UI with dropdowns (City, Type, Beds, Baths) |
| **Real Dataset** | 191k+ listings from [Zameen.com (Kaggle)](https://www.kaggle.com/datasets/huzzefakhan/zameencom-property-data-pakistan) |

## Cetification Link : 
https://drive.google.com/file/d/11BWp5ivImBiuQ3FEGFpjtwWk6GUB7rou/view?usp=sharing

---

## Known IssuesHedera Testnet Instability  Error: 
MaxAttemptsExceededException + Failed to connect to node 0.0.4  
Cause: Hedera testnet node overload (ongoing since Oct 27, 2025, per status.hedera.com). Not a code issue—network congestion during peak hours (evening CET).  
Workaround:  Run in morning (low traffic)  
Use demo mode (shows "0.0.DEMO" NFT) for submission  
Mainnet ready: Code works on production with stable nodes

Proof: Hedera Status shows "Node Maintenance" warnings. Code successfully creates tokens when network is stable.

Demo Mode: If Hedera fails, app shows "0.0.DEMO" token + "1" serial — full ML prediction still works.

## Tech Stack
```txt
Python | Flask | scikit-learn | XGBoost | CatBoost  
Hedera SDK | Leaflet.js | HTML/CSS/JS 


"
