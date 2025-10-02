# SABR + Copula Multi-Asset Pricing

A practical pricing toolkit that calibrates **SABR** smiles per asset and couples them via a **t-Copula** to simulate joint terminals/paths, price basket options, range accrual, and snowball structures.

- **SABR calibration** with robust loss, vega weighting, curvature regularization, 3-fold CV.  (see `sabr.py`)
- **Copula fitting & model selection** among Gaussian / t / Clayton / Frank / Gumbel with MLE and AIC/BIC comparison. (see `copula.py`)
- **Pricing & Greeks** using SABR-ATM marginals + t-Copula for joint terminal and GBM path simulation (basket call, range accrual, snowball). (see `pricing.py`)

## Environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) SABR calibration

Prepare your option dataset under `./data`. Run:

```bash
python sabr.py --input ./data/your_option_file.xlsx --outdir ./sabr_final
```

**Output:** `./sabr_final/sabr_params.csv` with columns:
`symbol, T, F, alpha, beta, rho, nu` (per maturity)

> Highlights: robust Huber loss, vega weighting, curvature penalty, 3-fold CV, adaptive beta grid.

## 2) Copula fitting & comparison (two modes)

### A) Online data (default, AkShare)
Fetch CN indices/ETF, compute returns, PIT transform, then MLE-fit multiple copulas with AIC/BIC comparison:

```bash
python copula.py
```

**Output** under `./copula_out/`:
- `copula_compare_aic.csv`
- If t-copula is selected, saves `t_copula_params.npz (R, nu)` and tail-dependence text.

### B) Local CSV mode (no network)
If you can't use AkShare, provide your own return CSVs and run:

```bash
python copula_local.py --files ./data/returns_510500.csv ./data/returns_000852.csv ./data/returns_000300.csv --names 510500 000852 000300 --outdir ./copula_out
```

Each CSV should contain a single column named `ret` with daily returns.

## 3) Pricing with SABR marginals + t-Copula

Given `sabr_final/sabr_params.csv` and `copula_out/`, run:

```bash
python pricing.py ^
  --sabr_csv ./sabr_final/sabr_params.csv ^
  --copula_dir ./copula_out ^
  --symbols 510500 000852 000300 ^
  --outdir ./pricing_out ^
  --allow_ko_after_ki ^
  --N_terminal 120000 --N_paths 60000
```

**What it does:**
- Build ATM marginal distributions via Hagan ATM approximation and interpolation.
- Load t-Copula params (`R, nu`) from `copula_out` or fallback defaults.
- Joint terminal sampling and path GBM simulation under t-Copula.
- Price **basket call** (ATM strike), compute **Greeks** (Delta/Gamma/Rho vs basket), price **Range Accrual** (monthly), and **Snowball (worst-of)** with daily KI / monthly KO.

**Key flags:**
- `--ko_level --ki_level --coupon_pa --ko_freq {monthly,daily} --months`
- `--range_lower --range_upper --range_coupon_pa`
- `--copula_R_default --copula_nu_default` as safe fallback.

**Outputs:** CSVs & charts under `./pricing_out`.

## Push to GitHub (Windows/PowerShell)

1) Initialize and commit:
```powershell
git init
git add .
git commit -m "Initial commit: SABR + t-Copula pricing toolkit"
```

2) Link and push via HTTPS (recommended initially):
```powershell
git remote add origin https://github.com/<your-username>/sabr-copula-pricing.git
git branch -M main
git push -u origin main
```
Use a Personal Access Token (classic, scope: `repo`) as the password when prompted.

---

**Notes**
- No GPU required.
- If AkShare fails (network/proxy), use `copula_local.py` with your own CSV returns.
- Directory placeholders `.keep` files are included so Git tracks empty folders.
