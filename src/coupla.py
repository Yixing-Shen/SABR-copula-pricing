"""
三资产 Copula 比较：Gaussian / t / Clayton / Frank / Gumbel
- 数据：仅使用 AkShare（指数&ETF）
- 边际：默认 经验分布 PIT（稳健）；可切换 GARCH(1,1)+正态PIT
- Copula：最大似然（MLE）+ AIC/BIC 对比；t Copula 计算尾依赖
- 可视化：PIT 散点；输出：对比表 CSV、情景模拟 CSV

依赖：
  pip install numpy pandas scipy matplotlib statsmodels arch akshare
Author: 沈依幸
Date: 2025-08-26
"""
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import timedelta
from typing import Tuple, Dict

from scipy.stats import norm, t as student_t, kendalltau
from scipy.optimize import minimize

# 只用到 logpdf/ppf 的接口，避免 statsmodels 版本差异
from statsmodels.distributions.copula.api import (
    GaussianCopula, ClaytonCopula, FrankCopula, GumbelCopula
)

# 选用 GARCH 时用到
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# ========== 用户配置 ==========
OUT_DIR        = "copula_out"

DATE_COL       = "date"
COL_510500     = "510500"
COL_000852     = "000852"
COL_000300     = "000300"

# 过去一年的时间窗：自动按数据最后一天回溯365天
LAST_YEAR_ONLY = True
START_DATE, END_DATE = None, None

# 边际选择
MARGINAL_MODE  = "empirical"   # "empirical" 或 "garch"
WINSOR_Q       = (0.01, 0.99)  # 收益 winsor 分位
# ============================


# ---------- 数据获取 ----------
def try_load_prices() -> pd.DataFrame:
    """使用 AkShare 抓取 000300、000852 指数与 510500 ETF 收盘价。"""
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError(
            "未安装 akshare 或导入失败。请先 `pip install akshare` 再运行。"
        ) from e

    print("[info] using akshare to fetch data...")
    try:
        idx_300 = ak.index_zh_a_hist(symbol="000300", period="daily",
                                     start_date="20000101", end_date="21000101")
        idx_852 = ak.index_zh_a_hist(symbol="000852", period="daily",
                                     start_date="20000101", end_date="21000101")
        etf_500 = ak.fund_etf_hist_em(symbol="510500", period="daily",
                                      start_date="20000101", end_date="21000101")

        idx_300 = idx_300.rename(columns={"日期": "date", "收盘": "000300"})[["date", "000300"]]
        idx_852 = idx_852.rename(columns={"日期": "date", "收盘": "000852"})[["date", "000852"]]
        if "收盘价" in etf_500.columns:
            etf_500 = etf_500.rename(columns={"日期": "date", "收盘价": "510500"})[["date", "510500"]]
        else:
            etf_500 = etf_500.rename(columns={"日期": "date", "收盘": "510500"})[["date", "510500"]]

        for d in (idx_300, idx_852, etf_500):
            d["date"] = pd.to_datetime(d["date"])

        df = (
            idx_300.merge(idx_852, on="date", how="outer")
                   .merge(etf_500, on="date", how="outer")
        )
        df = df.sort_values("date").set_index("date").ffill().dropna()
        return df
    except Exception as e:
        raise RuntimeError(f"akshare 抓取失败：{e}") from e


def trim_last_year(df: pd.DataFrame) -> pd.DataFrame:
    if LAST_YEAR_ONLY:
        end = df.index.max()
        start = end - timedelta(days=365)
    else:
        end = pd.to_datetime(END_DATE)
        start = pd.to_datetime(START_DATE)
    return df.loc[(df.index >= start) & (df.index <= end)].copy()


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    ret = np.log(prices).diff().dropna()
    # winsor
    lo, hi = ret.quantile(WINSOR_Q[0]), ret.quantile(WINSOR_Q[1])
    for c in ret.columns:
        ret[c] = ret[c].clip(lower=lo[c], upper=hi[c])
    return ret


# ---------- 边际转换 ----------
def empirical_pit(x: pd.Series) -> np.ndarray:
    r = x.rank(method="average")
    n = len(x)
    return ((r - 0.5) / n).to_numpy()


def garch_pit(x: pd.Series) -> np.ndarray:
    if not HAS_ARCH:
        raise RuntimeError("请先 pip install arch 才能使用 GARCH PIT")
    am = arch_model(x*100, mean="Constant", vol="GARCH", p=1, o=0, q=1, dist="normal")
    res = am.fit(disp="off")
    sigma = res.conditional_volatility.values / 100.0
    z = (x.values) / (sigma + 1e-12)
    u = norm.cdf(z)
    return u


def to_U(rets: pd.DataFrame, mode: str = "empirical") -> np.ndarray:
    if mode == "garch":
        U = np.column_stack([garch_pit(rets[c]) for c in rets.columns])
    else:
        U = np.column_stack([empirical_pit(rets[c]) for c in rets.columns])
    # 避免精确 0/1
    eps = 1e-6
    U = np.clip(U, eps, 1 - eps)
    return U


# ---------- Gaussian / t Copula：MLE ----------
def _corr_from_cholesky_params(th: np.ndarray) -> np.ndarray:
    """6 个参数生成 3x3 相关矩阵（正定）"""
    l11, l21, l22, l31, l32, l33 = th
    L = np.array([
        [np.exp(l11), 0.0,        0.0       ],
        [l21,         np.exp(l22),0.0       ],
        [l31,         l32,        np.exp(l33)]
    ])
    C = L @ L.T
    D = np.sqrt(np.diag(C))
    R = C / np.outer(D, D)
    return R


def _nll_gaussian_copula(U: np.ndarray, th: np.ndarray) -> float:
    R = _corr_from_cholesky_params(th)
    if not np.all(np.isfinite(R)) or np.any(np.linalg.eigvalsh(R) <= 1e-10):
        return 1e6
    Z = norm.ppf(U)
    invR = np.linalg.inv(R)
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0: return 1e6
    quad = np.einsum('ij,jk,ik->i', Z, invR - np.eye(U.shape[1]), Z)
    ll = -0.5*logdet - 0.5*quad
    return -float(np.sum(ll))


def fit_gaussian(U: np.ndarray) -> Dict:
    # Kendall tau 初值
    R0 = np.eye(3)
    for i in range(3):
        for j in range(i+1,3):
            tau = kendalltau(U[:,i], U[:,j]).correlation
            if tau is None or not np.isfinite(tau): tau = 0.0
            R0[i,j] = R0[j,i] = np.sin(0.5*np.pi * tau)
    # 保证正定
    try:
        L0 = np.linalg.cholesky(R0)
    except np.linalg.LinAlgError:
        R0 = 0.98*R0 + 0.02*np.eye(3)
        L0 = np.linalg.cholesky(R0)
    th0 = np.array([np.log(L0[0,0]), L0[1,0], np.log(L0[1,1]), L0[2,0], L0[2,1], np.log(L0[2,2])])

    res = minimize(lambda th: _nll_gaussian_copula(U, th), th0, method="L-BFGS-B",
                   options={"maxiter":3000,"ftol":1e-10})
    R = _corr_from_cholesky_params(res.x)
    nll = _nll_gaussian_copula(U, res.x)
    n, k = len(U), 3
    AIC = 2*k + 2*nll
    BIC = k*np.log(n) + 2*nll
    return {"model":"Gaussian", "R":R, "ll":-nll, "AIC":AIC, "BIC":BIC, "success":res.success}


def _nll_t_copula(U: np.ndarray, th: np.ndarray) -> float:
    """
    参数：th = [cholesky(6), phi]，nu = 2 + softplus(phi) > 2
    t-copula log密度：多元t密度 / 单变量t密度乘积
    """
    R = _corr_from_cholesky_params(th[:6])
    if not np.all(np.isfinite(R)) or np.any(np.linalg.eigvalsh(R) <= 1e-10):
        return 1e6
    phi = th[6]
    nu = 2.0 + math.log1p(math.exp(phi))  # softplus + 2，避免 nu<=2

    X = student_t.ppf(U, df=nu)    # n x d
    invR = np.linalg.inv(R)
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0: return 1e6
    d = U.shape[1]
    quad = np.einsum('ij,jk,ik->i', X, invR, X)  # x' invR x

    c1 = math.lgamma((nu + d)/2) - math.lgamma(nu/2) - d/2*math.log(nu*math.pi) - 0.5*logdet
    c2 = - np.sum(student_t.logpdf(X, df=nu), axis=1)
    ll = c1 + c2 - (nu + d)/2 * np.log1p(quad/nu)
    nll = -float(np.sum(ll))
    if not np.isfinite(nll): return 1e6
    return nll


def fit_t_copula(U: np.ndarray) -> Dict:
    # Kendall tau 初值
    R0 = np.eye(3)
    for i in range(3):
        for j in range(i+1,3):
            tau = kendalltau(U[:,i], U[:,j]).correlation
            if tau is None or not np.isfinite(tau): tau = 0.0
            R0[i,j] = R0[j,i] = np.sin(0.5*np.pi * tau)
    try:
        L0 = np.linalg.cholesky(R0)
    except np.linalg.LinAlgError:
        R0 = 0.98*R0 + 0.02*np.eye(3)
        L0 = np.linalg.cholesky(R0)
    # 初始 nu=8 -> phi0
    phi0 = math.log(math.exp(8-2)-1)
    th0 = np.array([np.log(L0[0,0]), L0[1,0], np.log(L0[1,1]), L0[2,0], L0[2,1], np.log(L0[2,2]), phi0])

    res = minimize(lambda th: _nll_t_copula(U, th), th0, method="L-BFGS-B",
                   options={"maxiter":4000,"ftol":1e-10})
    nll = _nll_t_copula(U, res.x)
    R = _corr_from_cholesky_params(res.x[:6])
    nu = 2.0 + math.log1p(math.exp(res.x[6]))

    n, k = len(U), 4  # 相关3 + 自由度1（粗略计）
    AIC = 2*k + 2*nll
    BIC = k*np.log(n) + 2*nll
    return {"model":"t", "R":R, "nu":nu, "ll":-nll, "AIC":AIC, "BIC":BIC, "success":res.success}


def tail_dep_t(rho: float, nu: float) -> float:
    """t Copula 上/下尾依赖（对称）：lambda = 2 * t_{nu+1}(- sqrt((nu+1)*(1-rho)/(1+rho)))"""
    arg = - math.sqrt((nu+1) * (1.0-rho) / (1.0+rho + 1e-12))
    return 2.0 * student_t.cdf(arg, df=nu+1)


# ---------- Archimedean：Clayton / Frank / Gumbel 的 MLE ----------
def _fit_archimedean_1param(U: np.ndarray, copula_class, theta_init: float, bounds: tuple) -> Dict:
    cop = copula_class()
    def nll(theta: float) -> float:
        try:
            v = -float(np.sum(cop.logpdf(U, np.array([theta]))))
            if not np.isfinite(v): return 1e6
            return v
        except Exception:
            return 1e6
    res = minimize(lambda x: nll(x[0]), x0=np.array([theta_init]), bounds=[bounds],
                   method="L-BFGS-B", options={"maxiter":3000,"ftol":1e-10})
    theta = float(res.x[0])
    ll = -nll(theta)
    n, k = len(U), 1
    AIC = 2*k - 2*ll
    BIC = k*np.log(n) - 2*ll
    return {"model":copula_class.__name__, "theta":theta, "ll":ll, "AIC":AIC, "BIC":BIC, "success":res.success}


def fit_clayton(U: np.ndarray) -> Dict:
    # tau = theta/(theta+2) -> theta = 2*tau/(1-tau)，Clayton 仅正相关
    taus = []
    for i in range(3):
        for j in range(i+1,3):
            t = kendalltau(U[:,i], U[:,j]).correlation
            if t is not None and np.isfinite(t): taus.append(t)
    tau_bar = np.median(taus) if taus else 0.0
    tau_bar = max(1e-6, min(0.95, tau_bar))
    theta0 = 2*tau_bar/(1.0 - tau_bar)
    return _fit_archimedean_1param(U, ClaytonCopula, theta0, bounds=(1e-6, 100.0))


def fit_gumbel(U: np.ndarray) -> Dict:
    # tau = 1 - 1/theta -> theta = 1/(1-tau)，theta>=1
    taus = []
    for i in range(3):
        for j in range(i+1,3):
            t = kendalltau(U[:,i], U[:,j]).correlation
            if t is not None and np.isfinite(t): taus.append(t)
    tau_bar = np.median(taus) if taus else 0.0
    tau_bar = max(0.0, min(0.95, tau_bar))
    theta0 = 1.0/(1.0 - tau_bar) if tau_bar < 0.999 else 50.0
    theta0 = max(1.0, min(50.0, theta0))
    return _fit_archimedean_1param(U, GumbelCopula, theta0, bounds=(1.0, 100.0))


def fit_frank(U: np.ndarray) -> Dict:
    # Frank 可正可负；用 tau 的符号给方向当初值
    taus = []
    for i in range(3):
        for j in range(i+1,3):
            t = kendalltau(U[:,i], U[:,j]).correlation
            if t is not None and np.isfinite(t): taus.append(t)
    tau_bar = np.median(taus) if taus else 0.0
    theta0 = 4.0 * np.sign(tau_bar) if abs(tau_bar) > 1e-3 else 0.5
    eps = 1e-3
    bounds = (-50.0, -eps) if theta0 < 0 else (eps, 50.0)
    return _fit_archimedean_1param(U, FrankCopula, theta0, bounds=bounds)


def compare_table(results: Dict) -> pd.DataFrame:
    rows=[]
    for name, r in results.items():
        row = {"copula":name, "loglik":r["ll"], "AIC":r["AIC"], "BIC":r["BIC"]}
        if "theta" in r: row["theta"] = r["theta"]
        if "R" in r:
            row["R_12"] = r["R"][0,1]; row["R_13"] = r["R"][0,2]; row["R_23"] = r["R"][1,2]
        if "nu" in r: row["nu"] = r["nu"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AIC")


# ---------- 主流程 ----------
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    prices = try_load_prices()
    prices = trim_last_year(prices)

    need_cols = [COL_510500, COL_000852, COL_000300]
    missing = [c for c in need_cols if c not in prices.columns]
    if missing:
        raise RuntimeError(f"缺少列：{missing}（AkShare 抓取异常）。")

    prices = prices[need_cols].dropna().sort_index()
    rets = to_returns(prices)

    print(f"[info] return sample size: {len(rets)}")
    U = to_U(rets, mode=MARGINAL_MODE)

    # 拟合各 Copula
    res = {}
    res["Gaussian"] = fit_gaussian(U)
    res["t"]        = fit_t_copula(U)
    res["Clayton"]  = fit_clayton(U)
    res["Frank"]    = fit_frank(U)
    res["Gumbel"]   = fit_gumbel(U)

    tab = compare_table(res)
    print("\n=== Model comparison (lower AIC/BIC is better) ===")
    print(tab.to_string(index=False))
    tab.to_csv(Path(OUT_DIR)/"copula_compare_aic.csv", index=False)

    # 若 t Copula 最优，给出尾依赖
    if tab.iloc[0]["copula"] == "t":
        R = res["t"]["R"]; nu = res["t"]["nu"]
        pairs = [("510500","000852", R[0,1]),
                 ("510500","000300", R[0,2]),
                 ("000852","000300", R[1,2])]
        print("\n=== Pairwise tail dependence (t Copula) ===")
        with open(Path(OUT_DIR)/"t_copula_tail_dep.txt","w",encoding="utf-8") as f:
            for a,b,rho in pairs:
                lam = tail_dep_t(rho, nu)
                line = f"{a}-{b}: rho={rho:.3f}, nu={nu:.2f},  lambda_U=lambda_L≈{lam:.4f}"
                print(line); f.write(line+"\n")

    # PIT散点
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    pairs_idx = [(0,1),(0,2),(1,2)]
    cols = [COL_510500, COL_000852, COL_000300]
    for ax, (i,j) in zip(axes, pairs_idx):
        ax.scatter(U[:,i], U[:,j], s=8, alpha=0.5)
        ax.set_xlabel(cols[i]); ax.set_ylabel(cols[j])
        ax.set_title("PIT scatter")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
    plt.tight_layout(); plt.savefig(Path(OUT_DIR)/"pit_scatter.png", dpi=150); plt.close()

    # 简单情景模拟：用最优 Copula + 边际经验逆CDF 回到收益
    best = tab.iloc[0]["copula"]
    np.random.seed(42)
    n_sims = 20000
    if best == "t":
        R = res["t"]["R"]; nu = res["t"]["nu"]
        L = np.linalg.cholesky(R)
        Z = np.random.standard_t(df=nu, size=(n_sims,3))
        X = Z @ L.T
        U_sim = student_t.cdf(X, df=nu)
    elif best == "Gaussian":
        R = res["Gaussian"]["R"]
        L = np.linalg.cholesky(R)
        Z = np.random.randn(n_sims,3) @ L.T
        U_sim = norm.cdf(Z)


    # 经验逆CDF回到收益
    sim_ret = np.zeros_like(U_sim)
    for k, c in enumerate(cols):
        hist = rets[c].dropna().sort_values().values
        ranks = (U_sim[:,k] * (len(hist)-1)).astype(int)
        sim_ret[:,k] = hist[ranks]

    pd.DataFrame(sim_ret, columns=cols).to_csv(Path(OUT_DIR)/"simulated_returns.csv", index=False)
    print(f"\n[done] Results saved in ./{OUT_DIR}/")
    print(" - copula_compare_aic.csv")
    print(" - pit_scatter.png")
    print(" - simulated_returns.csv")
    if tab.iloc[0]["copula"] == "t":
        print(" - t_copula_tail_dep.txt")

if __name__ == "__main__":
    main()