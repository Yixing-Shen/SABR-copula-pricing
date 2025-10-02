"""
SABR calibration
Author: 沈依幸
Date: 2025-08-26
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from datetime import datetime


# =============================================================================
# 0) 小工具函数
# =============================================================================
def _to_dt(x):
    if pd.isna(x):
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y", "%Y.%m.%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(x), fmt)
        except ValueError:
            pass
    try:
        return pd.to_datetime(x)
    except Exception:
        return None


def yearfrac(d1, d2, basis=365.0):
    """ACT/basis 年化（basis=365 与常见表格计天保持一致）。"""
    return (d2 - d1).days / basis


def huber(r, delta=0.008):
    """
    Huber 惩罚：|r|<=delta 时二次，尾部转线性，能抑制异常点对目标的影响。
    delta 取值与 IV 量级匹配（这里 0.8 个波动点）。
    """
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def mad_filter(x, k=4.0):
    """MAD（中位数绝对偏差）过滤异常值，k 越大越宽松。"""
    m = np.median(x)
    s = np.median(np.abs(x - m)) + 1e-12
    keep = np.abs(x - m) <= k * 1.4826 * s
    return keep


def normalize_symbol(s):
    """
    将代码规范化为 '000300.SH' 格式。
    - 若无 .SH/.SZ 后缀，尝试按末尾两位补点；纯数字默认 .SH。
    """
    s = str(s).strip()
    if ".SH" not in s and ".SZ" not in s:
        if s.endswith("SH") or s.endswith("SZ"):
            s = s[:-2] + "." + s[-2:]
        elif s.isdigit():
            s = s + ".SH"
    return s.upper()


# =============================================================================
# 1) Black (BSM) 定价 / vega / IV 反推
# =============================================================================
def bsm_price(F, K, T, sig, DF, cp):
    """
    Black 前向定价（折现因子 DF 在外），cp=+1: 看涨；cp=-1: 看跌。
    对于 sig/T/F/K 非正等退化情形，直接回到内在价值。
    """
    if sig <= 0 or T <= 0 or F <= 0 or K <= 0:
        payoff = max(F - K, 0.0) if cp == +1 else max(K - F, 0.0)
        return DF * payoff
    vsT = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vsT * vsT) / vsT
    d2 = d1 - vsT
    Se, Ke = F * DF, K * DF
    if cp == +1:  # Call
        return Se * norm.cdf(d1) - Ke * norm.cdf(d2)
    else:         # Put
        return Ke * norm.cdf(-d2) - Se * norm.cdf(-d1)


def bsm_vega(F, K, T, sig, DF=1.0):
    """Black vega（∂价格/∂σ），用于权重与数值稳定控制。"""
    if sig <= 0 or T <= 0 or F <= 0 or K <= 0:
        return 0.0
    vsT = sig * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * vsT * vsT) / vsT
    return F * DF * norm.pdf(d1) * np.sqrt(T)


def implied_vol(F, K, T, DF, price, cp, lo=1e-6, hi=5.0):
    """
    通过价格反推隐含波动率：
    - 先用无套利上下界过滤（Call: [max(DF*(F-K),0), DF*F]；Put: [max(DF*(K-F),0), DF*K]）
    - 再用 brentq 在 [lo, hi] 上求根。
    """
    Se, Ke = F * DF, K * DF
    lb = max((Se - Ke) if cp == +1 else (Ke - Se), 0.0)
    ub = Se if cp == +1 else Ke
    if not (np.isfinite(price) and lb - 1e-12 <= price <= ub + 1e-12):
        return np.nan
    f = lambda s: bsm_price(F, K, T, s, DF, cp) - price
    try:
        return brentq(f, lo, hi, maxiter=200, xtol=1e-12)
    except Exception:
        return np.nan


# =============================================================================
# 2) Hagan SABR 隐含波动率（分母形式，Eq. 2.17a / 2.18）
# =============================================================================
def sabr_iv_hagan_black(f, K, T, alpha, beta, rho, nu, atm_tol=1e-12):
    """
    Hagan (2002) 近似：σ = α / [(fK)^((1-β)/2) * {1 + ...}] * (z/x(z)) * {1 + [...]T}
    - 当 |ln(f/K)| < atm_tol 用 ATM 公式 (Eq. 2.18)
    - 否则用一般式 (Eq. 2.17a)
    - 对 z/x(z) 在小 z 区间做稳定展开，避免数值奇异
    K 可为标量或 1D 数组。
    """
    K = np.asarray(K, dtype=float)
    f = float(f)
    T = float(T)
    alpha = float(alpha)
    beta = float(beta)
    rho = float(rho)
    nu = float(nu)

    sigma = np.empty_like(K, dtype=float)
    log_fk = np.log(f / K)
    atm_mask = np.abs(log_fk) < atm_tol

    # ATM (Eq. 2.18)
    if np.any(atm_mask):
        f_1mb = f ** (1.0 - beta)
        term_T = ((1 - beta) ** 2 / 24.0) * (alpha ** 2) / (f ** (2 - 2 * beta)) \
                 + 0.25 * (rho * beta * nu * alpha) / (f_1mb) \
                 + ((2 - 3 * rho ** 2) / 24.0) * (nu ** 2)
        sigma[atm_mask] = (alpha / f_1mb) * (1.0 + term_T * T)

    # 非 ATM (Eq. 2.17a)
    if np.any(~atm_mask):
        Kn = K[~atm_mask]
        log_fk_n = log_fk[~atm_mask]

        # (fK)^((1-β)/2)
        fk_pow = (f * Kn) ** ((1.0 - beta) / 2.0)

        # 分母几何修正：1 + ((1-β)^2/24)*ln^2 + ((1-β)^4/1920)*ln^4
        geom_denom = 1.0 \
            + ((1 - beta) ** 2 / 24.0) * (log_fk_n ** 2) \
            + ((1 - beta) ** 4 / 1920.0) * (log_fk_n ** 4)

        # z/x(z) 项
        z = (nu / alpha) * fk_pow * log_fk_n
        sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z * z)
        xz = np.log((sqrt_term + z - rho) / (1.0 - rho))
        # 小 z 稳定展开
        zx = np.where(np.abs(z) < 1e-8,
                      1.0 - 0.5 * rho * z + ((2 - 3 * rho ** 2) / 12.0) * (z * z),
                      z / xz)

        # T 一阶修正
        term_T = ((1 - beta) ** 2 / 24.0) * (alpha ** 2) / ((f * Kn) ** (1.0 - beta)) \
                 + 0.25 * (rho * beta * nu * alpha) / ((f * Kn) ** ((1.0 - beta) / 2.0)) \
                 + ((2 - 3 * rho ** 2) / 24.0) * (nu ** 2)

        sigma[~atm_mask] = (alpha / (fk_pow * geom_denom)) * zx * (1.0 + term_T * T)

    return sigma


# =============================================================================
# 3) 通过 ATMF 回归估计 Forward 与 DF（鲁棒 IRLS）
# =============================================================================
def estimate_F_DF_local(df_term):
    """
    利用恒等式 C - P = DF * (F - K)：
      - 选取 ATMF 邻域（C-P ≈ 0），用 OLS + IRLS 稳健回归解出截距/斜率
      - 由截距/斜率反推 F 与 DF
    """
    d = df_term.copy()
    d["cp"] = d["call_put"].map(lambda x: +1 if x == "C" else -1)
    # 以 K 为索引，分别聚合看涨/看跌的中位数
    piv = d.pivot_table(index="strike_price", columns="cp",
                        values="price_unit", aggfunc="median").dropna()
    # 若数据不足，回退到简单近似（F≈K_atmf，DF≈1）
    if piv.empty or (+1 not in piv.columns) or (-1 not in piv.columns):
        k = float(d["strike_price"].median())
        return k, 1.0

    K = piv.index.values.astype(float)
    y = (piv[+1] - piv[-1]).values.astype(float)  # y = C - P
    # ATMF 近似点：|C-P| 最小处的 K
    kstar = float(K[np.argmin(np.abs(y))])

    # 从紧到松的 ATMF 窗口，逐步 IRLS 以提升稳健性
    for thr in [0.10, 0.12, 0.15, 0.18, 0.22]:
        keep = (np.abs(np.log(K / kstar)) <= thr)
        K2, y2 = K[keep], y[keep]
        if len(K2) >= 4:
            X = np.vstack([np.ones_like(K2), -K2]).T           # y ≈ a - b*K
            w = np.ones_like(y2)
            for _ in range(12):
                W = np.diag(w)
                a, b = np.linalg.lstsq(W @ X, W @ y2, rcond=None)[0]
                r = y2 - (X @ np.array([a, b]))
                s = np.median(np.abs(r - np.median(r))) + 1e-12
                dlt = 1.5 * s
                w = np.where(np.abs(r) <= dlt, 1.0, dlt / np.abs(r))
            DF = max(b, 1e-8)     # 斜率 ~ DF
            F = a / DF            # 截距 ~ DF*F
            return float(F), float(DF)

    # 回退：全局 OLS
    X = np.vstack([np.ones_like(K), -K]).T
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    DF = max(b, 1e-8)
    F = a / DF
    return float(F), float(DF)


# =============================================================================
# 4) 数据加载（宽容 schema 的 Excel 读取）
# =============================================================================
def load_df(path, sheet=None):
    """
    读取并标准化输入数据：
      - 支持 sheet 名/索引
      - 统一小写列名
      - price_unit = (RT_SETTLE 或 CLOSE) / multiplier
      - 提取 call_put（支持 '认购/认沽'）
      - 解析估值日与到期日，计算 T(年)
    """
    if sheet:
        if str(sheet).isdigit():
            df = pd.read_excel(path, sheet_name=int(sheet))
        else:
            df = pd.read_excel(path, sheet_name=sheet)
    else:
        df = pd.read_excel(path)

    # 统一列名为小写、去空格
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    # 代码标准化
    df["us_code"] = df["us_code"].apply(normalize_symbol)

    # 单位价格（按合约乘数折算）
    price_col = "rt_settle" if "rt_settle" in df.columns else ("close" if "close" in df.columns else None)
    if price_col is None:
        raise ValueError("输入文件需包含 RT_SETTLE 或 CLOSE 列以取得价格。")
    mult = pd.to_numeric(df.get("multiplier", 1), errors="coerce").fillna(1.0)
    df["price_unit"] = pd.to_numeric(df[price_col], errors="coerce") / mult
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")

    # 看涨/看跌标志
    cp = df.get("call_put")
    if cp is None:
        cp = df.get("认购认沽", "")
    cp = cp.astype(str)
    df["call_put"] = np.where(cp.str.contains("C", case=False) | cp.str.contains("认购"), "C",
                              np.where(cp.str.contains("P", case=False) | cp.str.contains("认沽"), "P", None))

    # 估值日与到期时间 T（年）
    df["valuation_dt"] = df.get("valuation_date", pd.NaT).apply(_to_dt)
    Ts = []
    for _, r in df.iterrows():
        T = np.nan
        e = r.get("expiredate", np.nan)
        vd = r.get("valuation_dt", None)
        if pd.isna(e):
            T = np.nan
        else:
            try:
                # 若为数值天数，直接年化
                T = max(float(e) / 365.0, 0.0)
            except Exception:
                # 若为日期，做年化天数差
                ed = _to_dt(e)
                T = max(yearfrac(vd, ed, 365.0), 0.0) if (ed is not None and vd is not None) else np.nan
        Ts.append(T)
    df["T"] = Ts
    return df


# =============================================================================
# 5) 校准辅助函数（曲率正则 / 交叉验证切分）
# =============================================================================
def _curvature_penalty(K, sig):
    """二阶差分的均方（按 dK^2 归一），惩罚过度弯折，提高外推稳定性。"""
    if len(sig) < 5:
        return 0.0
    K = np.asarray(K, float)
    sig = np.asarray(sig, float)
    dK = np.diff(K).mean()
    curv = sig[2:] - 2 * sig[1:-1] + sig[:-2]
    return float(np.mean((curv / (dK ** 2)) ** 2))


def _split_folds(n, k=3):
    """交错式 k 折划分，保持每折分布均衡且可复现。"""
    idx = np.arange(n)
    folds = [idx[i::k] for i in range(k)]
    return folds


# =============================================================================
# 6) SABR 单到期校准（混合损失 + 3 折交叉验证 + 自适应 β）
# =============================================================================
def calibrate_term(F, T, K, iv):
    """
    在单个到期 T 上拟合 SABR 参数：
      - 目标：0.75 * IV Huber 残差（vega 权重、两翼指数衰减）
             + 0.25 * 价格相对误差（以 OTM 对偶合成的近似）
             + 曲率正则 + rho/nu 的软先验
      - 3 折交叉验证（折内 0.7 训练 + 0.3 验证再聚合）
      - β 网格由微笑曲率粗估后自适应（平、适中、陡）选择
    返回: dict(alpha, beta, rho, nu, rmse) 或 None
    """
    short_t = (T < 0.12)
    logM = np.log(K / F)

    # ---- 用二次项系数近似“微笑曲率”，决定 β 网格范围 ----
    x = np.abs(logM)
    try:
        curvature = np.polyfit(x, iv, 2)[0]
    except Exception:
        curvature = 0.0
    if curvature < 5e-4:
        beta_grid = np.arange(0.75, 1.00 + 1e-9, 0.05)
    elif curvature < 1.5e-3:
        beta_grid = np.arange(0.65, 0.95 + 1e-9, 0.05)
    else:
        beta_grid = np.arange(0.55, 0.85 + 1e-9, 0.05)
    if short_t:
        beta_grid = np.clip(beta_grid, 0.70, 1.00)

    # ν 的硬边界（短期限更宽）
    nu_floor = 0.04 if short_t else 0.02
    nu_max = 12.0 if short_t else 6.0

    folds = _split_folds(len(K), k=3)
    best = None

    for beta in beta_grid:
        # ---- 初值：α 用 ATM 缩放，ρ 用斜率符号，ν 与 ATM IV/T 相关 ----
        atm_idx = int(np.argmin(np.abs(logM)))
        atm_iv = max(iv[atm_idx], 1e-4)
        alpha0 = max(atm_iv * (max(F, K[atm_idx]) ** (1 - beta)), 1e-4)
        slope = np.polyfit(logM, iv, 1)[0] if len(K) >= 3 else 0.0
        rho0 = float(np.clip(-np.sign(slope) * 0.2, -0.5, 0.5))
        nu0 = max(0.6 * atm_iv / max(np.sqrt(T), 1e-4), nu_floor)
        if short_t:
            nu0 *= 0.7

        def obj(theta):
            # 变量限制：α>0, |ρ|<=0.95, ν 在 [nu_floor, nu_max]
            alpha, rho, nu = theta
            rho = float(np.clip(rho, -0.95, 0.95))
            alpha = float(max(alpha, 1e-6))
            nu = float(np.clip(nu, nu_floor, nu_max))

            fold_losses = []
            for ki in range(3):
                val_idx = folds[ki]
                trn_idx = np.setdiff1d(np.arange(len(K)), val_idx)

                K_tr, iv_tr = K[trn_idx], iv[trn_idx]
                K_va, iv_va = K[val_idx], iv[val_idx]

                sig_tr = sabr_iv_hagan_black(F, K_tr, T, alpha, beta, rho, nu)
                sig_va = sabr_iv_hagan_black(F, K_va, T, alpha, beta, rho, nu)

                # vega 权重 + 两翼指数衰减（远离 F 权重降）
                vega_tr = np.maximum(np.array([bsm_vega(F, k, T, max(s, 1e-6))
                                               for k, s in zip(K_tr, sig_tr)]), 1e-12)
                vega_va = np.maximum(np.array([bsm_vega(F, k, T, max(s, 1e-6))
                                               for k, s in zip(K_va, sig_va)]), 1e-12)
                wing_scale = (0.18 if short_t else 0.22)
                w_tr = (vega_tr / vega_tr.max()) ** 1.0 * np.exp(-np.abs(np.log(K_tr / F)) / wing_scale)
                w_va = (vega_va / vega_va.max()) ** 1.0 * np.exp(-np.abs(np.log(K_va / F)) / wing_scale)

                # (A) IV 空间的 Huber 残差
                loss_iv_tr = float(np.nanmean(huber(iv_tr - sig_tr) * w_tr))
                loss_iv_va = float(np.nanmean(huber(iv_va - sig_va) * w_va))

                # (B) 价格空间的相对误差（用 OTM 对偶合成近似价格）
                def _price_rel_loss(Kx, ivx, sigx, w):
                    px_m = np.array([bsm_price(F, k, T, max(s, 1e-6), 1.0,
                                               +1 if k >= F else -1) for k, s in zip(Kx, sigx)])
                    px_iv = np.array([bsm_price(F, k, T, max(sv, 1e-6), 1.0,
                                                +1 if k >= F else -1) for k, sv in zip(Kx, ivx)])
                    rel = (px_m - px_iv) / np.maximum(px_iv, 1e-6)
                    return float(np.nanmean(huber(rel) * w))

                loss_px_tr = _price_rel_loss(K_tr, iv_tr, sig_tr, w_tr)
                loss_px_va = _price_rel_loss(K_va, iv_va, sig_va, w_va)

                # (C) 曲率正则（只在训练集上计算，避免泄露）
                pen_curv = _curvature_penalty(K_tr, sig_tr)

                # (D) 软先验：|ρ|<=0.8，ν√T ∈ [0.05, 2.0]
                rho_pen = max(0.0, abs(rho) - 0.80) ** 2
                nu_scaled = nu * np.sqrt(T)
                nu_pen = max(0.0, nu_scaled - 2.0) ** 2 + max(0.0, 0.05 - nu_scaled) ** 2

                lam_curv = 2e-3 if short_t else 1e-3
                lam_rho = 3e-4
                lam_nu = 3e-4

                train_loss = 0.75 * loss_iv_tr + 0.25 * loss_px_tr \
                             + lam_curv * pen_curv + lam_rho * rho_pen + lam_nu * nu_pen
                valid_loss = 0.75 * loss_iv_va + 0.25 * loss_px_va

                fold_losses.append(0.7 * train_loss + 0.3 * valid_loss)

            return float(np.mean(fold_losses))

        bnds = [(1e-6, 10.0), (-0.95, 0.95), (nu_floor, nu_max)]
        res = minimize(obj, np.array([alpha0, rho0, nu0]),
                       method="L-BFGS-B", bounds=bnds, options={"maxiter": 500})
        if not res.success:
            continue

        a, r, n = res.x
        sig = sabr_iv_hagan_black(F, K, T, a, beta, r, n)
        rmse = float(np.sqrt(np.nanmean((iv - sig) ** 2)))
        cand = {"alpha": float(a), "beta": float(beta), "rho": float(r),
                "nu": float(n), "rmse": rmse}
        if (best is None) or rmse < best["rmse"]:
            best = cand

    return best


# =============================================================================
# 7) 主流程：逐标的/到期校准，画图与汇总
# =============================================================================
def run(df, symbols, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []

    for sym in symbols:
        sub = df[df["us_code"] == sym].copy()
        if sub.empty:
            print(f"[WARN] 无数据: {sym}")
            continue

        # 以 T（年）聚合，每个到期做一次校准
        sub["Tbin"] = sub["T"].round(6)
        for T, g0 in sub.groupby("Tbin"):
            # 跳过不合理/过短到期
            if not np.isfinite(T) or T <= 0 or T < 0.03:
                continue

            # 合并重复 (K, CP) 到中位价
            g = g0.groupby(["strike_price", "call_put"], as_index=False) \
                  .agg(price_unit=("price_unit", "median"))

            # 基于 ATMF 回归估计 F 与 DF
            F, DF = estimate_F_DF_local(g)

            # 仅用 OTM 点反推 IV（看涨取 K>=F，看跌取 K<=F）
            K_list, IV_list = [], []
            for _, r in g.iterrows():
                K = float(r["strike_price"])
                cp = +1 if r["call_put"] == "C" else -1
                if cp == +1 and K < 0.99 * F:  # Call 仅用 OTM（略加 1% 缓冲）
                    continue
                if cp == -1 and K > 1.01 * F:  # Put 仅用 OTM
                    continue
                px = float(r["price_unit"])
                iv = implied_vol(F, K, T, DF, px, cp)
                if np.isfinite(iv) and 0.01 <= iv <= 3.0:
                    K_list.append(K)
                    IV_list.append(iv)
            if len(K_list) < 8:
                continue

            K = np.array(K_list)
            iv = np.array(IV_list)

            # 先按 IV 做 MAD 去异常，再保证对称 moneyness 窗口
            keep = mad_filter(iv, k=4.0)
            K, iv = K[keep], iv[keep]
            logm = np.abs(np.log(K / F))
            thr = 0.22 if T < 0.12 else 0.30
            while (logm <= thr).sum() < 12 and thr < 0.90:
                thr += 0.04
            sel = (logm <= thr)
            K, iv = K[sel], iv[sel]
            ord_ = np.argsort(K)
            K, iv = K[ord_], iv[ord_]
            if len(K) < 10:
                continue

            # 硬门限：两翼至少 3 点，覆盖至少（短期 0.20，长一点 0.25）
            left = int((K < F).sum())
            right = int((K > F).sum())
            cover = float(np.max(np.abs(np.log(K / F))))
            min_cover = 0.20 if T < 0.12 else 0.25
            if left < 3 or right < 3 or cover < min_cover:
                continue

            # 进行 SABR 校准
            best = calibrate_term(F, T, K, iv)
            if best is None:
                print(f"[WARN] 校准失败: {sym} T={T:.3f}")
                continue

            # 写入一行结果
            rows.append({
                "symbol": sym,
                "valuation_date": pd.to_datetime(sub["valuation_dt"].iloc[0]) if "valuation_dt" in sub else pd.NaT,
                "T": T, "F": F, "DF": DF,
                **best,
                "n": len(K),
                "Kmin": float(np.min(K)),
                "Kmax": float(np.max(K)),
                "thr_logM": float(thr),
                "left_pts": left, "right_pts": right
            })

            # 绘制微笑拟合图
            sig = sabr_iv_hagan_black(F, K, T, best["alpha"], best["beta"], best["rho"], best["nu"])
            plt.figure(figsize=(8.6, 5.2))
            plt.scatter(K, iv, s=28, alpha=0.95, label="市场 IV (OTM)")
            plt.plot(K, sig, lw=2.4, label=f"SABR 拟合 β={best['beta']:.2f}")
            plt.axvline(F, ls="--", lw=1.2, label="Forward ~ F")  # ← 修正这里
            plt.title(f"{sym} | T≈{T:.3f}y | RMSE={best['rmse']:.4f} | n={len(K)}")
            plt.xlabel("执行价 K")
            plt.ylabel("隐含波动率 (BSM)")
            plt.legend()
            p = os.path.join(outdir, f"{sym.replace('.', '_')}_T{T:.3f}.png")
            plt.tight_layout()
            plt.savefig(p, dpi=170)
            plt.close()

            print(f"[OK] {sym} | T≈{T:.3f}y | RMSE={best['rmse']:.4f} | n={len(K)}")

    # 保存汇总 CSV
    if rows:
        res = pd.DataFrame(rows).sort_values(["symbol", "T"])
        csv_path = os.path.join(outdir, "sabr_params.csv")
        res.to_csv(csv_path, index=False)
        print("\n=== 参数汇总已保存 ===")
        print(csv_path)
        try:
            print(res.to_string(index=False))
        except Exception:
            pass
    else:
        print("[WARN] 无成功的到期校准结果")


# =============================================================================
# 8) 命令行入口
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="SABR (Hagan denominator) calibration with anti-overfitting.")
    ap.add_argument("--input", default="option_data.xlsx", help="输入 Excel 路径")
    ap.add_argument("--sheet", default=None, help="表名或索引")
    ap.add_argument("--symbols", nargs="+", default=["510500.SH", "000852.SH", "000300.SH"],
                    help="需要校准的标的代码列表")
    ap.add_argument("--outdir", default="./sabr_final", help="输出目录（图与CSV）")
    args = ap.parse_args()

    df = load_df(args.input, args.sheet)
    run(df, args.symbols, args.outdir)


if __name__ == "__main__":
    main()
