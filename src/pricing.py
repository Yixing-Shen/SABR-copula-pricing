"""
三资产：SABR→边际(ATM波) + t-Copula → 终值/路径蒙特卡洛
定价：篮子欧式(ATM)、Range Accrual（月度）、雪球（每日KI、月度KO）
Greeks：随到期（Delta/Rho）与随篮子价格（Delta/Gamma/Rho）

Terminal 输入 相关参数 示例：
python pricing_multi_assets.py \
  --sabr_csv ./sabr_final/sabr_params.csv \
  --copula_dir ./copula_out \
  --symbols 510500 000852 000300 \
  --outdir ./pricing_out \
  --allow_ko_after_ki \
  --N_terminal 120000 --N_paths 60000
"""

import os, argparse, json, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
import matplotlib.pyplot as plt

# =============================================================================
# 0) 工具函数与默认参数
# =============================================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def savefig(path, dpi=140, tight=True):
    if tight: plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()

def monthly_indices(n_steps_plus: int, months: int) -> np.ndarray:
    """
    在没有交易日日历的前提下，均匀取"月份观察点"。
    约定：索引0是起点，不计观察；返回的是(1..n_steps)之间的month-ends。
    """
    months = max(1, int(months))
    if n_steps_plus <= 1:
        return np.array([], dtype=int)
    idx = np.linspace(1, n_steps_plus - 1, months, dtype=int)
    return np.unique(idx)

# =============================================================================
# 1) 输入读取：SABR 参数 + Copula 参数
# =============================================================================

def load_sabr_params_from_csv(path: str, symbols: list[str]) -> pd.DataFrame:
    """
    读取并筛选 SABR 拟合结果。
    只保留定价所需的列；symbol 去掉后缀（例如 .SH）。
    """
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].astype(str).str.replace(r"\..*$", "", regex=True)
    if symbols:
        df = df[df["symbol"].isin(symbols)].copy()
    cols = ["symbol", "T", "F", "alpha", "beta", "rho", "nu"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"SABR CSV 缺少列: {missing}")
    return df[cols].sort_values(["symbol", "T"])

def _load_tcopula_from_npz(npz_path: str):
    z = np.load(npz_path)
    R = z["R"]; nu = float(z["nu"])
    return R, nu

def _load_tcopula_from_txt(txt_path: str, d: int):
    """
    兼容一个简单的文本格式：
    第一行形如: nu=4.5
    后续 d 行或 dxd 矩阵给相关阵 R（或 JSON/CSV 也能解析）。
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    nu_line = [ln for ln in lines if ln.lower().startswith("nu")]
    if not nu_line:
        raise ValueError("文本中未找到 nu=")
    nu = float(nu_line[0].split("=")[1])

    # 尝试找 JSON 行或 CSV 方阵
    R = None
    for ln in lines[1:]:
        if ln.startswith("[[") or ln.startswith("["):
            try:
                R = np.array(json.loads(ln), dtype=float)
                break
            except Exception:
                pass
    if R is None:
        # 退而求其次：解析多行以空格/逗号分隔
        rows = []
        for ln in lines[1:1+d]:
            rows.append([float(x) for x in ln.replace(",", " ").split()])
        R = np.array(rows, dtype=float)
    if R.shape[0] != d or R.shape[1] != d:
        raise ValueError(f"R 维度不符: {R.shape}, 期望 {d}x{d}")
    return R, nu

def _estimate_tcopula_from_uniform_csv(csv_path: str, d: int) -> np.ndarray:
    """
    从一个含 U(0,1) 样本（列数=d）的 CSV 粗估相关阵 R（用正态分位变换后的皮尔逊相关）。
    仅作为兜底，不拟合 ν；ν 由外部默认给。
    """
    U = pd.read_csv(csv_path).to_numpy(dtype=float)
    if U.shape[1] != d:
        U = U[:, :d]
    Z = norm.ppf(np.clip(U, 1e-10, 1-1e-10))
    R = np.corrcoef(Z, rowvar=False)
    return R

def load_tcopula_params(copula_dir: str, d: int,
                        R_default: np.ndarray,
                        nu_default: float) -> tuple[np.ndarray, float, str]:
    """
    读取 t-Copula 参数的容错流程（按优先级）：
    1) t_copula_params.npz（包含 R 和 nu）
    2) t_copula_tail_dep.txt（nu=... + R 矩阵）
    3) simulated_returns.csv（或其它 U 样本CSV）估 R，ν 用默认
    4) 否则用默认 R, ν

    返回：(R, nu, source_tag)
    """
    npz_path = os.path.join(copula_dir, "t_copula_params.npz")
    txt_path = os.path.join(copula_dir, "t_copula_tail_dep.txt")
    csv_path = os.path.join(copula_dir, "simulated_returns.csv")

    if os.path.exists(npz_path):
        R, nu = _load_tcopula_from_npz(npz_path)
        return R, nu, "npz"
    if os.path.exists(txt_path):
        try:
            R, nu = _load_tcopula_from_txt(txt_path, d)
            return R, nu, "txt"
        except Exception:
            pass
    if os.path.exists(csv_path):
        try:
            R = _estimate_tcopula_from_uniform_csv(csv_path, d)
            return R, nu_default, "csv+nu_default"
        except Exception:
            pass
    return R_default, nu_default, "default"


# =============================================================================
# 2) 边际：SABR → ATM vol（Hagan 近似）+ 插值器
# =============================================================================

def hagan_atm_vol(F, T, alpha, beta, rho, nu) -> float:
    """
    Hagan(2002) 的 ATM 近似：
    sigma_ATM = α / F^{1-β} * [1 + ((2 - 3ρ²) ν² T)/24]
    """
    base = alpha / (F ** (1.0 - beta))
    corr = 1.0 + ((2.0 - 3.0 * (rho**2.0)) * (nu**2.0) * T) / 24.0
    return float(base * corr)

def build_marginal_atm_vol_funcs(df_sabr: pd.DataFrame, symbols: list[str]):
    """
    返回字典 funcs[symbol](T) -> (F(T), sigma_ATM(T))
    - F 走 log 线性插值，sigma 走线性插值；
    - 端点外推。
    """
    funcs = {}
    for s in symbols:
        g = df_sabr[df_sabr["symbol"] == s].copy()
        if g.empty:
            raise ValueError(f"SABR CSV 中无 {s} 的参数")
        Tg = g["T"].to_numpy()
        Fg = g["F"].to_numpy()
        sig = np.array([hagan_atm_vol(Fg[i], Tg[i], g["alpha"].iat[i],
                                      g["beta"].iat[i], g["rho"].iat[i], g["nu"].iat[i])
                        for i in range(len(g))])

        def _f(T, Tg=Tg, Fg=Fg, sg=sig):
            T = float(T)
            if T <= Tg[0]:  return Fg[0], sg[0]
            if T >= Tg[-1]: return Fg[-1], sg[-1]
            i = np.searchsorted(Tg, T) - 1
            w = (T - Tg[i]) / (Tg[i+1] - Tg[i])
            F_T = math.exp((1-w)*math.log(Fg[i]) + w*math.log(Fg[i+1]))
            s_T = (1-w)*sg[i] + w*sg[i+1]
            return float(F_T), float(s_T)

        funcs[s] = _f
    return funcs


# =============================================================================
# 3) t-Copula 抽样（含反对称）
# =============================================================================

def sample_t_copula_u(n: int, R: np.ndarray, nu: float, rng: np.random.Generator):
    d = R.shape[0]
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal(size=(n, d)) @ L.T
    W = rng.chisquare(df=nu, size=(n, 1)) / nu
    T = Z / np.sqrt(W)
    U = student_t.cdf(T, df=nu)
    return np.clip(U, 1e-10, 1 - 1e-10)

def sample_t_copula_u_antithetic(n: int, R: np.ndarray, nu: float, rng):
    m = (n + 1) // 2
    U = sample_t_copula_u(m, R, nu, rng)
    Ua = 1.0 - U
    return np.vstack([U, Ua])[:n]


# =============================================================================
# 4) 终值分布、篮子欧式价格与 Greeks
# =============================================================================

def ppf_log_normal_from_forward(U, F, vol, T):
    """
    远期测度：ln(S_T/F) ~ N(-0.5σ²T, σ²T)
    """
    z = norm.ppf(U)
    mu = -0.5 * vol * vol * T
    st = vol * np.sqrt(T)
    return F * np.exp(mu + st * z)

def sample_joint_terminal(N: int, T: float, ppf_funcs: dict,
                          symbols: list[str], R: np.ndarray, nu: float,
                          rng, antithetic=True) -> dict[str, np.ndarray]:
    U = sample_t_copula_u_antithetic(N, R, nu, rng) if antithetic else sample_t_copula_u(N, R, nu, rng)
    out = {}
    for j, s in enumerate(symbols):
        F_T, sig_T = ppf_funcs[s](T)
        out[s] = ppf_log_normal_from_forward(U[:, j], F_T, sig_T, T)
    return out

def basket_call_price(samples: dict[str, np.ndarray], r: float, T: float,
                      symbols: list[str], weights=None, strike=None) -> float:
    if weights is None:
        weights = np.ones(len(symbols)) / len(symbols)
    w = np.asarray(weights)
    S_mat = np.column_stack([samples[s] for s in symbols])
    basket = S_mat @ w

    if strike is None:
        strike = float(np.dot(w, [samples[s].mean() for s in symbols]))
    payoff = np.maximum(basket - strike, 0.0)
    return float(np.exp(-r*T) * payoff.mean())

def greeks_delta_rho(N: int, T: float, r: float, ppf_funcs: dict,
                     symbols: list[str], R: np.ndarray, nu: float,
                     w=None, K=None, bump=0.005, seed=1234) -> tuple[float, dict, float]:
    rng = np.random.default_rng(seed)
    U = sample_t_copula_u_antithetic(N, R, nu, rng)
    if w is None:
        w = np.ones(len(symbols)) / len(symbols)

    Fs, sigs = [], []
    base = {}
    for j, s in enumerate(symbols):
        F_T, sig_T = ppf_funcs[s](T)
        Fs.append(F_T); sigs.append(sig_T)
        base[s] = ppf_log_normal_from_forward(U[:, j], F_T, sig_T, T)
    Fs = np.array(Fs); sigs = np.array(sigs)
    strike = float(np.dot(w, Fs)) if K is None else float(K)

    S_mat = np.column_stack([base[s] for s in symbols])
    basket = S_mat @ w
    price0 = float(np.exp(-r*T) * np.maximum(basket - strike, 0.0).mean())

    deltas = {}
    for j, s in enumerate(symbols):
        F_up = Fs.copy(); F_up[j] = Fs[j] * (1 + bump)
        bumped = [ppf_log_normal_from_forward(U[:, k], F_up[k], sigs[k], T) for k in range(len(symbols))]
        basket_b = np.column_stack(bumped) @ w
        price_up = float(np.exp(-r*T) * np.maximum(basket_b - strike, 0.0).mean())
        deltas[s] = (price_up - price0) / (Fs[j] * bump)

    # Rho（仅贴现项；若有曲线，可在外层传入DF）
    dr = 1e-4
    price_r_up = float(np.exp(-(r+dr)*T) * np.maximum(basket - strike, 0.0).mean())
    rho = (price_r_up - price0) / dr

    return price0, deltas, rho

def greeks_vs_basket_level(T: float, ppf_funcs: dict, symbols: list[str],
                           R: np.ndarray, nu: float, r=0.02, N=120_000,
                           seed=202409, scales=None, eps=0.005, weights=None,
                           out_png=None):
    if weights is None:
        weights = np.ones(len(symbols)) / len(symbols)
    w = np.asarray(weights, float)

    Fs_base, sigs = [], []
    for s in symbols:
        F_T, sig_T = ppf_funcs[s](T)
        Fs_base.append(F_T); sigs.append(sig_T)
    Fs_base, sigs = np.array(Fs_base), np.array(sigs)
    B0 = float(np.dot(w, Fs_base))
    K_fixed = B0

    if scales is None:
        scales = np.linspace(0.85, 1.15, 9)
    basket_levels = B0 * scales

    rng = np.random.default_rng(seed)
    U = sample_t_copula_u_antithetic(N, R, nu, rng)

    def _price(scale):
        samples = [ppf_log_normal_from_forward(U[:, j], Fs_base[j]*scale, sigs[j], T)
                   for j in range(len(symbols))]
        basket = np.column_stack(samples) @ w
        return float(np.exp(-r*T) * np.maximum(basket - K_fixed, 0.0).mean())

    rows = []
    for sc, b_lvl in zip(scales, basket_levels):
        P0  = _price(sc)
        Pup = _price(sc*(1+eps))
        Pdn = _price(sc*(1-eps))
        dB = eps * (sc * B0)
        delta = (Pup - Pdn) / (2.0 * dB)
        gamma = (Pup - 2.0*P0 + Pdn) / (dB**2)
        # Rho
        dr = 1e-4
        rho = ((_price(sc) * math.exp(-dr*T)) - P0) / dr  # 近似：仅贴现变化
        rows.append([b_lvl, P0, delta, gamma, rho])

    df = pd.DataFrame(rows, columns=["basket_level", "price", "delta", "gamma", "rho"])

    # 可视化（若指定 out_png 前缀）
    if out_png:
        for col in ["delta", "gamma", "rho"]:
            plt.figure(figsize=(7.2, 4.5))
            plt.plot(df["basket_level"], df[col], marker="o")
            plt.title(f"Basket Call {col.title()} vs Basket (T≈{T:.2f}y)")
            plt.xlabel("Basket Level"); plt.ylabel(col.title()); plt.grid(alpha=0.3)
            savefig(f"{out_png}_{col}.png")

    return df


# =============================================================================
# 5) 路径法：GBM + t-Copula，Range Accrual 与 雪球
# =============================================================================

def simulate_paths_gbm_tcopula(S0: dict, vol: dict, T: float, dt: float,
                               symbols: list[str], R: np.ndarray, nu: float,
                               r=0.02, n_paths=50_000, seed=4321, z_clip=8.0):
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, T + 1e-12, dt)   # 含终点
    n_steps = len(times) - 1
    d = len(symbols)

    out = {s: np.empty((n_paths, n_steps+1), dtype=np.float32) for s in symbols}
    for s in symbols:
        out[s][:, 0] = S0[s]

    L = np.linalg.cholesky(R)
    sig = np.array([vol[s] for s in symbols], dtype=float)

    for k in range(n_steps):
        Z = rng.standard_normal(size=(n_paths, d)) @ L.T
        W = rng.chisquare(df=nu, size=(n_paths, 1)) / nu
        Tz = Z / np.sqrt(W)
        if z_clip:
            Tz = np.clip(Tz, -z_clip, z_clip)

        drift = (r - 0.5 * sig**2) * dt
        diffu = sig * np.sqrt(dt)

        for j, s in enumerate(symbols):
            S_prev = out[s][:, k]
            S_next = S_prev * np.exp(drift[j] + diffu[j] * Tz[:, j])
            out[s][:, k+1] = S_next.astype(np.float32)

    return times, out

def price_range_accrual_monthly(paths: dict, symbols: list[str], r: float, T: float,
                                lower=0.95, upper=1.05, coupon_pa=0.12,
                                months=12, weights=None) -> float:
    if weights is None:
        weights = np.ones(len(symbols)) / len(symbols)
    w = np.asarray(weights)
    any_s = symbols[0]
    n_paths, n_plus = paths[any_s].shape
    idx = monthly_indices(n_plus, months)
    if len(idx) == 0:
        return float(np.exp(-r*T))  # 没有观察，视作0票息

    B0 = sum(w[j]*paths[s][:,0] for j, s in enumerate(symbols))
    B  = sum(w[j]*paths[s]      for j, s in enumerate(symbols))
    in_rng = (B[:, idx] >= lower * B0[:, None]) & (B[:, idx] <= upper * B0[:, None])

    accr = in_rng.sum(axis=1) / float(len(idx))
    coupon = coupon_pa * accr * T
    return float(np.exp(-r*T) * (1.0 + coupon).mean())

def price_snowball_worst_of(paths: dict, symbols: list[str], r: float, T: float,
                            ko_lvl=1.00, ki_lvl=0.80, coupon_pa=0.10,
                            ko_freq="monthly", months=12,
                            allow_ko_after_ki=True) -> float:
    """
    标准雪球（Worst-of）四情景：
      1) 触发 KO（先KI或未KI均可，取决于 allow_ko_after_ki）：立即终止，付 (1 + c * 已持有年化)
      2) 未 KO 未 KI：到期付 (1 + c * T)
      3) 曾 KI 但到期 worst >= 1：0（本金100%）
      4) 曾 KI 且到期 worst < 1：worst（本金按 worst 比例亏损）
    观察：KI = 每日；KO = 月度或每日（按 ko_freq）。
    """
    any_s = symbols[0]
    n_paths, n_plus = paths[any_s].shape
    n_steps = n_plus - 1
    dt = T / n_steps
    disc = math.exp(-r*T)

    S0 = np.array([paths[s][:, 0] for s in symbols])          # (d, n)
    S  = np.stack([paths[s] for s in symbols], axis=0)        # (d, n, n_plus)
    R  = S / S0[:, :, None]                                   # 归一比率

    # KI：逐日（除 t=0）
    ever_ki = (R[:, :, 1:].min(axis=0) <= ki_lvl).any(axis=1)  # (n,)

    # KO：按频率采样
    if ko_freq == "monthly":
        ko_idx = monthly_indices(n_plus, months)
    else:
        ko_idx = np.arange(1, n_plus)

    payoff = np.zeros(n_paths, dtype=np.float64)
    alive  = np.ones(n_paths, dtype=bool)

    for t in ko_idx:
        worst_t = R[:, :, t].min(axis=0)                      # (n,)
        ko_now  = (worst_t >= ko_lvl)
        if not allow_ko_after_ki:
            ko_now &= (~ever_ki)
        hit = ko_now & alive
        if hit.any():
            tau = t * dt
            payoff[hit] = 1.0 + coupon_pa * tau
            alive[hit]  = False
        if not alive.any():
            break

    # 未 KO 到期
    if alive.any():
        idx = np.flatnonzero(alive)
        worst_T = R[:, idx, -1].min(axis=0)
        payoff[idx] = np.where(ever_ki[idx], np.maximum(worst_T, 0.0), 1.0 + coupon_pa * T)

    return float(disc * payoff.mean())


# =============================================================================
# 6打印 & 可视化
# =============================================================================

def _mean_std_str(x, pm=4, ps=3):
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    return f"{m:.{pm}f} ± {s:.{ps}f}"

def pretty_terminal_block(samples: dict, symbols: list[str], r: float, T: float):
    df = pd.DataFrame({s: samples[s] for s in symbols})
    print(f"\n=== T ≈ {T:.3f}y 终值联合模拟 (N={len(df):,}; Copula=t) ===")
    print(df.describe(percentiles=[.25,.5,.75]).T.to_string(float_format=lambda x: f"{x:,.4f}"))

def pretty_summary_table(rows, symbols):
    header = ["到期 (年)"] + [f"{s} (均值±std)" for s in symbols] + ["篮子期权价格 (K≈ATM)"]
    lines = []
    for (T, samples, basket_px, K) in rows:
        row = [f"{T:.3f}"]
        for s in symbols:
            row.append(_mean_std_str(samples[s]))
        row.append(f"≈ {basket_px:.4f}")
        lines.append(row)
    out = pd.DataFrame(lines, columns=header)
    print("\n" + out.to_string(index=False))


# =============================================================================
# 7) 主程序
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sabr_csv", required=True, help="sabr_final/sabr_params.csv")
    ap.add_argument("--copula_dir", required=True, help="包含 t_copula_params.npz / t_copula_tail_dep.txt / simulated_returns.csv 的目录")
    ap.add_argument("--symbols", nargs="+", default=["510500", "000852", "000300"])
    ap.add_argument("--risk_free", type=float, default=0.02)
    ap.add_argument("--trading_days", type=int, default=252)
    ap.add_argument("--seed", type=int, default=202409)
    ap.add_argument("--outdir", default="./pricing_out")

    # Copula 兜底参数（若文件里取不到）
    ap.add_argument("--copula_nu_default", type=float, default=4.5)
    ap.add_argument("--copula_R_default", type=float, nargs="+",
                    default=[1.0, 0.82, 0.88, 0.82, 1.0, 0.96, 0.88, 0.96, 1.0],
                    help="按行展开的相关阵，长度应为 d*d")

    # Monte Carlo 规模
    ap.add_argument("--N_terminal", type=int, default=100_000)
    ap.add_argument("--N_paths", type=int, default=50_000)

    # 雪球条款
    ap.add_argument("--ko_level", type=float, default=1.00)
    ap.add_argument("--ki_level", type=float, default=0.80)
    ap.add_argument("--coupon_pa", type=float, default=0.10)
    ap.add_argument("--ko_freq", choices=["monthly", "daily"], default="monthly")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--allow_ko_after_ki", action="store_true")

    # Range Accrual
    ap.add_argument("--range_lower", type=float, default=0.95)
    ap.add_argument("--range_upper", type=float, default=1.05)
    ap.add_argument("--range_coupon_pa", type=float, default=0.12)

    args = ap.parse_args()
    ensure_dir(args.outdir)

    SYMBOLS = args.symbols
    d = len(SYMBOLS)

    # 读 SABR 并构造边际
    sabr_df = load_sabr_params_from_csv(args.sabr_csv, SYMBOLS)
    ppf_funcs = build_marginal_atm_vol_funcs(sabr_df, SYMBOLS)

    # 读 Copula（或用默认）
    R_def = np.array(args.copula_R_default, dtype=float).reshape(d, d)
    R, nu, tag = load_tcopula_params(args.copula_dir, d, R_def, args.copula_nu_default)
    print(f"[t-Copula] source={tag}, nu={nu:.4f}\nR=\n{R}\n")

    # 2) 终值法：三档到期 + 表格汇总
    maturities = [0.08, 0.18, 0.42]
    rng = np.random.default_rng(args.seed)
    rows = []
    for T in maturities:
        samples = sample_joint_terminal(args.N_terminal, T, ppf_funcs, SYMBOLS, R, nu, rng, antithetic=True)
        pretty_terminal_block(samples, SYMBOLS, args.risk_free, T)
        # 等权 ATM 篮子
        w = np.ones(d) / d
        K_atm = float(np.dot(w, [samples[s].mean() for s in SYMBOLS]))
        basket_px = basket_call_price(samples, args.risk_free, T, SYMBOLS, w, K_atm)
        rows.append((T, samples, basket_px, K_atm))
    pretty_summary_table(rows, SYMBOLS)

    # 3) Greeks：随到期（Delta/Rho）
    print("\n[Greeks: Basket Call（等权、K≈ATM）]")
    Ts_plot = np.array([0.06, 0.12, 0.18, 0.30, 0.42, 0.60])
    delta_curve = {s: [] for s in SYMBOLS}
    rho_curve = []
    for T in Ts_plot:
        P0, deltas, rho = greeks_delta_rho(80_000, T, args.risk_free, ppf_funcs,
                                           SYMBOLS, R, nu, bump=0.005, seed=args.seed+7)
        for s in SYMBOLS:
            delta_curve[s].append(deltas[s])
        rho_curve.append(rho)
        print(f"T={T:.3f} | P≈{P0:,.4f} | " +
              ", ".join([f"Δ_{s}={deltas[s]:.5f}" for s in SYMBOLS]) +
              f" | Rho={rho:,.4f}")

    # 画图
    plt.figure(figsize=(7.6, 4.6))
    for s in SYMBOLS:
        plt.plot(Ts_plot, delta_curve[s], marker="o", label=f"Delta {s}")
    plt.xlabel("Maturity (y)"); plt.ylabel("Delta")
    plt.title("Basket Call Delta (per asset)")
    plt.legend(); plt.grid(alpha=0.3)
    savefig(os.path.join(args.outdir, "greeks_delta_vs_T.png"))

    plt.figure(figsize=(7.6, 4.6))
    plt.plot(Ts_plot, rho_curve, marker="o")
    plt.xlabel("Maturity (y)"); plt.ylabel("Rho")
    plt.title("Basket Call Rho")
    plt.grid(alpha=0.3)
    savefig(os.path.join(args.outdir, "greeks_rho_vs_T.png"))

    # 4) Greeks：随篮子价格
    df_gb = greeks_vs_basket_level(T=0.30, ppf_funcs=ppf_funcs, symbols=SYMBOLS,
                                   R=R, nu=nu, r=args.risk_free, N=120_000,
                                   seed=args.seed, scales=np.linspace(0.85, 1.15, 9),
                                   out_png=os.path.join(args.outdir, "greeks_vs_basket"))
    df_gb.to_csv(os.path.join(args.outdir, "greeks_vs_basket.csv"), index=False)

    # 5) 路径法：Range Accrual（月）、雪球
    T_path = 0.33
    dt = 1.0 / args.trading_days
    S0 = {s: ppf_funcs[s](T_path)[0] for s in SYMBOLS}
    VOL = {s: ppf_funcs[s](T_path)[1] for s in SYMBOLS}

    _, paths = simulate_paths_gbm_tcopula(
        S0, VOL, T=T_path, dt=dt, symbols=SYMBOLS, R=R, nu=nu,
        r=args.risk_free, n_paths=args.N_paths, seed=args.seed, z_clip=8.0
    )

    px_range = price_range_accrual_monthly(
        paths, symbols=SYMBOLS, r=args.risk_free, T=T_path,
        lower=args.range_lower, upper=args.range_upper,
        coupon_pa=args.range_coupon_pa, months=args.months
    )

    px_snow  = price_snowball_worst_of(
        paths, symbols=SYMBOLS, r=args.risk_free, T=T_path,
        ko_lvl=args.ko_level, ki_lvl=args.ki_level, coupon_pa=args.coupon_pa,
        ko_freq=args.ko_freq, months=args.months,
        allow_ko_after_ki=args.allow_ko_after_ki
    )

    # 终值分布快照
    end_vals = {s: paths[s][:, -1] for s in SYMBOLS}
    snap = pd.DataFrame({s: end_vals[s] for s in SYMBOLS})
    snap.describe().to_csv(os.path.join(args.outdir, "path_terminal_snapshot.csv"))

    print("\n[路径法 产品定价]")
    print(f"Range Accrual (±{(args.range_upper-1)*100:.1f}%/{(1-args.range_lower)*100:.1f}%, {args.range_coupon_pa*100:.1f}% pa) ≈ {px_range:,.6f}")
    print(f"Snowball Autocall (KO={args.ko_level:.0%}, KI={args.ki_level:.0%}, {args.coupon_pa*100:.1f}% pa, ko={args.ko_freq}, months={args.months}, allowKOafterKI={args.allow_ko_after_ki}) ≈ {px_snow:,.6f}")

    # 小图存档：终值散点（第一对资产）
    plt.figure(figsize=(5.6, 5.0))
    plt.scatter(snap[SYMBOLS[0]], snap[SYMBOLS[1]], s=6, alpha=0.3)
    plt.xlabel(SYMBOLS[0]); plt.ylabel(SYMBOLS[1])
    plt.title("Terminal Scatter (paths)")
    plt.grid(alpha=0.3)
    savefig(os.path.join(args.outdir, "terminal_scatter.png"))

    print(f"\n=== 输出已保存到: {os.path.abspath(args.outdir)} ===")

if __name__ == "__main__":
    main()
