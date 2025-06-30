import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import numpy_financial as npf
import QuantLib as ql

# --- Title ---
st.title("Hull-White Simulated Mortgage Bond Valuation")

# --- Sidebar Inputs ---
st.sidebar.header("Bond Parameters")
principal = st.sidebar.number_input("Nominal (DKK)", value=1_000_000, step=100_000)
coupon = st.sidebar.number_input("Annual Coupon Rate (%)", value=2.5, step=0.1) / 100
term_years = st.sidebar.slider("Term (Years)", 1, 30, 29)
a = st.sidebar.slider("Hull-White Mean Reversion (a)", 0.0001, 0.05, 0.0009)
sigma = st.sidebar.slider("Hull-White Volatility (sigma)", 0.005, 0.1, 0.022)

st.sidebar.header("Simulation")
num_paths = st.sidebar.number_input("# Simulations", 100, 2000, 500, step=100)

# Swap rates for the zero curve
swap_rates = [(1, 0.0205), (2, 0.020879), (3, 0.022006), (4, 0.023196), (5, 0.024289),
              (6, 0.025137), (7, 0.025853), (8, 0.026505), (9, 0.027091), (10, 0.027606),
              (15, 0.0295), (30, 0.0282)]

# Build zero curve
today = ql.Date.todaysDate()
dates = [today + ql.Period(int(y), ql.Years) for y, _ in swap_rates]
rates = [r for _, r in swap_rates]
zero_curve = ql.ZeroCurve(dates, rates, ql.Actual365Fixed())
curve_handle = ql.YieldTermStructureHandle(zero_curve)

# Hull-White process
hw_process = ql.HullWhiteProcess(curve_handle, a, sigma)
total_months = term_years * 12

rng = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(total_months, ql.UniformRandomGenerator()))
generator = ql.GaussianPathGenerator(hw_process, term_years, total_months, rng, False)

# Prepayment model
def prepayment_rate(r, balance, nper, yd):
    r_m = r / 12
    if nper > 0:
        new_yd = npf.pmt(rate=r_m, nper=nper, pv=-balance)
        gain = yd / new_yd - 1
        if gain < 0:
            return 0
        return ss.norm.cdf(gain, 0.0, 0.05)
    return 0

# Calculate payment
yd_m = npf.pmt(rate=coupon / 12, nper=total_months, pv=-principal)

npvs = []
paths = []

for _ in range(num_paths):
    path = generator.next().value()
    rates = [path[i] for i in range(len(path))]
    paths.append(rates)

    bal = principal
    npv = 0

    for m in range(total_months):
        if bal <= 0:
            break

        r = rates[m]
        nper = total_months - m

        prepay = prepayment_rate(r, bal, nper, yd_m)
        interest = bal * (coupon / 12)
        amort = yd_m - interest
        extra = prepay * bal
        cashflow = yd_m + extra
        bal -= (amort + extra)

        discount = np.exp(-(1/12) * sum(rates[:m+1]))
        npv += cashflow * discount

    npvs.append(npv)

# --- Results ---
st.subheader("Results")
st.write(f"**Average NPV:** {np.mean(npvs):,.2f} DKK")
st.write(f"**As % of Nominal:** {100 * np.mean(npvs) / principal:.2f}%")

# --- Plot ---
st.subheader("Rate Paths")
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(min(50, num_paths)):
    ax.plot(np.linspace(0, term_years, total_months+1), paths[i], lw=0.5, alpha=0.6)
ax.set_xlabel("Years")
ax.set_ylabel("Simulated Short Rate")
st.pyplot(fig)
