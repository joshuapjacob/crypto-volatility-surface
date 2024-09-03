"""Streamlit App"""

from datetime import datetime, timezone

import streamlit as st

from lib.data import load_data
from lib.plot import generate_figures
from lib.surface import kernel_smoothing, svi

UNIVERSE = ["BTCUSDT", "ETHUSDT"]

st.set_page_config(
    page_icon=":material/candlestick_chart:",
    page_title="Crypto Volatility Surface",
    layout="wide",
    menu_items={},
)

# ------------------------------------------------------------------------------

st.title("Crypto Volatility Surface")
st.markdown(
    """**A dashboard to visualize cryptocurrency implied volatility surfaces
    constructed with option data from Binance.**"""
)

st.divider()  # ----------------------------------------------------------------

cols = st.columns(2, gap="large")
with cols[0]:
    st.markdown(
        r"""
        Implied volatility (IV) shows the market's expected volatility of an
        option's underlying over the life of the option. Not all options on the
        same underlying have the same IV. High IV results in higher premiums and
        vice versa. 
        
        A [volatility surface](https://www.investopedia.com/articles/stock-analysis/081916/volatility-surface-explained.asp)
        is a three-dimensional plot of IVs of various options listed on the same
        underlying. It can be used to visualize the volatility smile/skew and
        term structure. We use cryptocurrency option data from Binance to
        construct volatility surfaces using two approaches: kernel smoothing and
        the Stochastic Volatility Inspired (SVI) parametrization.
        
        Cryptocurrency options listed on Binance are European-style. For each
        option, Binance uses the [Black-Scholes model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model)
        to provide three different IVs: $\sigma^{\text{bid}}$, 
        $\sigma^{\text{ask}}$, and $\sigma^{\text{mark}}$, calculated from the
        best bid, best ask, and midprice respectively. We use $\tau$ to denote
        the number of days to expiry. For an option with strike price $K$ and
        underlying spot price $S$, we define log moneyness $k$ as
        """
    )
    st.latex(
        r"""
        k = \begin{cases}
            \log(S/K) \quad \text{if call} \\
            \log (K/S) \quad \text{if put}
        \end{cases}
        """
    )
    st.markdown(
        r"""
        In both of our approaches, we also weight observed option data points
        with the following scheme:
        """
    )
    st.latex(
        r"""
        w = \frac{1}{1 + \left(\sigma^\text{ask} - \sigma^\text{bid}\right)}
        """
    )
    st.markdown(
        r"""
        This downweights options with higher IV spreads i.e. options with mark
        IVs that are less "agreed upon".
        """
    )

with cols[1]:
    # Load option data (with caching).
    now_dt = datetime.now(tz=timezone.utc)
    now_hr_ts = (now_dt.timestamp() // 3600) * 3600
    df, update_dt = load_data(now_hr_ts, UNIVERSE)  # Cache hit if same hour.
    min_ago = int(max((now_dt - update_dt).total_seconds() // 60, 0))
    s = "" if min_ago == 1 else "s"
    st.markdown(
        f"""
        Option market data was last updated {min_ago} minute{s} ago at
        {str(update_dt)[:16]} UTC."""
    )
    st.dataframe(df, use_container_width=True)

st.divider()  # ----------------------------------------------------------------

st.header("Kernel Smoothing")
cols = st.columns(2, gap="large")
with cols[0]:
    st.markdown(
        r"""
        [Kernel smoothing](https://en.wikipedia.org/wiki/Kernel_smoother) is a 
        non-parametric approach to volatility surface construction. The kernel
        function determines the weight assigned to each option data point based
        on its distance from the point being estimated. Kernel smoothing does
        not assume a specific functional form for the volatility surface, making
        it adaptable to various shapes and market conditions. This approach
        ensures that the estimate at each point is primarily influenced by
        nearby data, capturing local market behaviors.
        
        At each grid point $j$ on the volatility surface, the smoothed
        volatility $\hat{\sigma}_j$ is calculated as a weighted sum where $i$ is
        the index over all the options on the given underlying.
        """
    )
    st.latex(
        r"""
        \hat{\sigma}_j = \frac{
            \sum_{i=1}^n \sigma^\text{mark}_i  w_i \phi(x_{ij},y_{ij})
        }{
            \sum_{i=1}^n  w_i \phi(x_{ij},y_{ij})
        }
        """
    )
    st.markdown(
        r"""
        Our choice of kernel function $\phi$ is Gaussian-like and inspired by
        the industry standard "TradFi" data vendor
        [OptionMetrics](https://optionmetrics.com/).
        """
    )
    st.latex(
        r"""
        \phi(x,y) = \frac{1}{\sqrt{2 \pi}}
        \exp \left(-\frac{x^2}{2h_x} -\frac{y^2}{2h_y} \right)
        """
    )

with cols[1]:
    st.markdown(
        r"""
        The bandwidth hyperparameters $h_x$ and $h_y$ control the smoothness of
        the surface and the parameters to the kernel function $x_{ij}$ and
        $y_{ij}$ are measures of distance between the option data point and the
        target grid point.
        """
    )
    st.latex(r"x_{ij} = k_i - k_j \quad\quad y_{ij} = \log(\tau_i / \tau_j)")
    h_x = st.slider(
        label="Moneyness Smoothing $h_x$",
        min_value=0.005,
        max_value=0.05,
        value=0.01,
        step=0.005,
        format="%f",
    )
    h_y = st.slider(
        label="Horizon Smoothing $h_y$",
        min_value=0.05,
        max_value=0.2,
        value=0.10,
        step=0.01,
    )

with st.spinner("Constructing kernel smoothed surface..."):
    figs = generate_figures(df, kernel_smoothing, h_x, h_y)
cols = st.columns(len(figs), gap="large")
for col, fig in zip(cols, figs):
    with col:
        st.plotly_chart(fig, use_container_width=True)

st.divider()  # ----------------------------------------------------------------

st.header("Stochastic Volatility Inspired (SVI) Parameterization")
cols = st.columns(2, gap="large")
with cols[0]:
    st.markdown(
        r"""
        The [Stochastic Volatility Inspired (SVI)](https://api.semanticscholar.org/CorpusID:156076635)
        parameterization of the volatility surface was introduced by Merill 
        Lynch in 1999 and popularized by Jim Gatheral. It is designed for
        calibration of a time-specific slice of the implied volatility surface, 
        capturing the dynamics of the skew/smile across strikes, which can be
        challenging to model using other traditional approaches. SVI aligns with
        the behavior of the [Heston model](https://en.wikipedia.org/wiki/Heston_model),
        a widely-used stochastic volatility model, in the long-term maturity
        limit.
        
        For a given parameter set $\chi  = \{a, b, \rho, m, s\}$, the raw SVI
        parameterization of total implied variance is
        """
    )
    st.latex(
        r"""
        f(k; \chi) = a + b \left( \rho (k+m) + \sqrt{(k-m)^2 + s^2}\right)
        """
    )
    st.markdown(r"with implied volatility for a grid point $j$ given by")
    st.latex(r"\hat{\sigma}_j = \sqrt{\frac{f(k_j; \chi_{\tau_j})}{\tau_j}}")


with cols[1]:
    st.markdown(
        r"""
        For each time slice $\tau$, where $i$ is the index over all options on
        the given underlying with $\tau$ days to expiry, we solve the following
        non-linear weighted least squares problem:
        """
    )
    st.latex(
        r"""
        \begin{align*}
        \min_{\chi_{\tau}} \quad& \sum_{i=1}^n
        w_i \left(\hat{\sigma}_i - \sigma^{\text{mark}}_i \right)^2 \\
        \text{s.t.} \quad& b \geq 0 \\
        &|\rho| < 1 \\
        &s > 0
        \end{align*}
        """
    )
    st.markdown(
        r"""
        Note that our calibration of the SVI surface does not guarantee the 
        absence of static (calendar and butterfly) arbitrage. Interpolation over
        various time slices is performed with quadratic bivariate spline
        approximations.
        """
    )

with st.spinner("Constructing SVI surface..."):
    figs = generate_figures(df, svi)
cols = st.columns(len(figs), gap="large")
for col, fig in zip(cols, figs):
    with col:
        st.plotly_chart(fig, use_container_width=True)

st.divider()  # ----------------------------------------------------------------

st.markdown(
    "[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white&color=rgba(0%2C0%2C0%2C0))](https://github.com/joshuapjacob/crypto-volatility-surface)"
)
