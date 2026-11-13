# Fiscal Policy Simulator — a classroom tool

An interactive general-equilibrium model for teaching public economics.
Students choose how a government taxes and what it spends on, then see the
effects on output, hours, capital, tax rates, inequality and welfare, both in
the long run and year by year along the transition.

**Open the app:** <https://profguclass.github.io/PE-Fiscal-Policy/>

## What students can change

| Sidebar item | Setting |
|---|---|
| A | Total government spending, G/Y, before any change |
| B | Composition of spending: public investment, in-kind welfare and cash welfare |
| C | Change in G/Y: by how many percentage points (C1), so the current economy has G/Y = A + C1; whether the change is permanent or temporary, and for how many years (C2); how the change splits across public investment, in-kind welfare and cash welfare, in percentage points (C3) |
| D | Tax mix: the shares of revenue from the capital, labor and consumption taxes |
| E | How much households value in-kind welfare, φ |
| F | Share of hand-to-mouth households, λ |
| G | Public-capital productivity, θ_G |

Items A–D are fiscal policy. Items E–G are exogenous parameters, features of the
economy rather than policy choices, grouped in their own box in the sidebar.

The benchmark economy raises revenue 30% from the capital tax, 40% from the labor
tax and 30% from the consumption tax, with G/Y = 20% split 20/50/30 across public investment, in-kind
welfare and cash welfare, θ_G = 0.10, φ = 0.10 and λ = 0.30. Users can replace
this benchmark with their own (**⚙ Benchmark**).

Sidebar changes take effect when you click **▶ Run**, so several settings can be
changed before the model is solved again. **↺ Reset** returns the sidebar to the
benchmark.

## What the app shows

S1 and S2 compare two economies: **(A) the benchmark** and **(B) the economy
under the sidebar's current settings**, with G/Y = A + C1. S3 changes only government spending, in
both, and compares the multipliers.

- **S1. Comparison between Old and New Equilibrium:** a table of the two
  steady states, (A), (B) and the % change, with every variable numbered and
  grouped by category: commodity market, capital market (including the real
  interest rate and the rental rate of capital), labor market (hours, wage,
  labor productivity), government expenditure, government revenue, and, with
  hand-to-mouth households in both economies, each household type's
  consumption and hours (per person and group total), household income
  (labor, capital, market and disposable income), inequality (Ginis of market
  and disposable income, income and consumption ratios, the hours gap) and
  welfare. Each welfare index (A = 100) has two values: the **new equilibrium**
  and the value **accumulated** over the move from the old to the new
  equilibrium. A bar chart shows the % changes. If the change in C is
  temporary (C2), (B) is the equilibrium the economy ends at, with G/Y back at
  A.
- **S2. Transition from Old to New Equilibrium:** the year-by-year path from
  (A) to (B), with one tab per S1 category and every chart in S1's units. In
  each chart a dotted line marks the benchmark level and a dashed line the new
  equilibrium. The Welfare tab shows each year's welfare and the welfare
  accumulated up to that year.
- **S3. Multiplier Effect of Government Spending:** changes G/Y by item C
  (a rise or a cut, split as in C3, permanent or temporary per C2) in both (A)
  and the current settings at G/Y = A,
  holding everything else fixed. For S1's variables 1–37, it reports the
  multipliers ΔX/ΔG, the change in X per dollar of the change in spending,
  long-run and on impact (year 0): (C) and (E) in the benchmark, (D) and (F)
  in the current settings, and the differences (D-C) and (F-E).
  Bar charts compare them.

An Appendix, opened in a separate window, tells the model agent by agent, with
the equations in boxes.

## The model, briefly

- **Households.** A share λ are hand-to-mouth: they own no capital, do not save
  and receive all of the cash welfare. The rest are optimizers: they own the capital
  and pay the capital tax. Everyone pays the labor and consumption taxes, but
  hand-to-mouth households pay half the optimizers' labor-tax rate (τⁿ/2), a simple
  progressive scheme. Utility is `ln(C + φ·Gᶜ) + θ_L·ln(1 − N)`.
- **Production.** Cobb-Douglas in private capital, labor and public capital, which
  raises productivity when θ_G > 0.
- **Government.** The three taxes together raise revenue equal to total spending
  every year, each a fixed share of it, so their rates adjust year by year. The
  capital tax falls on rental income from capital and distorts saving. The labor tax
  falls on wages and distorts work. The consumption tax distorts work only, as long as
  its rate stays constant over time.
- **Spending.** Public investment accumulates into public capital. In-kind welfare
  uses up output and is valued by households through φ. Cash welfare only moves
  income to households.
- **Solution.** Steady states are solved exactly. Transition paths come from
  log-linearizing around the steady state and solving the saddle path by
  eigen-decomposition. With λ > 0, the linearization is numerical.
- **Incomes, inequality and welfare.** Each year's levels come from solving the
  static equations exactly along the path. Welfare is measured as a
  consumption-equivalent index (benchmark = 100), valued with the current φ.
  Accumulated welfare is discounted with β = 1/(1+r), adds up years 0 to 100 and
  counts year 100 for every year after that.

The labor share (θ_N = 0.58), depreciation (δ = 10%), real interest rate (r = 6.5%)
and target hours (20% of time) are fixed.

## What's inside

| File | Purpose |
|---|---|
| `fiscal_model.py` | The economics: steady-state and transition-path solvers. No UI code. |
| `app.py` | The Streamlit front end: the sidebar controls and Sections S1–S3. |
| `appendix.py` | The Appendix, shown on its own page via `?view=appendix`. |
| `kgu_emblem.png` | The emblem shown on the page. |
| `build_stlite.py` | Packages the app into `docs/` for GitHub Pages. |
| `docs/` | The published site (generated; do not edit by hand). |

## The website

The app is published with GitHub Pages (Settings > Pages > main branch, `/docs`
folder). There is no server: GitHub only serves the files, and each student's
browser downloads a Python runtime
([stlite](https://github.com/whitphx/stlite)) and runs the app itself.

After changing `app.py`, `appendix.py` or `fiscal_model.py`, rebuild and preview:

```bash
python build_stlite.py
python -m http.server 8000 --directory docs
```

Then open `http://localhost:8000`, and commit and push `docs/` along with the
changed source files.
