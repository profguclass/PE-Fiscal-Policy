# Fiscal Policy Simulator — a classroom tool

An interactive general-equilibrium model for teaching public economics.
Students choose how a government taxes and what it spends on, then see the
effects on output, hours, capital, tax rates, inequality and welfare, both in
the long run and year by year along the transition.

**Open the app:** <https://profguclass.github.io/PE-Fiscal-Policy/>

## What students can change

| Sidebar item | Setting |
|---|---|
| A | Tax mix: the shares of revenue from the lump-sum, income and consumption taxes |
| B | Public-capital productivity, θ_G |
| C | How much households value in-kind welfare, φ |
| D | Size of government, G/Y (D.1), a further rise in G/Y (D.2) and how long that rise lasts (D.3) |
| E | Composition of spending: public investment, in-kind welfare and cash welfare |
| F | Share of hand-to-mouth households, λ |

The benchmark economy raises revenue 50% lump-sum, 20% income tax and 30%
consumption tax, with G/Y = 20% split 20/50/30 across public investment, in-kind
welfare and cash welfare, θ_G = 0.10, φ = 0.10 and λ = 0.30. Users can replace
this benchmark with their own.

## What the app shows

**Part I. The benchmark vs the sidebar's settings**

- **S1A. Comparison between Old and New Equilibrium:** steady-state levels and
  % changes, benchmark vs the current settings, including inequality between the
  household types (with a two-group Gini) and the long-run consumption-equivalent
  welfare gain.
- **S1B. Transition from Old to New Equilibrium:** the year-by-year path from the
  benchmark to the current settings, in levels, with dashed lines at each series'
  new level, plus inequality and welfare year by year.
- **S2. Growth, incomes, and hours over time:** year-by-year growth, and incomes
  and hours by household type.

**Part II. The Effect of Raising G/Y**

- **S3. Now vs After raising G/Y, in levels and over time:** the effect of more
  spending starting from the current settings, with everything else held fixed.
- **S4. Now vs After raising G/Y, per dollar of new spending:** long-run and
  impact multipliers (ΔX/ΔG) for the same experiment.

An Appendix, opened in a separate window, tells the model agent by agent, with
the equations in boxes.

## The model, briefly

- **Households.** A share λ are hand-to-mouth: they own no capital, do not save,
  pay no income tax and receive all of the cash welfare. The rest are optimizers:
  they own the capital and pay the income tax. Everyone pays the lump-sum and
  consumption taxes. Utility is `ln(C + φ·Gᶜ) + θ_L·ln(1 − N)`.
- **Production.** Cobb-Douglas in private capital, labor and public capital, which
  raises productivity when θ_G > 0.
- **Government.** The three taxes together raise revenue equal to total spending
  every year, each a fixed share of it, so their rates adjust year by year. The
  lump-sum tax distorts nothing. The income tax distorts both work and saving. The
  consumption tax distorts work only, as long as its rate stays constant over time.
- **Spending.** Public investment accumulates into public capital. In-kind welfare
  uses up output and is valued by households through φ. Cash welfare only moves
  income to households.
- **Solution.** Steady states are solved exactly. Transition paths come from
  log-linearizing around the steady state and solving the saddle path by
  eigen-decomposition. With λ > 0, the linearization is numerical.
- **Inequality and welfare.** Each year's levels come from solving the static
  equations exactly along the path. Welfare is discounted with β = 1/(1+r), adds up
  the years shown ("Years to display", 50 by default) and counts the last year for
  every year after that.

The labor share (θ_N = 0.58), depreciation (δ = 10%), real interest rate (r = 6.5%)
and target hours (20% of time) are fixed.

## What's inside

| File | Purpose |
|---|---|
| `fiscal_model.py` | The economics: steady-state and transition-path solvers. No UI code. |
| `app.py` | The Streamlit front end: the sidebar controls and Sections S1–S4. |
| `appendix.py` | The Appendix, shown on its own page via `?view=appendix`. |
| `build_stlite.py` | Packages the app into `docs/` for GitHub Pages. |
| `docs/` | The published site (generated; do not edit by hand). |
| `test_model.py` | Internal consistency checks on the model. |
| `fiscal_model_explainer.tex` / `.pdf` | A written guide to the model and the app. |
| `fiscal_model_code_walkthrough.tex` / `.pdf` | A walkthrough of the code. |

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
