import numpy as np, pandas as pd, json, warnings
from scipy import stats
from scipy.optimize import brentq
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, roc_curve
warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)
sig = lambda x: 1/(1+np.exp(-x))
R = {}

# ======================= 1. SIMULATE =======================
N_PAT = 17500
pat = pd.DataFrame({"patient_id": np.arange(N_PAT)})
pat["age"] = np.clip(rng.normal(60, 17, N_PAT), 18, 96).round()
pat["male"] = rng.binomial(1, 0.48, N_PAT)
pat["rural"] = rng.binomial(1, 0.30, N_PAT)
pr = np.where(pat.rural == 1, [0.38, 0.38, 0.24][0], 0)  # placeholder
inc = np.where(pat.rural == 1, rng.choice(["low","mid","high"], N_PAT, p=[0.46,0.38,0.16]),
                               rng.choice(["low","mid","high"], N_PAT, p=[0.28,0.40,0.32]))
pat["income"] = inc
pat["comorb"] = np.clip(rng.poisson(np.clip(0.9 + 0.03*(pat.age-50), 0.2, None)), 0, 7)
pat["frailty"] = rng.normal(0, 0.4, N_PAT)
n_enc = 1 + rng.poisson(0.46, N_PAT)
enc = pat.loc[pat.index.repeat(n_enc)].reset_index(drop=True)
M = len(enc)
enc["encounter_id"] = np.arange(M)
enc["month"] = np.clip(np.round(rng.uniform(-0.5, 23.5, M) + 0.0), 0, 23).astype(int)
enc["hospital"] = rng.choice(list("ABCD"), M, p=[0.30, 0.28, 0.20, 0.22])
enc["ward"] = rng.choice(["Medical","Surgical","Cardiac","Respiratory","Orthopaedic"], M, p=[0.34,0.22,0.16,0.14,0.14])
enc["prior_adm"] = np.clip(rng.poisson(0.25 + 0.30*enc.comorb), 0, 8)
wl = enc.ward.map({"Medical":0.20,"Respiratory":0.25,"Cardiac":0.05,"Surgical":-0.05,"Orthopaedic":0.0})
enc["los"] = np.clip(np.round(np.exp(1.1 + 0.05*enc.comorb + wl + rng.normal(0, 0.55, M))), 1, 40).astype(int)
pf = sig(-3.3 + 0.03*(enc.age-60) + 0.20*enc.comorb); ph = sig(-1.8 + 0.015*(enc.age-60))
u = rng.uniform(size=M)
enc["dest"] = np.where(u < pf, "facility", np.where(u < pf + (1-pf)*ph, "home_services", "home"))
enc["weekend"] = rng.binomial(1, 0.22, M)
enc["medrec"] = rng.binomial(1, np.where(enc.hospital == "D", 0.68, 0.80), M)
# exclusions
enc["died"] = rng.binomial(1, np.clip(0.014 + 0.003*enc.comorb, 0, 0.05), M)
enc["hospice"] = rng.binomial(1, 0.008, M)
enc["transfer"] = rng.binomial(1, 0.014, M)
# follow-up completed within 7 days
decl = np.where(enc.hospital == "D", -0.14*np.clip(enc.month-13, 0, None), 0.0)
lf = (0.45 - 0.45*enc.rural - 0.30*(enc.income == "low") + 0.15*(enc.income == "high")
      - 0.10*(enc.dest == "facility") + 0.20*(enc.dest == "home_services") + decl)
enc["followup7"] = rng.binomial(1, sig(lf))
# outcome
def lin(b0):
    return (b0 + 0.008*(enc.age-58) + 0.08*enc.male + 0.17*enc.comorb + 0.20*enc.prior_adm + 0.30*(enc.los > 7)
            + 0.10*(enc.dest == "home_services") + 0.18*(enc.dest == "facility")
            + 0.30*enc.rural + 0.15*(enc.income == "low") + 0.05*(enc.income == "mid")
            + 0.22*enc.weekend + enc.ward.map({"Medical":0.25,"Respiratory":0.30,"Cardiac":0.10,"Surgical":0.0,"Orthopaedic":-0.20})
            - 0.75*enc.followup7 - 0.30*enc.medrec
            + enc.frailty)
b0 = brentq(lambda b: sig(lin(b)).mean() - 0.145, -6, 2)
enc["p_true"] = sig(lin(b0))
enc["readmit30"] = rng.binomial(1, enc.p_true)
# missingness (raw extract)
raw = enc.drop(columns=["p_true", "frailty"]).copy()
miss_inc = rng.uniform(size=M) < np.where(raw.rural == 1, 0.11, 0.05)
raw.loc[miss_inc, "income"] = np.nan
raw.loc[rng.uniform(size=M) < 0.025, "comorb"] = np.nan
raw.loc[rng.uniform(size=M) < 0.040, "medrec"] = np.nan
raw.loc[rng.uniform(size=M) < 0.008, "los"] = np.nan
dups = raw.sample(frac=0.006, random_state=1)
raw = pd.concat([raw, dups], ignore_index=True)
raw.to_csv("data/simulated_raw_extract.csv", index=False)

# ======================= 2. CLEAN =======================
flow = {"raw": len(raw)}
df = raw.drop_duplicates("encounter_id").copy(); flow["after_dup"] = len(df); flow["dups"] = flow["raw"] - len(df)
flow["died"] = int((df.died == 1).sum()); df = df[df.died == 0]
flow["hospice"] = int((df.hospice == 1).sum()); df = df[df.hospice == 0]
flow["transfer"] = int((df.transfer == 1).sum()); df = df[df.transfer == 0]
flow["los_missing"] = int(df.los.isna().sum()); df = df[df.los.notna()].copy()
flow["final"] = len(df)
R["flow"] = flow
miss = {"income": int(df.income.isna().sum()), "comorb": int(df.comorb.isna().sum()), "medrec": int(df.medrec.isna().sum())}
R["missing"] = miss; R["missing_pct"] = {k: round(100*v/len(df), 1) for k, v in miss.items()}
df["income"] = df.income.fillna("unknown")
df["comorb_missing"] = df.comorb.isna().astype(int)
med_c = df.comorb.median(); df["comorb"] = df.comorb.fillna(med_c)
df["medrec_cat"] = df.medrec.map({1.0: "yes", 0.0: "no"}).fillna("unknown")
df["los"] = df.los.astype(int); df["los7"] = (df.los > 7).astype(int)
df["month_label"] = pd.Series(pd.period_range("2024-09", periods=24, freq="M").strftime("%b %Y"))[df.month].values
df["pid"] = pd.factorize(df.patient_id)[0]
df.to_csv("data/simulated_clean_cohort.csv", index=False)
n = len(df); R["n"] = n; R["n_patients"] = int(df.patient_id.nunique()); R["events"] = int(df.readmit30.sum())
R["rate"] = round(100*df.readmit30.mean(), 1)
months = pd.period_range("2024-09", periods=24, freq="M").strftime("%b %Y").tolist()
R["months"] = months

# ======================= 3. DESCRIPTIVE (Table 1) =======================
def msd(x): return f"{x.mean():.1f} ({x.std():.1f})"
def med(x): return f"{x.median():.0f} ({x.quantile(.25):.0f}-{x.quantile(.75):.0f})"
def pct(x): return f"{100*x.mean():.1f}%"
a, b = df[df.readmit30 == 1], df[df.readmit30 == 0]
def pcat(col, val):
    t = pd.crosstab(df[col] == val, df.readmit30); return stats.chi2_contingency(t)[1]
def fp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"
rows = []
rows.append(["Age, years: mean (SD)", msd(df.age), msd(a.age), msd(b.age), fp(stats.ttest_ind(a.age, b.age)[1])])
rows.append(["Male", pct(df.male), pct(a.male), pct(b.male), fp(pcat("male", 1))])
rows.append(["Rural residence", pct(df.rural), pct(a.rural), pct(b.rural), fp(pcat("rural", 1))])
rows.append(["Lowest income group", pct(df.income == "low"), pct(a.income == "low"), pct(b.income == "low"), fp(pcat("income", "low"))])
rows.append(["Comorbidities: mean (SD)", msd(df.comorb), msd(a.comorb), msd(b.comorb), fp(stats.ttest_ind(a.comorb, b.comorb)[1])])
rows.append(["Admissions in prior 12 months: mean (SD)", msd(df.prior_adm), msd(a.prior_adm), msd(b.prior_adm), fp(stats.ttest_ind(a.prior_adm, b.prior_adm)[1])])
rows.append(["Length of stay, days: median (IQR)", med(df.los), med(a.los), med(b.los), fp(stats.mannwhitneyu(a.los, b.los)[1])])
rows.append(["Weekend discharge", pct(df.weekend), pct(a.weekend), pct(b.weekend), fp(pcat("weekend", 1))])
rows.append(["Discharged to facility", pct(df.dest == "facility"), pct(a.dest == "facility"), pct(b.dest == "facility"), fp(pcat("dest", "facility"))])
rows.append(["Follow-up completed within 7 days", pct(df.followup7), pct(a.followup7), pct(b.followup7), fp(pcat("followup7", 1))])
rows.append(["Medication reconciliation done", pct(df.medrec_cat == "yes"), pct(a.medrec_cat == "yes"), pct(b.medrec_cat == "yes"), fp(pcat("medrec_cat", "yes"))])
R["table1"] = rows; R["n_a"] = len(a); R["n_b"] = len(b)

# ======================= 4. TREND =======================
def spc(d):
    t = d.groupby("month").readmit30.agg(["sum", "count"]); t["rate"] = t["sum"]/t["count"]
    pb = t.loc[0:11, "sum"].sum()/t.loc[0:11, "count"].sum()
    t["ucl"] = pb + 3*np.sqrt(pb*(1-pb)/t["count"]); t["lcl"] = np.clip(pb - 3*np.sqrt(pb*(1-pb)/t["count"]), 0, None)
    t["signal"] = t.rate > t.ucl
    ab = (t.rate > pb).astype(int).values; rl = 0; st = None
    for i, a_ in enumerate(ab):
        rl = rl+1 if a_ else 0
        if rl >= 8 and st is None: st = i-7
    t["run"] = False
    if st is not None:
        j = st
        while j < len(ab) and ab[j]: t.iloc[j, t.columns.get_loc("run")] = True; j += 1
    return t, pb, st
mt, pbar, rs = spc(df)
R["pbar"] = round(100*pbar, 1); R["first_month_rate"] = round(100*mt.rate.iloc[0], 1)
R["signal_months"] = [months[i] for i in mt.index[mt.signal]]; R["n_signals"] = int(mt.signal.sum())
R["run_start"] = months[rs] if rs is not None else None; R["run_len"] = int(mt.run.sum())
dD = df[df.hospital == "D"]; mtD, pbarD, rsD = spc(dD)
R["pbarD"] = round(100*pbarD, 1); R["signal_months_D"] = [months[i] for i in mtD.index[mtD.signal]]; R["n_signals_D"] = int(mtD.signal.sum())
R["run_start_D"] = months[rsD] if rsD is not None else None; R["run_len_D"] = int(mtD.run.sum())
R["D_above_last6"] = int((mtD.rate.loc[18:23] > pbarD).sum()); R["D_last6_monthly"] = [round(100*v,1) for v in mtD.rate.loc[18:23]]
R["maxD"] = round(100*mtD.rate.max(), 1); R["maxD_month"] = months[int(mtD.rate.idxmax())]
R["rate_first12"] = round(100*pbar, 1)
late = df[df.month >= 18]; early = df[df.month < 18]
R["rate_early18"] = round(100*early.readmit30.mean(), 1); R["rate_last6"] = round(100*late.readmit30.mean(), 1)
ct = pd.crosstab(df.month >= 18, df.readmit30); R["chi_last6_p"] = fp(stats.chi2_contingency(ct)[1])
R["rate_last3"] = round(100*df[df.month >= 21].readmit30.mean(), 1)
tm = smf.glm("readmit30 ~ month", df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": df.pid})
R["trend_or"] = round(float(np.exp(tm.params["month"])), 3); lo, hi = np.exp(tm.conf_int().loc["month"]); R["trend_ci"] = [round(float(lo), 3), round(float(hi), 3)]
R["trend_p"] = fp(tm.pvalues["month"])
# Hospital-specific
hs = df.groupby("hospital").readmit30.agg(["sum", "count"]); hs["rate"] = hs["sum"]/hs["count"]
R["hospital_rates"] = {h: round(100*r, 1) for h, r in hs.rate.items()}
hl = df[df.month >= 18].groupby("hospital").readmit30.mean()
he = df[df.month < 18].groupby("hospital").readmit30.mean()
R["hospital_change"] = {h: [round(100*he[h], 1), round(100*hl[h], 1)] for h in "ABCD"}
fu_hosp = df.assign(grp=np.where(df.hospital == "D", "D", "ABC")).groupby(["grp", "month"]).followup7.mean().unstack(0)
R["fu_ABC_early"] = round(100*df[(df.month < 14) & (df.hospital != "D")].followup7.mean(), 1)
R["fu_D_early"] = round(100*df[(df.month < 14) & (df.hospital == "D")].followup7.mean(), 1)
R["fu_D_last6"] = round(100*df[(df.month >= 18) & (df.hospital == "D")].followup7.mean(), 1)
R["fu_ABC_last6"] = round(100*df[(df.month >= 18) & (df.hospital != "D")].followup7.mean(), 1)
R["fu_overall_first"] = round(100*df[df.month < 14].followup7.mean(), 1); R["fu_overall_last6"] = round(100*late.followup7.mean(), 1)
R["fu_D_m23"] = round(100*df[(df.month >= 21) & (df.hospital == "D")].followup7.mean(), 1)

# ======================= 5. WARD / SUBGROUP RATES =======================
def ci_rate(g):
    r = g.readmit30.agg(["sum", "count"]); lo, hi = proportion_confint(r["sum"], r["count"], method="wilson"); return pd.Series({"rate": r["sum"]/r["count"], "lo": lo, "hi": hi, "n": r["count"]})
wd = df.groupby("ward").apply(ci_rate).sort_values("rate", ascending=False)
R["ward"] = {w: [round(100*r.rate, 1), round(100*r.lo, 1), round(100*r.hi, 1), int(r.n)] for w, r in wd.iterrows()}
R["ward_chi_p"] = fp(stats.chi2_contingency(pd.crosstab(df.ward, df.readmit30))[1])
ward_order = list(wd.index)
R["ward_order"] = ward_order
# ward trend last 6 vs earlier
wl6 = df[df.month >= 18].groupby("ward").readmit30.mean(); we18 = df[df.month < 18].groupby("ward").readmit30.mean()
R["ward_change"] = {w: [round(100*we18[w], 1), round(100*wl6[w], 1)] for w in ward_order}
fu = df.groupby("followup7").apply(ci_rate)
R["fu_no"] = [round(100*fu.loc[0, "rate"], 1), round(100*fu.loc[0, "lo"], 1), round(100*fu.loc[0, "hi"], 1)]
R["fu_yes"] = [round(100*fu.loc[1, "rate"], 1), round(100*fu.loc[1, "lo"], 1), round(100*fu.loc[1, "hi"], 1)]
for col, key in [("rural", "rural"), ("dest", "dest")]:
    pass
rr = df.groupby("rural").readmit30.mean(); R["rural_rate"] = [round(100*rr[0], 1), round(100*rr[1], 1)]
ir = df[df.income != "unknown"].groupby("income").readmit30.mean(); R["inc_rate"] = {k: round(100*v, 1) for k, v in ir.items()}
R["fu_rate_by_rural"] = {int(k): round(100*v, 1) for k, v in df.groupby("rural").followup7.mean().items()}
R["fu_rate_by_inc"] = {k: round(100*v, 1) for k, v in df[df.income != "unknown"].groupby("income").followup7.mean().items()}
ag = pd.cut(df.age, [17, 44, 64, 79, 100], labels=["18-44", "45-64", "65-79", "80+"]); R["age_rate"] = {str(k): round(100*v, 1) for k, v in df.groupby(ag).readmit30.mean().items()}
R["prior_rate"] = {"0": round(100*df[df.prior_adm == 0].readmit30.mean(), 1), "3plus": round(100*df[df.prior_adm >= 3].readmit30.mean(), 1)}
R["prior3_share"] = round(100*(df.prior_adm >= 3).mean(), 1)

# ======================= 6. MULTIVARIABLE LOGISTIC =======================
f = ("readmit30 ~ I((age-58)/10) + male + rural + C(income, Treatment('high')) + comorb + prior_adm + los7 "
     "+ C(dest, Treatment('home')) + weekend + C(ward, Treatment('Surgical')) + followup7 + C(medrec_cat, Treatment('no')) + C(hospital, Treatment('A'))")
mm = smf.glm(f, df, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": df.pid})
ci = np.exp(mm.conf_int()); orr = np.exp(mm.params)
names = {"I((age - 58) / 10)": "Age (per 10 years)", "male": "Male sex", "rural": "Rural residence",
         "C(income, Treatment('high'))[T.low]": "Income: lowest vs highest", "C(income, Treatment('high'))[T.mid]": "Income: middle vs highest",
         "C(income, Treatment('high'))[T.unknown]": "Income: unknown vs highest", "comorb": "Comorbidities (per additional)", "prior_adm": "Prior admissions (per additional)",
         "los7": "Length of stay over 7 days", "C(dest, Treatment('home'))[T.home_services]": "Discharge: home with services vs home",
         "C(dest, Treatment('home'))[T.facility]": "Discharge: facility vs home", "weekend": "Weekend discharge",
         "C(ward, Treatment('Surgical'))[T.Medical]": "Ward: Medical vs Surgical", "C(ward, Treatment('Surgical'))[T.Respiratory]": "Ward: Respiratory vs Surgical",
         "C(ward, Treatment('Surgical'))[T.Cardiac]": "Ward: Cardiac vs Surgical", "C(ward, Treatment('Surgical'))[T.Orthopaedic]": "Ward: Orthopaedic vs Surgical",
         "followup7": "Follow-up within 7 days", "C(medrec_cat, Treatment('no'))[T.yes]": "Medication reconciliation done (vs not)",
         "C(medrec_cat, Treatment('no'))[T.unknown]": "Medication reconciliation unknown (vs not)",
         "C(hospital, Treatment('A'))[T.B]": "Hospital B vs A", "C(hospital, Treatment('A'))[T.C]": "Hospital C vs A", "C(hospital, Treatment('A'))[T.D]": "Hospital D vs A"}
# patsy names
orows = []
for k in mm.params.index:
    if k == "Intercept": continue
    lab = names.get(k, k)
    orows.append([lab, float(orr[k]), float(ci.loc[k, 0]), float(ci.loc[k, 1]), float(mm.pvalues[k])])
R["or_rows"] = [[l, round(o, 2), round(lo, 2), round(hi, 2), fp(p)] for l, o, lo, hi, p in orows]
R["or_unmapped"] = [r[0] for r in orows if r[0] not in names.values()]
OR = {r[0]: r for r in orows}
# trend attenuation
m1 = smf.glm("readmit30 ~ month + C(hospital)", df, family=sm.families.Binomial()).fit()
m2 = smf.glm("readmit30 ~ month + C(hospital) + followup7", df, family=sm.families.Binomial()).fit()
m3 = smf.glm("readmit30 ~ month + C(hospital) + followup7 + comorb + prior_adm + los7 + medrec_cat + weekend", df, family=sm.families.Binomial()).fit()
b1, b2 = m1.params["month"], m2.params["month"]
R["trend_b_unadj"] = round(float(b1), 4); R["trend_b_adjfu"] = round(float(b2), 4); R["trend_attenuation_pct"] = round(float(100*(1-b2/b1)), 0)
dl = dD[dD.month >= 12]
c1 = smf.glm("readmit30 ~ month", dl, family=sm.families.Binomial()).fit().params["month"]
c2 = smf.glm("readmit30 ~ month + followup7", dl, family=sm.families.Binomial()).fit().params["month"]
R["D_attenuation_pct"] = round(float(100*(1-c2/c1)), 0)
ctD = pd.crosstab(dD.month >= 18, dD.readmit30); R["D_chi_p"] = fp(stats.chi2_contingency(ctD)[1])
R["D_rates"] = [round(100*dD[dD.month < 18].readmit30.mean(), 1), round(100*dD[dD.month >= 18].readmit30.mean(), 1)]
R["ABC_rates"] = [round(100*df[(df.hospital != "D") & (df.month < 18)].readmit30.mean(), 1), round(100*df[(df.hospital != "D") & (df.month >= 18)].readmit30.mean(), 1)]
R["D_n_last6"] = int(len(dD[dD.month >= 18]))
# scenario: restore 7-day follow-up in C/D during last 6 months to baseline (hospital C/D early rate)
base_target = R["fu_D_early"]/100
sc = df[(df.month >= 18) & (df.hospital == "D")].copy()
d1 = sc.copy(); d1["followup7"] = 1; d0 = sc.copy(); d0["followup7"] = 0
p1 = mm.predict(d1); p0 = mm.predict(d0)
obs_f = sc.followup7.mean(); q = max((base_target-obs_f)/(1-obs_f), 0)
exp_obs = (sc.followup7*p1 + (1-sc.followup7)*p0).sum()
exp_sc = (sc.followup7*p1 + (1-sc.followup7)*(q*p1 + (1-q)*p0)).sum()
R["scn_n"] = len(sc); R["scn_exp_obs"] = round(float(exp_obs)); R["scn_exp_scn"] = round(float(exp_sc))
R["scn_averted_6m"] = round(float(exp_obs-exp_sc)); R["scn_averted_year"] = round(float(2*(exp_obs-exp_sc)))
R["scn_q"] = round(100*float(q), 0)
R["scn_bed_days"] = round(float(2*(exp_obs-exp_sc))*5.0)


bfu = float(mm.params["followup7"])
lg0 = np.log(p0/(1-p0)); p1h = sig(lg0 + 0.5*bfu); p1f = sig(lg0 + bfu)
exp_obs_h = (sc.followup7*p1h + (1-sc.followup7)*p0).sum(); exp_sc_h = (sc.followup7*p1h + (1-sc.followup7)*(q*p1h + (1-q)*p0)).sum()
R["scn_averted_year_half"] = round(float(2*(exp_obs_h-exp_sc_h)))
R["medrec_yes"] = round(100*(df.medrec_cat == "yes").mean(), 1)
R["medrec_D"] = round(100*(dD.medrec_cat == "yes").mean(), 1); R["medrec_ABC"] = round(100*(df[df.hospital != "D"].medrec_cat == "yes").mean(), 1)
R["weekend_share"] = round(100*df.weekend.mean(), 1); R["los7_share"] = round(100*df.los7.mean(), 1)
R["weekend_rate"] = [round(100*df[df.weekend == 0].readmit30.mean(), 1), round(100*df[df.weekend == 1].readmit30.mean(), 1)]
R["los7_rate"] = [round(100*df[df.los7 == 0].readmit30.mean(), 1), round(100*df[df.los7 == 1].readmit30.mean(), 1)]
R["hosp_share"] = {h: round(100*(df.hospital == h).mean(), 1) for h in "ABCD"}
R["rural_share"] = round(100*df.rural.mean(), 1); R["age65_share"] = round(100*(df.age >= 65).mean(), 1)
R["auc_final_name"] = ""

# ======================= 7. PREDICTIVE MODEL =======================
feat_num = ["age", "comorb", "prior_adm", "los", "male", "weekend", "comorb_missing"]
feat_cat = ["dest", "ward", "hospital", "medrec_cat"]
X = df[feat_num + feat_cat]; y = df.readmit30.values
tr, va, te = (df.month <= 17).values, ((df.month >= 18) & (df.month <= 20)).values, (df.month >= 21).values
R["n_train"] = int(tr.sum()); R["n_val"] = int(va.sum()); R["n_test"] = int(te.sum()); R["ev_test"] = int(y[te].sum()); R["rate_test"] = round(100*y[te].mean(), 1)
R["ev_train"] = int(y[tr].sum())
pre = ColumnTransformer([("num", StandardScaler(), feat_num), ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat)])
cands = {}
best_lr = None
for C in [0.01, 0.1, 1, 10]:
    p = Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=2000))]).fit(X[tr], y[tr])
    cands[f"LR C={C}"] = (log_loss(y[va], p.predict_proba(X[va])[:, 1]), roc_auc_score(y[va], p.predict_proba(X[va])[:, 1]), p)
pg = ColumnTransformer([("num", "passthrough", feat_num), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feat_cat)])
for lr, depth in [(0.05, 3), (0.05, 4)]:
    p = Pipeline([("pre", pg), ("clf", HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth, max_iter=150, l2_regularization=1.0, random_state=0))]).fit(X[tr], y[tr])
    cands[f"GB lr={lr} d={depth}"] = (log_loss(y[va], p.predict_proba(X[va])[:, 1]), roc_auc_score(y[va], p.predict_proba(X[va])[:, 1]), p)
R["val_table"] = [[k, round(v[0], 4), round(v[1], 3)] for k, v in cands.items()]
lr_key = min([k for k in cands if k.startswith("LR")], key=lambda k: cands[k][0]); gb_key = min([k for k in cands if k.startswith("GB")], key=lambda k: cands[k][0])
R["best_lr"] = lr_key; R["best_gb"] = gb_key
final_key = min([lr_key, gb_key], key=lambda k: cands[k][0])
R["final_model"] = final_key
def test_metrics(p):
    pt = p.predict_proba(X[te])[:, 1]
    return pt
pt_lr = cands[lr_key][2].predict_proba(X[te])[:, 1]; pt_gb = cands[gb_key][2].predict_proba(X[te])[:, 1]
pt = cands[final_key][2].predict_proba(X[te])[:, 1]; yt = y[te]
pv = cands[final_key][2].predict_proba(X[va])[:, 1]
def boot_auc(yv, pv_, B=400):
    r = np.random.default_rng(5); out = []
    for _ in range(B):
        i = r.integers(0, len(yv), len(yv)); out.append(roc_auc_score(yv[i], pv_[i]))
    return np.percentile(out, [2.5, 97.5])
for nm, pp in [("lr", pt_lr), ("gb", pt_gb)]:
    lo, hi = boot_auc(yt, pp); R[f"auc_{nm}"] = [round(roc_auc_score(yt, pp), 3), round(float(lo), 3), round(float(hi), 3)]
    R[f"auprc_{nm}"] = round(average_precision_score(yt, pp), 3); R[f"brier_{nm}"] = round(brier_score_loss(yt, pp), 4)
R["brier_null"] = round(brier_score_loss(yt, np.full(len(yt), y[tr].mean())), 4)
# calibration of final
lg = np.log(pt/(1-pt)); cal = sm.GLM(yt, sm.add_constant(lg), family=sm.families.Binomial()).fit()
R["cal_slope"] = [round(float(cal.params[1]), 2), round(float(cal.conf_int()[1][0]), 2), round(float(cal.conf_int()[1][1]), 2)]
cal0 = sm.GLM(yt, np.ones(len(yt)), family=sm.families.Binomial(), offset=lg).fit(); R["cal_intercept"] = round(float(cal0.params[0]), 2)
R["oe_ratio"] = round(float(yt.sum()/pt.sum()), 2)
# threshold: top 20% of validation scores
thr = np.quantile(pv, 0.80); R["thr"] = round(float(thr), 3)
flag = pt >= thr; R["flag_share_test"] = round(100*flag.mean(), 1)
tp = (flag & (yt == 1)).sum(); R["sens20"] = round(100*tp/yt.sum(), 1); R["ppv20"] = round(100*tp/flag.sum(), 1)
R["lift20"] = round(float((tp/flag.sum())/yt.mean()), 2); R["nnc20"] = round(float(flag.sum()/tp), 1)
sp = ((~flag) & (yt == 0)).sum()/(yt == 0).sum(); R["spec20"] = round(100*float(sp), 1)
# gains table
order = np.argsort(-pt); gains = []
for share in [0.05, 0.10, 0.20, 0.30]:
    k = int(share*len(pt)); cap = yt[order][:k].sum()/yt.sum(); ppv = yt[order][:k].mean()
    gains.append([f"{int(share*100)}%", k, f"{100*cap:.1f}%", f"{100*ppv:.1f}%", f"{ppv/yt.mean():.2f}", f"{1/ppv:.1f}"])
R["gains"] = gains
# decision curve
th = np.arange(0.05, 0.41, 0.01); nb_m = []; nb_all = []
for t in th:
    fl = pt >= t; tp_ = (fl & (yt == 1)).sum(); fp_ = (fl & (yt == 0)).sum(); N = len(yt)
    nb_m.append(tp_/N - fp_/N*t/(1-t)); nb_all.append(yt.mean() - (1-yt.mean())*t/(1-t))
R["nb_at_15"] = [round(float(nb_m[10]), 4), round(float(nb_all[10]), 4)]
R["nb_at_20"] = [round(float(nb_m[15]), 4), round(float(nb_all[15]), 4)]
# subgroup audit on test set
dte = df[te].copy(); dte["p"] = pt; dte["flag"] = flag
def grp_row(label, mask):
    g = dte[mask]; e = g.readmit30.sum(); N = len(g)
    if e < 10: return None
    sens = ((g.flag == 1) & (g.readmit30 == 1)).sum()/e; lo, hi = proportion_confint(((g.flag == 1) & (g.readmit30 == 1)).sum(), e, method="wilson")
    try: auc = roc_auc_score(g.readmit30, g.p)
    except: auc = np.nan
    return [label, int(N), int(e), round(100*g.flag.mean(), 1), round(100*sens, 1), round(100*lo, 1), round(100*hi, 1), round(float(auc), 3), round(float(e/g.p.sum()), 2)]
sub = []
for lab, m in [("Urban", dte.rural == 0), ("Rural", dte.rural == 1), ("Income: highest", dte.income == "high"), ("Income: middle", dte.income == "mid"), ("Income: lowest", dte.income == "low"),
               ("Age 18-64", dte.age < 65), ("Age 65+", dte.age >= 65), ("Female", dte.male == 0), ("Male", dte.male == 1)]:
    r = grp_row(lab, m)
    if r: sub.append(r)
for h in "ABCD":
    r = grp_row(f"Hospital {h}", dte.hospital == h)
    if r: sub.append(r)
R["subgroup"] = sub; R["overall_sens"] = R["sens20"]
# permutation-style importance via logistic coefficients not needed; compute simple GB importance by permutation on validation
from sklearn.inspection import permutation_importance
pi = permutation_importance(cands[final_key][2], X[va], y[va], scoring="roc_auc", n_repeats=10, random_state=0)
imp = sorted(zip(feat_num + feat_cat, pi.importances_mean), key=lambda t: -t[1]); R["importance"] = [[k, round(float(v), 4)] for k, v in imp]
R["auc_final"] = R["auc_lr"] if final_key.startswith("LR") else R["auc_gb"]
R["auprc_final"] = R["auprc_lr"] if final_key.startswith("LR") else R["auprc_gb"]
R["brier_final"] = R["brier_lr"] if final_key.startswith("LR") else R["brier_gb"]


# ---- extra numbers for worked examples ----
R["net_min"] = round(100*mt.rate.min(), 1); R["net_max"] = round(100*mt.rate.max(), 1)
R["net_ucl"] = round(100*mt.ucl.mean(), 1); R["net_lcl"] = round(100*mt.lcl.mean(), 1)
last = mtD.iloc[-1]
R["D_aug"] = {"n": int(last["count"]), "events": int(last["sum"]), "rate": round(100*float(last["rate"]), 1), "ucl": round(100*float(last["ucl"]), 1), "lcl": round(100*float(last["lcl"]), 1)}
R["D_min"] = round(100*mtD.rate.min(), 1)
prof = pd.DataFrame([
 dict(age=74, comorb=4, prior_adm=3, los=9, male=1, weekend=1, comorb_missing=0, dest="home_services", ward="Medical", hospital="D", medrec_cat="no"),
 dict(age=58, comorb=2, prior_adm=1, los=5, male=0, weekend=0, comorb_missing=0, dest="home", ward="Surgical", hospital="A", medrec_cat="yes"),
 dict(age=36, comorb=0, prior_adm=0, los=3, male=0, weekend=0, comorb_missing=0, dest="home", ward="Orthopaedic", hospital="B", medrec_cat="yes")])
pp = cands[final_key][2].predict_proba(prof[feat_num+feat_cat])[:, 1]
R["ex_patients"] = [[ "Patient 1: 74-year-old man, Medical ward, Hospital D, 4 comorbidities, 3 prior admissions, 9-day stay, weekend discharge with home services, no medication reconciliation", round(100*float(pp[0]),1), bool(pp[0]>=thr)],
                    [ "Patient 2: 58-year-old woman, Surgical ward, Hospital A, 2 comorbidities, 1 prior admission, 5-day stay, weekday discharge home, medication reconciliation done", round(100*float(pp[1]),1), bool(pp[1]>=thr)],
                    [ "Patient 3: 36-year-old woman, Orthopaedic ward, Hospital B, no comorbidities, no prior admissions, 3-day stay, discharged home, medication reconciliation done", round(100*float(pp[2]),1), bool(pp[2]>=thr)]]
# class imbalance check: class_weight balanced
Cbest = float(lr_key.split("=")[1])
cw = Pipeline([("pre", pre), ("clf", LogisticRegression(C=Cbest, max_iter=2000, class_weight="balanced"))]).fit(X[tr], y[tr])
pcw = cw.predict_proba(X[te])[:, 1]; lgc = np.log(pcw/(1-pcw))
calc = sm.GLM(yt, sm.add_constant(lgc), family=sm.families.Binomial()).fit(); calc0 = sm.GLM(yt, np.ones(len(yt)), family=sm.families.Binomial(), offset=lgc).fit()
R["cw_auc"] = round(roc_auc_score(yt, pcw), 3); R["cw_mean_pred"] = round(100*float(pcw.mean()), 1); R["cw_obs"] = round(100*float(yt.mean()), 1)
R["cw_cal_int"] = round(float(calc0.params[0]), 2); R["cw_cal_slope"] = round(float(calc.params[1]), 2); R["cw_brier"] = round(brier_score_loss(yt, pcw), 4)
R["plain_mean_pred"] = round(100*float(pt_lr.mean()), 1)
R["n_pos_weight"] = round(float((1-y[tr].mean())/y[tr].mean()), 1)

# ======================= 8. FIGURES =======================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
BLUE="#0072B2"; ORANGE="#E69F00"; VERM="#D55E00"; GREEN="#009E73"; SKY="#56B4E9"; INK="#222222"; GREY="#6B6B6B"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
# F1 p-charts
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
for ax, t, pb, ttl in [(axs[0], mt, pbar, "Whole network: no clear change"), (axs[1], mtD, pbarD, "Hospital D: rate climbing above baseline")]:
    x = t.index.values
    ax.axvspan(11.5, 23.5, color="#F3F5F7", zorder=0)
    ax.plot(x, 100*t.rate, color=BLUE, lw=2, marker="o", ms=4, zorder=3)
    ax.step(x, 100*t.ucl, where="mid", color=VERM, ls="--", lw=1); ax.step(x, 100*t.lcl, where="mid", color=VERM, ls="--", lw=1)
    ax.axhline(100*pb, color=GREY, lw=1)
    s_ = t[t.signal | t.run]; ax.scatter(s_.index, 100*s_.rate, color=VERM, marker="s", s=50, zorder=5)
    ax.set_xticks(range(0, 24, 4)); ax.set_xticklabels([months[i] for i in range(0, 24, 4)], fontsize=8); ax.set_xlim(-0.5, 23.5)
    ax.text(0, 100*pb+0.3, f"Baseline {100*pb:.1f}%", fontsize=8, color=GREY); ax.text(12, 100*t.lcl.min()-0.9, "Monitoring period", fontsize=8, color=GREY)
    ax.set_title(ttl, loc="left", fontsize=10, fontweight="bold"); ax.grid(axis="y", color="#E5E8EB", lw=0.8)
axs[0].set_ylabel("30-day readmission rate (%)"); axs[0].set_ylim(5, 28)
fig.savefig("images/f1_trend.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
# F2 ward + hospital change
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axs[0]; w = list(wd.index)[::-1]
for i, wn in enumerate(w):
    r = wd.loc[wn]; col = VERM if r.rate == wd.rate.max() else BLUE
    ax.barh(i, 100*r.rate, color=col, height=0.6); ax.errorbar(100*r.rate, i, xerr=[[100*(r.rate-r.lo)], [100*(r.hi-r.rate)]], color=INK, capsize=3, lw=1)
    ax.text(100*r.hi+0.4, i, f"{100*r.rate:.1f}%", va="center", fontsize=8.5, fontweight="bold")
ax.set_yticks(range(len(w))); ax.set_yticklabels(w); ax.set_xlim(0, 22); ax.set_xlabel("Readmission rate (%) with 95% CI"); ax.axvline(100*df.readmit30.mean(), color=GREY, ls="--", lw=1)
ax.set_title("Medical and Respiratory wards have the highest rates", loc="left", fontsize=9.5, fontweight="bold")
ax = axs[1]; xs = np.arange(4); wd_ = 0.36
e_ = [R["hospital_change"][h][0] for h in "ABCD"]; l_ = [R["hospital_change"][h][1] for h in "ABCD"]
ax.bar(xs-wd_/2, e_, wd_, color=SKY, label="Sep 2024 - Feb 2026"); ax.bar(xs+wd_/2, l_, wd_, color=[BLUE, BLUE, BLUE, VERM], label="Mar - Aug 2026")
for i in range(4): ax.text(xs[i]+wd_/2, l_[i]+0.3, f"{l_[i]}", ha="center", fontsize=8.5, fontweight="bold"); ax.text(xs[i]-wd_/2, e_[i]+0.3, f"{e_[i]}", ha="center", fontsize=8.5)
ax.set_xticks(xs); ax.set_xticklabels([f"Hospital {h}" for h in "ABCD"]); ax.set_ylim(0, max(l_)+4); ax.set_ylabel("Readmission rate (%)")
ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_title("The recent rise is concentrated in Hospital D", loc="left", fontsize=9.5, fontweight="bold")
fig.savefig("images/f2_ward_hospital.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
# F3 follow-up
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
ax = axs[0]
ax.plot(fu_hosp.index, 100*fu_hosp["ABC"], color=BLUE, lw=2, marker="o", ms=3.5, label="Hospitals A, B and C")
ax.plot(fu_hosp.index, 100*fu_hosp["D"], color=VERM, lw=2, marker="s", ms=3.5, ls="--", label="Hospital D")
ax.set_xticks(range(0, 24, 6)); ax.set_xticklabels([months[i] for i in range(0, 24, 6)], fontsize=8); ax.set_ylabel("Follow-up completed within 7 days (%)"); ax.set_ylim(20, 80)
ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.grid(axis="y", color="#E5E8EB", lw=0.8); ax.set_title("7-day follow-up fell in Hospital D", loc="left", fontsize=9.5, fontweight="bold")
ax = axs[1]; vals = [R["fu_yes"], R["fu_no"]]; labs = ["Follow-up within\n7 days", "No follow-up\nwithin 7 days"]
for i, (v, c) in enumerate(zip(vals, [BLUE, VERM])):
    ax.bar(i, v[0], color=c, width=0.55); ax.errorbar(i, v[0], yerr=[[v[0]-v[1]], [v[2]-v[0]]], color=INK, capsize=4); ax.text(i, v[2]+0.4, f"{v[0]}%", ha="center", fontweight="bold", fontsize=10)
ax.set_xticks([0, 1]); ax.set_xticklabels(labs); ax.set_ylabel("30-day readmission rate (%)"); ax.set_ylim(0, 22)
ax.set_title("Readmission is higher without early follow-up", loc="left", fontsize=9.5, fontweight="bold")
fig.savefig("images/f3_followup.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
# F4 forest
fig, ax = plt.subplots(figsize=(8.2, 6.2))
fr = [r for r in orows if not r[0].startswith("Income: unknown") and not r[0].startswith("Medication reconciliation unknown")]
fr = sorted(fr, key=lambda r: r[1])
for i, (lab, o, lo, hi, p) in enumerate(fr):
    col = VERM if o > 1 and lo > 1 else (BLUE if hi < 1 else GREY)
    ax.errorbar(o, i, xerr=[[o-lo], [hi-o]], fmt="o" if o > 1 else "s", color=col, capsize=3, ms=6, lw=1.6)
ax.axvline(1, color=INK, lw=1); ax.set_yticks(range(len(fr))); ax.set_yticklabels([r[0] for r in fr], fontsize=8)
from matplotlib.ticker import NullFormatter
ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3]); ax.set_xticklabels(["0.5", "0.75", "1", "1.5", "2", "3"]); ax.set_xlabel("Adjusted odds ratio (log scale; 95% CI)")
ax.text(1.02, len(fr)-0.3, "higher risk \u2192", fontsize=8, color=VERM, ha="left"); ax.text(0.98, len(fr)-0.3, "\u2190 lower risk", fontsize=8, color=BLUE, ha="right")
ax.grid(axis="x", color="#E5E8EB", lw=0.8); ax.set_title("Factors associated with 30-day readmission", loc="left", fontsize=10, fontweight="bold")
fig.savefig("images/f4_forest.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
# F5 ROC + calibration
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axs[0]
for nm, pp, c in [("Logistic regression", pt_lr, BLUE), ("Gradient boosting", pt_gb, ORANGE)]:
    fpr, tpr, _ = roc_curve(yt, pp); ax.plot(fpr, tpr, color=c, lw=2, label=f"{nm} (AUROC {roc_auc_score(yt, pp):.2f})")
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1); ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity"); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Discrimination on the test period", loc="left", fontsize=9.5, fontweight="bold"); ax.grid(color="#E5E8EB", lw=0.8)
ax = axs[1]
fy, fx = calibration_curve(yt, pt, n_bins=8, strategy="quantile"); ax.plot([0, 0.4], [0, 0.4], color=GREY, ls="--", lw=1, label="Perfect calibration")
ax.plot(fx, fy, color=BLUE, lw=2, marker="o", ms=5, label="Model (test period)")
ax.set_xlabel("Predicted risk"); ax.set_ylabel("Observed proportion readmitted"); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_xlim(0, 0.4); ax.set_ylim(0, 0.4)
ax.set_title("Calibration on the test period", loc="left", fontsize=9.5, fontweight="bold"); ax.grid(color="#E5E8EB", lw=0.8)
fig.savefig("images/f5_roc_cal.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
# F6 gains + decision curve
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axs[0]; ks = np.arange(1, len(pt)+1); cum = np.cumsum(yt[order])/yt.sum()
ax.plot(100*ks/len(pt), 100*cum, color=BLUE, lw=2, label="Model ranking"); ax.plot([0, 100], [0, 100], color=GREY, ls="--", lw=1, label="Random selection")
ax.axvline(20, color=VERM, ls=":", lw=1.2); ax.scatter([20], [R["sens20"]], color=VERM, zorder=5, s=40); ax.text(24, 12, f"Top 20% flagged captures\n{R['sens20']}% of readmissions", fontsize=8.5, color=VERM)
ax.set_xlabel("% of discharges flagged (highest risk first)"); ax.set_ylabel("% of all readmissions captured"); ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.grid(color="#E5E8EB", lw=0.8)
ax.set_title("Share of readmissions reached", loc="left", fontsize=9.5, fontweight="bold")
ax = axs[1]; ax.plot(th, nb_m, color=BLUE, lw=2, label="Model"); ax.plot(th, nb_all, color=ORANGE, lw=2, ls="--", label="Follow up everyone"); ax.axhline(0, color=INK, lw=1, label="Follow up no one")
ax.set_ylim(-0.03, 0.06); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.legend(frameon=False, fontsize=8); ax.grid(color="#E5E8EB", lw=0.8)
ax.set_title("Net benefit of acting on the model", loc="left", fontsize=9.5, fontweight="bold")
fig.savefig("images/f6_gains_dca.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
# F7 subgroup
fig, ax = plt.subplots(figsize=(8.4, 4.4)); sg = sub[::-1]
for i, r in enumerate(sg):
    lo, hi, v = r[5], r[6], r[4]; col = VERM if hi < R["sens20"] else BLUE
    ax.errorbar(v, i, xerr=[[v-lo], [hi-v]], fmt="o" if col == BLUE else "s", color=col, capsize=3, ms=6, lw=1.6)
ax.axvline(R["sens20"], color=GREY, ls="--", lw=1); ax.text(R["sens20"]+0.4, len(sg)-0.4, f"Overall {R['sens20']}%", fontsize=8, color=GREY)
ax.set_yticks(range(len(sg))); ax.set_yticklabels([r[0] for r in sg], fontsize=8.5); ax.set_xlabel("Sensitivity when the top 20% are flagged (%, 95% CI)")
ax.grid(axis="x", color="#E5E8EB", lw=0.8); ax.set_title("Subgroup audit on the test period", loc="left", fontsize=10, fontweight="bold")
fig.savefig("images/f7_subgroup.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

json.dump(R, open("results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("done"); print("saved")
