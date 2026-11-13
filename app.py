import difflib
import io
import json
import time
import math
import re
import unicodedata
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import binom, nbinom, poisson

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Combo Master 4.0", layout="wide", page_icon="⚽")

# Campionati (fonte: football-data.co.uk)
LEAGUES = {
    "Serie A (Italia)": "I1", "Premier League (Inghilterra)": "E0",
    "La Liga (Spagna)": "SP1", "Bundesliga (Germania)": "D1",
    "Ligue 1 (Francia)": "F1", "Eredivisie (Olanda)": "N1",
}
API_CODICI = {"I1": "SA", "E0": "PL", "SP1": "PD", "D1": "BL1", "F1": "FL1", "N1": "DED"}

# Competizioni Internazionali (fonte: football-data.org + The Odds API)
INTL_COMPS = {
    "Champions League": {"fd_id": "CL", "odds_key": "soccer_uefa_champs_league"},
    "Europa League": {"fd_id": "EL", "odds_key": "soccer_uefa_europa_league"},
    "Conference League": {"fd_id": "UECL", "odds_key": "soccer_uefa_europa_conference_league"},
    "Nations League": {"fd_id": "UNL", "odds_key": "soccer_uefa_nations_league"},
    "Europei (Qualificazioni)": {"fd_id": "EC", "odds_key": "soccer_uefa_euro_qualification"},
}

BASE_URL = "https://football-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0 (ComboMaster4)"}
TIMEOUT = 20

N_STAGIONI = 3
N_STAGIONI_TEST = 8
FINESTRA_GIORNI = 1000
PASSO_TEST = 14

RIDGE_GRID = (0.5, 2.0, 6.0)
HALFLIFE_GRID = (180, 365, 730, 0)
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

COL_PRESA = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
COL_CHIUSURA = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
COL_OU_PRESA = [("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5"), ("Max>2.5", "Max<2.5")]
COL_OU_CHIUSURA = [("AvgC>2.5", "AvgC<2.5"), ("PC>2.5", "PC<2.5"), ("B365C>2.5", "B365C<2.5")]

STAT_COLONNE = {"Tiri": ("HS", "AS"), "Tiri in Porta": ("HST", "AST"), "Corner": ("HC", "AC"), "Falli": ("HF", "AF"), "Ammonizioni": ("HY", "AY")}

SOGLIA_FIDUCIA_ALTA = 0.70
SOGLIA_FIDUCIA_MEDIA = 0.55
SOGLIA_FAVORITA_NETTA = 0.55
SOGLIA_EQUILIBRIO_GAP = 0.08
SOGLIA_EQUILIBRIO_X = 0.25
SOGLIA_X_FORZATA = 0.30
PROB_MIN_DEFAULT = 30
EDGE_MIN_DEFAULT = 3.0

COL_PRIMARIO = "#38bdf8"
COL_ACCENTO = "#a855f7"
COL_CASA = "#38bdf8"
COL_TRASF = "#f43f5e"

# ============================================================
# CSS & HEADER
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stApp {{ background-color: #0b0f19; color: #f3f4f6; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px; }}
[data-testid="stSidebar"] {{ background-color: #111827; border-right: 1px solid #1f2937; }}
.stButton>button {{
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: white; border: none; border-radius: 10px; font-weight: 600;
    padding: 0.5rem 1rem; transition: all 0.2s;
}}
.stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,.4); }}
[data-testid="stMetric"] {{
    background-color: #111827; border: 1px solid #1f2937;
    padding: 15px; border-radius: 12px;
}}
[data-testid="stMetricValue"] {{ color: {COL_PRIMARIO} !important; font-weight: 700 !important; }}
div[data-testid="stExpander"] {{ border: 1px solid #1f2937; border-radius: 12px; }}

/* TOOLTIP CSS */
.tooltip {{ position: relative; display: inline-block; cursor: help; color: #64748b; font-size: 0.8rem; margin-left: 4px; }}
.tooltip .tooltiptext {{ visibility: hidden; width: 220px; background-color: #1e293b; color: #f8fafc; text-align: center; border-radius: 6px; padding: 8px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -110px; opacity: 0; transition: opacity 0.3s; font-size: 0.75rem; font-weight: normal; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #334155; }}
.tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}

/* OTTIMIZZAZIONE MOBILE / iOS */
@media (max-width: 768px) {{
    .block-container {{ padding-top: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }}
    h1 {{ font-size: 1.5rem !important; }}
    h3 {{ font-size: 1.1rem !important; }}
    [data-testid="stMetricValue"] {{ font-size: 1.2rem !important; }}
    .stButton>button {{ width: 100%; }}
    .stDataFrame {{ font-size: 0.8rem; }}
    .tooltip .tooltiptext {{ width: 160px; margin-left: -80px; }}
}}
</style>
""", unsafe_allow_html=True)

def tooltip_testo(testo):
    return f'<span class="tooltip">❓<span class="tooltiptext">{testo}</span></span>'

st.markdown(f"""
<div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 25px; border-radius: 20px; border: 1px solid #312e81; text-align: center; margin-bottom: 20px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.4);">
    <h1 style="color: #f8fafc; font-weight: 800; font-size: 2rem; margin: 0;">⚽ COMBO MASTER 4.0</h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 8px;">Dixon-Coles onesto · la X emerge quando deve · test separati dal training</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# UTILITY
# ============================================================
_STOP_NOMI = {"fc", "ac", "as", "cf", "afc", "acf", "ssc", "us", "ud", "rc", "rcd", "sv", "vfl", "vfb", "fsv", "tsg", "fk", "cd", "ca", "sc", "bv", "de", "calcio", "ogc", "sco"}
_ALIAS_NOMI = {"inter": "internazionale milano", "verona": "hellas verona", "man united": "manchester united", "man city": "manchester city", "nott m forest": "nottingham forest", "wolves": "wolverhampton wanderers", "ath madrid": "club atletico madrid", "ath bilbao": "athletic club", "espanol": "espanyol barcelona", "bayern munich": "bayern munchen", "m gladbach": "borussia monchengladbach", "ein frankfurt": "eintracht frankfurt", "paris sg": "paris saint germain", "lyon": "olympique lyonnais", "rennes": "stade rennais", "brest": "stade brestois", "for sittard": "fortuna sittard", "manchester city fc": "manchester city", "manchester united fc": "manchester united", "arsenal fc": "arsenal", "chelsea fc": "chelsea", "liverpool fc": "liverpool", "tottenham hotspur fc": "tottenham"}

def _norm_sq(nome):
    n = unicodedata.normalize("NFKD", str(nome or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = " ".join(t for t in n.split() if t not in _STOP_NOMI and not t.isdigit())
    return _ALIAS_NOMI.get(n, n)

def _similarita(a, b):
    if not a or not b: return 0.0
    if a == b: return 1.0
    ta, tb = set(a.split()), set(b.split())
    if ta <= tb or tb <= ta: return 0.92
    return difflib.SequenceMatcher(None, a, b).ratio()

def mappa_nomi(nomi_api, nomi_target):
    norm_t = {t: _norm_sq(t) for t in nomi_target}
    out = {}
    for n in nomi_api:
        na = _norm_sq(n)
        best, best_s = None, 0.0
        for t, nt in norm_t.items():
            s_ = _similarita(na, nt)
            if s_ > best_s:
                best, best_s = t, s_
        out[n] = best if best is not None and best_s >= 0.75 else n
    return out

def prima_tripla(riga, colonne):
    for cols in colonne:
        try:
            v = [float(riga[c]) for c in cols]
        except (KeyError, TypeError, ValueError): continue
        if all(np.isfinite(v)) and all(x > 1.0 for x in v): return v
    return None

def prima_coppia(riga, colonne):
    for a, b in colonne:
        try:
            v = [float(riga[a]), float(riga[b])]
        except (KeyError, TypeError, ValueError): continue
        if all(np.isfinite(v)) and all(x > 1.0 for x in v): return v
    return None

def _leggi_csv(url):
    ultimo = "errore sconosciuto"
    for tentativo in range(3):
        try:
            res = requests.get(url, timeout=TIMEOUT, headers=UA)
        except Exception as e:
            ultimo = f"connessione fallita ({type(e).__name__})"
        else:
            if res.status_code == 200:
                for enc in ("utf-8-sig", "latin-1"):
                    try:
                        d = pd.read_csv(io.BytesIO(res.content), encoding=enc, on_bad_lines="skip")
                        d.columns = [str(c).strip().lstrip("\ufeff") for c in d.columns]
                        return d, None
                    except UnicodeDecodeError: continue
                    except Exception as e: return None, f"file non leggibile ({type(e).__name__})"
                return None, "codifica non riconosciuta"
            ultimo = f"risposta HTTP {res.status_code}"
            if res.status_code < 500 and res.status_code != 429: return None, ultimo
        if tentativo < 2: time.sleep(2 + 3 * tentativo)
    return None, ultimo

# ============================================================
# DEVIG
# ============================================================
METODO_MARGINE = {"m": "proporzionale"}

def _devig_potenza(pi):
    lo, hi = 1.0, 10.0
    for _ in range(60):
        k = (lo + hi) / 2
        if (pi ** k).sum() > 1: lo = k
        else: hi = k
    p = pi ** ((lo + hi) / 2)
    return p / p.sum()

def _devig_shin(pi):
    B = pi.sum()
    def somma(z):
        return ((np.sqrt(z * z + 4 * (1 - z) * pi * pi / B) - z) / (2 * (1 - z))).sum()
    if somma(0.0) <= 1: return pi / B
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if somma(mid) > 1: lo = mid
        else: hi = mid
    z = (lo + hi) / 2
    p = (np.sqrt(z * z + 4 * (1 - z) * pi * pi / B) - z) / (2 * (1 - z))
    return p / p.sum()

def senza_margine(quote, metodo=None):
    pi = 1.0 / np.asarray(quote, dtype=float)
    metodo = metodo or METODO_MARGINE["m"]
    if metodo == "potenza": return _devig_potenza(pi)
    if metodo == "shin": return _devig_shin(pi)
    return pi / pi.sum()

def _diff_se(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return f"{d.mean():+.4f} ± {se:.4f}"

# ============================================================
# DATA LAYER
# ============================================================
@st.cache_data(ttl=6 * 3600)
def carica_storico(div, n_stagioni, giorno):
    oggi = date.today()
    anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    frames = []
    for k in range(n_stagioni):
        y = anno - k
        codice = f"{y % 100:02d}{(y + 1) % 100:02d}"
        d, _ = _leggi_csv(f"{BASE_URL}/mmz4281/{codice}/{div}.csv")
        if d is None or not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(d.columns): continue
        d["Stag"] = y
        frames.append(d)
    if not frames: return None
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def carica_fixtures(div):
    d, err = _leggi_csv(f"{BASE_URL}/fixtures.csv")
    if d is None: return None, err
    if not {"Div", "HomeTeam", "AwayTeam"}.issubset(d.columns): return None, "colonne attese non trovate"
    d = d[d["Div"] == div].dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce", format="mixed")
    return d.reset_index(drop=True), None

@st.cache_data(ttl=3600)
def scarica_calendario_api(chiave, codice):
    if not chiave: return None
    try:
        res = requests.get(f"https://api.football-data.org/v4/competitions/{codice}/matches", headers={"X-Auth-Token": chiave}, timeout=TIMEOUT)
        if res.status_code == 200: return res.json()
    except Exception: pass
    return None

def assegna_giornate(righe):
    conta = {}
    for r in righe:
        g = max(conta.get(r["Casa"], 0), conta.get(r["Trasferta"], 0)) + 1
        r["Giornata"] = g
        conta[r["Casa"]] = conta.get(r["Casa"], 0) + 1
        conta[r["Trasferta"]] = conta.get(r["Trasferta"], 0) + 1
    return righe

def costruisci_calendario(div, df_storico, fx, chiave, squadre_modello):
    dati = scarica_calendario_api(chiave, API_CODICI[div]) if chiave else None
    if dati and dati.get("matches"):
        nomi = {m["homeTeam"]["name"] for m in dati["matches"]} | {m["awayTeam"]["name"] for m in dati["matches"]}
        nomi = {n for n in nomi if n}
        mp = mappa_nomi(nomi, squadre_modello)
        righe = []
        for m in dati["matches"]:
            h, a = m["homeTeam"].get("name"), m["awayTeam"].get("name")
            if not h or not a: continue
            ft = (m.get("score") or {}).get("fullTime") or {}
            ts = pd.to_datetime(m["utcDate"], utc=True).tz_convert("Europe/Rome")
            righe.append({"Giornata": m.get("matchday") or 0, "Data": ts.strftime("%Y-%m-%d"), "Ora": ts.strftime("%H:%M"), "Casa": mp.get(h, h), "Trasferta": mp.get(a, a), "GolC": ft.get("home"), "GolT": ft.get("away"), "Stato": m.get("status", "")})
        return pd.DataFrame(righe), "football-data.org (calendario completo)"

    righe = []
    stag = df_storico[df_storico["Stag"] == df_storico["Stag"].max()]
    for _, r in stag.iterrows():
        righe.append({"Data": r["Date"].strftime("%Y-%m-%d"), "Ora": str(r["Time"])[:5] if "Time" in r and pd.notna(r.get("Time")) else "", "Casa": r["HomeTeam"], "Trasferta": r["AwayTeam"], "GolC": int(r["FTHG"]), "GolT": int(r["FTAG"]), "Stato": "FINISHED", "_d": r["Date"]})
    giocate = {(r["Data"], r["Casa"], r["Trasferta"]) for r in righe}
    if fx is not None:
        for _, r in fx.iterrows():
            d = r["Date"] if "Date" in r and pd.notna(r["Date"]) else pd.Timestamp.max
            key = (d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "", r["HomeTeam"], r["AwayTeam"])
            if key in giocate: continue
            righe.append({"Data": d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "", "Ora": str(r["Time"])[:5] if "Time" in r and pd.notna(r.get("Time")) else "", "Casa": r["HomeTeam"], "Trasferta": r["AwayTeam"], "GolC": None, "GolT": None, "Stato": "SCHEDULED", "_d": d})
    righe.sort(key=lambda r: r["_d"])
    righe = assegna_giornate(righe)
    return pd.DataFrame(righe).drop(columns="_d"), "football-data.co.uk (giocate + prossimo turno)"

@st.cache_data(ttl=3600)
def carica_storico_intl(chiave, fd_id, n_stagioni, giorno):
    if not chiave: return None
    oggi = date.today()
    anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    frames = []
    for k in range(n_stagioni):
        y = anno - k
        url = f"https://api.football-data.org/v4/competitions/{fd_id}/matches?season={y}&status=FINISHED"
        try:
            res = requests.get(url, headers={"X-Auth-Token": chiave}, timeout=TIMEOUT)
            if res.status_code == 200:
                matches = res.json().get("matches", [])
                righe = []
                for m in matches:
                    ft = (m.get("score") or {}).get("fullTime") or {}
                    if ft.get("home") is not None and ft.get("away") is not None:
                        righe.append({"Date": pd.to_datetime(m["utcDate"]).tz_localize(None), "HomeTeam": m["homeTeam"]["name"], "AwayTeam": m["awayTeam"]["name"], "FTHG": ft["home"], "FTAG": ft["away"], "Stag": y})
                if righe: frames.append(pd.DataFrame(righe))
        except Exception: pass
        time.sleep(6) 
    if not frames: return None
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def carica_odds_api(api_key, sport_key):
    if not api_key: return {}
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions=eu&markets=h2h&oddsFormat=decimal"
    try:
        res = requests.get(url, timeout=TIMEOUT)
        if res.status_code == 200:
            dati = res.json()
            quotes = {}
            for match in dati:
                home, away = match["home_team"], match["away_team"]
                h_odds, d_odds, a_odds = [], [], []
                for bm in match.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt["key"] == "h2h":
                            for out in mkt["outcomes"]:
                                if out["name"] == home: h_odds.append(out["price"])
                                elif out["name"] == away: a_odds.append(out["price"])
                                elif out["name"] == "Draw": d_odds.append(out["price"])
                if h_odds and d_odds and a_odds:
                    quotes[(home, away)] = [np.mean(h_odds), np.mean(d_odds), np.mean(a_odds)]
            return quotes
    except Exception: pass
    return {}

# ============================================================
# MODELLO DIXON-COLES
# ============================================================
_K = np.arange(11)
_LOGFACT = gammaln(_K + 1)
_I, _J = np.meshgrid(_K, _K, indexing="ij")
_TOT = _I + _J
_MASK = {"p1": _I > _J, "px": _I == _J, "p2": _I < _J, "over1.5": _TOT > 1.5, "over2.5": _TOT > 2.5, "over3.5": _TOT > 3.5, "btts": (_I > 0) & (_J > 0)}

def adatta_modello(df, data_rif, ridge, halflife, usa_dc=True, elo_diff=None):
    squadre = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    n = len(squadre)
    idx = {t: i for i, t in enumerate(squadre)}
    hi = df["HomeTeam"].map(idx).to_numpy()
    ai = df["AwayTeam"].map(idx).to_numpy()
    x = df["FTHG"].to_numpy(float)
    y = df["FTAG"].to_numpy(float)
    eta_gg = (data_rif - df["Date"]).dt.days.clip(lower=0).to_numpy(float)
    w = np.exp(-np.log(2) * eta_gg / halflife) if halflife and halflife > 0 else np.ones(len(df))
    use_elo = elo_diff is not None
    de = np.asarray(elo_diff, float) if use_elo else np.zeros(len(df))

    def f(th):
        mu, hm = th[0], th[1]
        att, dfn = th[2:2 + n], th[2 + n:2 + 2 * n]
        beta = th[2 + 2 * n] if use_elo else 0.0
        eh = mu + hm + att[hi] + dfn[ai] + beta * de
        ea = mu + att[ai] + dfn[hi] - beta * de
        lh, la = np.exp(eh), np.exp(ea)
        nll = -np.sum(w * (x * eh - lh + y * ea - la)) + ridge * (att @ att + dfn @ dfn)
        rh, ra = w * (x - lh), w * (y - la)
        g = np.zeros_like(th)
        g[0] = -(rh.sum() + ra.sum())
        g[1] = -rh.sum()
        g[2:2 + n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * ridge * att
        g[2 + n:2 + 2 * n] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * ridge * dfn
        if use_elo: g[2 + 2 * n] = -np.sum(de * (rh - ra))
        return nll, g

    th0 = np.zeros(2 + 2 * n + (1 if use_elo else 0))
    th0[0] = np.log(max((x.mean() + y.mean()) / 2, 0.3))
    th = minimize(f, th0, jac=True, method="L-BFGS-B").x
    mu, hm, att, dfn = th[0], th[1], th[2:2 + n], th[2 + n:2 + 2 * n]
    beta = float(th[2 + 2 * n]) if use_elo else 0.0

    rho = 0.0
    if usa_dc:
        lh = np.exp(mu + hm + att[hi] + dfn[ai] + beta * de)
        la = np.exp(mu + att[ai] + dfn[hi] - beta * de)
        m00, m01, m10, m11 = (x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)
        def neg(r):
            t = np.ones(len(x))
            t[m00] = np.maximum(1 - lh[m00] * la[m00] * r, 1e-6)
            t[m01] = np.maximum(1 + lh[m01] * r, 1e-6)
            t[m10] = np.maximum(1 + la[m10] * r, 1e-6)
            t[m11] = np.maximum(1 - r, 1e-6)
            return -np.sum(w * np.log(t))
        rho = float(minimize_scalar(neg, bounds=(-0.3, 0.3), method="bounded").x)

    return {"squadre": squadre, "idx": idx, "mu": float(mu), "hm": float(hm), "att": att, "dfn": dfn, "rho": rho, "beta": beta}

def lambda_partita(m, casa, trasf, d_elo=0.0):
    ic, it = m["idx"].get(casa), m["idx"].get(trasf)
    ac, dc = (m["att"][ic], m["dfn"][ic]) if ic is not None else (0.0, 0.0)
    at, dt = (m["att"][it], m["dfn"][it]) if it is not None else (0.0, 0.0)
    b = m.get("beta", 0.0) * d_elo
    return math.exp(m["mu"] + m["hm"] + ac + dt + b), math.exp(m["mu"] + at + dc - b)

def _pmf(lam): return np.exp(-lam + _K * np.log(np.maximum(lam, 1e-9)) - _LOGFACT)

def matrice_punteggi(lh, la, rho):
    M = np.outer(_pmf(lh), _pmf(la))
    if rho != 0:
        M[0, 0] *= max(1 - lh * la * rho, 1e-6)
        M[0, 1] *= max(1 + lh * rho, 1e-6)
        M[1, 0] *= max(1 + la * rho, 1e-6)
        M[1, 1] *= max(1 - rho, 1e-6)
    return M / M.sum()

def mercati(M):
    d = {"p1": float(np.tril(M, -1).sum()), "px": float(np.trace(M)), "p2": float(np.triu(M, 1).sum()), "btts": float(M[1:, 1:].sum())}
    for s in (1.5, 2.5, 3.5): d[f"over{s}"] = float(M[_TOT > s].sum())
    return d

def gol_attesi(M): return float((M.sum(axis=1) * _K).sum()), float((M.sum(axis=0) * _K).sum())

def matrice_frequenze(df):
    cm = np.zeros((11, 11))
    np.add.at(cm, (np.minimum(df["FTHG"].to_numpy(), 10), np.minimum(df["FTAG"].to_numpy(), 10)), 1)
    return (cm + 0.05) / (cm.sum() + 0.05 * 121)

def matrice_ancorata(p1x2, p_over25, rho, start):
    def err(th):
        mk = mercati(matrice_punteggi(math.exp(th[0]), math.exp(th[1]), rho))
        e = ((mk["p1"] - p1x2[0]) ** 2 + (mk["px"] - p1x2[1]) ** 2 + (mk["p2"] - p1x2[2]) ** 2)
        if p_over25 is not None: e += (mk["over2.5"] - p_over25) ** 2
        return e
    res = minimize(err, np.log(np.asarray(start, float)), method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-10, "maxiter": 250})
    return matrice_punteggi(math.exp(res.x[0]), math.exp(res.x[1]), rho)

# ============================================================
# PARAMETRI E BACKTEST
# ============================================================
def esegui_backtest_df(df, ridge, halflife, usa_dc, passo):
    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) < 2: return pd.DataFrame()
    t = df.loc[df["Stag"] == stagioni[1], "Date"].min()
    fine = df["Date"].max()
    righe = []
    while t <= fine:
        blocco = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=passo))]
        train = df[df["Date"] < t]
        if len(blocco) and len(train) >= 150:
            m = adatta_modello(train, t, ridge, halflife, usa_dc)
            esiti_tr = np.where(train["FTHG"] > train["FTAG"], 0, np.where(train["FTHG"] == train["FTAG"], 1, 2))
            freq = np.bincount(esiti_tr, minlength=3) / len(esiti_tr)
            for _, r in blocco.iterrows():
                lh, la = lambda_partita(m, r["HomeTeam"], r["AwayTeam"])
                mk = mercati(matrice_punteggi(lh, la, m["rho"]))
                esito = 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2)
                qp, qc = prima_tripla(r, COL_PRESA), prima_tripla(r, COL_CHIUSURA)
                pm = senza_margine(qc) if qc else [np.nan] * 3
                righe.append({"Date": r["Date"], "Partita": f"{r['HomeTeam']} - {r['AwayTeam']}", "y": esito, "p1": mk["p1"], "px": mk["px"], "p2": mk["p2"], "f1": freq[0], "fx": freq[1], "f2": freq[2], "q1": qp[0] if qp else np.nan, "qx": qp[1] if qp else np.nan, "q2": qp[2] if qp else np.nan, "m1": pm[0], "mx": pm[1], "m2": pm[2]})
        t += pd.Timedelta(days=passo)
    return pd.DataFrame(righe)

def _ll_brier(P, y):
    P = np.asarray(P, float)
    ll = -np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)))
    br = np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))
    return ll, br

def metriche_backtest(bt):
    y = bt["y"].to_numpy(int)
    out = {"n": len(bt)}
    out["modello"] = _ll_brier(bt[["p1", "px", "p2"]].to_numpy(), y)
    out["frequenze"] = _ll_brier(bt[["f1", "fx", "f2"]].to_numpy(), y)
    ok = bt[["m1", "mx", "m2"]].notna().all(axis=1).to_numpy()
    if ok.sum() > 30:
        out["mercato"] = _ll_brier(bt.loc[ok, ["m1", "mx", "m2"]].to_numpy(), y[ok])
        out["modello_su_mercato"] = _ll_brier(bt.loc[ok, ["p1", "px", "p2"]].to_numpy(), y[ok])
        out["n_mercato"] = int(ok.sum())
    return out

def scegli_parametri(df):
    default = {"ridge": 2.0, "halflife": 365, "log_loss": None, "ricerca": [], "auto": False}
    if df is None or df["Stag"].nunique() < 2: return default
    ricerca = []
    for r_ in RIDGE_GRID:
        for h_ in HALFLIFE_GRID:
            bt = esegui_backtest_df(df, r_, h_, True, 28)
            if not bt.empty:
                ricerca.append({"Shrinkage λ": r_, "Emivita (gg)": str(h_) if h_ else "nessuna", "Log loss": round(metriche_backtest(bt)["modello"][0], 4), "_r": r_, "_h": h_})
    if not ricerca: return default
    best = min(ricerca, key=lambda z: z["Log loss"])
    return {"ridge": best["_r"], "halflife": best["_h"], "log_loss": best["Log loss"], "auto": True, "ricerca": [{k: v for k, v in z.items() if not k.startswith("_")} for z in ricerca]}

@st.cache_data(ttl=6 * 3600, show_spinner="Ottimizzo i parametri (solo la prima volta di giornata)...")
def parametri_auto(div, giorno): return scegli_parametri(carica_storico(div, N_STAGIONI, giorno))

@st.cache_data(ttl=6 * 3600)
def modello_corrente(div, ridge, halflife, giorno):
    df = carica_storico(div, N_STAGIONI, giorno)
    if df is None or len(df) < 100: return None
    return adatta_modello(df, pd.Timestamp.now().normalize(), ridge, halflife, True)

@st.cache_data(ttl=6 * 3600, show_spinner=False)
def backtest_cached(div, ridge, halflife, passo, giorno):
    df = carica_storico(div, N_STAGIONI, giorno)
    return None if df is None else esegui_backtest_df(df, ridge, halflife, True, passo)

# ============================================================
# FUNZIONI PER I PRONOSTICI E LA GIOCATA CONSIGLIATA
# ============================================================
_ESITI_MASK = {"1": _MASK["p1"], "X": _MASK["px"], "2": _MASK["p2"], "1X": _MASK["p1"] | _MASK["px"], "X2": _MASK["px"] | _MASK["p2"], "12": _MASK["p1"] | _MASK["p2"]}
_MERC_MASK = {"Goal": _MASK["btts"], "No Goal": ~_MASK["btts"], "Over 1.5": _MASK["over1.5"], "Under 1.5": ~_MASK["over1.5"], "Over 2.5": _MASK["over2.5"], "Under 2.5": ~_MASK["over2.5"], "Over 3.5": _MASK["over3.5"], "Under 3.5": ~_MASK["over3.5"]}
LINEE_OU = ("Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5")
NOMI_ESITO = {"1": "1 (Casa)", "X": "X (Pareggio)", "2": "2 (Trasferta)"}
NOMI_SCELTA = {"1": "1 (vince la casa)", "X": "X (pareggio)", "2": "2 (vince la trasferta)", "1X": "1X (la casa non perde)", "X2": "X2 (la trasferta non perde)", "12": "12 (non pareggia)"}

def _esiti(xs, ys):
    esito = np.where(xs > ys, 0, np.where(xs == ys, 1, 2))
    outc = {"over1.5": (xs + ys > 1.5).astype(float), "over2.5": (xs + ys > 2.5).astype(float), "over3.5": (xs + ys > 3.5).astype(float), "btts": ((xs > 0) & (ys > 0)).astype(float)}
    return esito, outc

def _riga_prob(etichetta, p, freq=None):
    r = {"Mercato": etichetta, "Probabilità %": round(float(p) * 100, 1), "Quota equa": round(1 / p, 2) if p > 1e-6 else None}
    if freq is not None and etichetta in freq:
        r["Media storica %"] = round(freq[etichetta] * 100, 1)
        r["Differenza (punti)"] = round(float(p) * 100 - freq[etichetta] * 100, 1)
    return r

def matrice_congiunta(M_1x2, M_gol):
    out = M_gol.copy()
    for chiave in ("p1", "px", "p2"):
        mask = _MASK[chiave]
        pg, po = M_gol[mask].sum(), M_1x2[mask].sum()
        if pg > 0: out[mask] *= po / pg
    return out / out.sum()

def frequenze_combo(df):
    gh, ga = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
    esiti = {"1": gh > ga, "X": gh == ga, "2": gh < ga}
    esiti["1X"] = esiti["1"] | esiti["X"]; esiti["X2"] = esiti["X"] | esiti["2"]; esiti["12"] = esiti["1"] | esiti["2"]
    gol = (gh > 0) & (ga > 0)
    merc = {"Goal": gol, "No Goal": ~gol}
    for ln in (1.5, 2.5, 3.5):
        ov = (gh + ga) > ln
        merc[f"Over {ln}"] = ov; merc[f"Under {ln}"] = ~ov
    out = {e: float(m.mean()) for e, m in esiti.items()}
    out.update({f"{e} + {m}": float((esiti[e] & merc[m]).mean()) for e in esiti for m in merc})
    out.update({f"{g_} + {o_}": float((merc[g_] & merc[o_]).mean()) for g_ in ("Goal", "No Goal") for o_ in LINEE_OU})
    tot = gh + ga
    out.update({f"{a}-{b} gol": float(((tot >= a) & (tot <= b)).mean()) for a, b in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (3, 6)]})
    for nome, g_ in (("Casa", gh), ("Trasferta", ga)):
        for k in (0, 1, 2): out[f"{nome} {k} gol"] = float((g_ == k).mean())
        out[f"{nome} 3+ gol"] = float((g_ >= 3).mean())
        for a, b in [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)]: out[f"{nome} {a}-{b} gol"] = float(((g_ >= a) & (g_ <= b)).mean())
    return out

def _sopra_media(finestre, freq, chiave_fn):
    medie = {k: (freq.get(chiave_fn(k)) if freq else None) for k in finestre}
    if freq and all(v is not None for v in medie.values()):
        k = max(finestre, key=lambda x: finestre[x] - medie[x])
        return k, finestre[k], medie[k]
    k = max(finestre, key=finestre.get)
    return k, finestre[k], None

def giocata_consigliata(pr, soglia_x=None):
    pp = pr["p"]
    gap12 = abs(pp["1"] - pp["2"]) * 100
    equilibrata = gap12 <= SOGLIA_EQUILIBRIO_GAP * 100 and pp["X"] >= SOGLIA_EQUILIBRIO_X

    if pr.get("x_regola"):
        return "X", f"X al {pp['X']*100:.0f}%", "soglia X superata (dal backtest)"
    if equilibrata and pp["X"] >= SOGLIA_X_FORZATA:
        return "X", f"X al {pp['X']*100:.0f}%", "partita molto equilibrata: il pareggio è concreto"
    if equilibrata:
        best_d = pr["doppia_x_best"]
        return best_d, f"{best_d} al {pp[best_d]*100:.0f}%", "equilibrata: meglio coprire il pareggio con la doppia"
    p_fav = max(pp["1"], pp["2"])
    chiave = "1" if pp["1"] > pp["2"] else "2"
    if p_fav >= SOGLIA_FAVORITA_NETTA:
        return chiave, f"{chiave} al {p_fav*100:.0f}%", "favorita netta"
    return chiave, f"{chiave} al {p_fav*100:.0f}% (debole)", "favorita debole: bassa fiducia"

def tabelle_pronostico(M_1x2, M_gol, freq=None, soglia_x=None):
    Mj = matrice_congiunta(M_1x2, M_gol)
    p = {k: float(M_1x2[m].sum()) for k, m in _ESITI_MASK.items()}
    out = {"p": p}
    best = max(("1", "X", "2"), key=lambda k: p[k])
    out["x_regola"] = False
    if soglia_x is not None and p["X"] >= soglia_x and best != "X":
        best = "X"; out["x_regola"] = True
    out["pronostico"] = best
    out["doppia_x_best"] = max(("1X", "X2"), key=lambda k: p[k])
    out["doppie"] = pd.DataFrame([_riga_prob(k, p[k], freq) for k in ("1X", "X2", "12")]).sort_values("Probabilità %", ascending=False)
    out["doppia_best"] = max(("1X", "X2", "12"), key=lambda k: p[k])

    def combo(esiti, mercati):
        righe = [(f"{e} + {m}", float(Mj[_ESITI_MASK[e] & _MERC_MASK[m]].sum())) for e in esiti for m in mercati]
        return pd.DataFrame([_riga_prob(l, v, freq) for l, v in sorted(righe, key=lambda r: -r[1])])
    
    out["combo_goal"] = combo(("1", "X", "2"), ("Goal", "No Goal"))
    out["combo_ou"] = combo(("1", "X", "2"), ("Over 2.5", "Under 2.5"))
    out["combo_doppie"] = combo(("1X", "X2", "12"), ("Goal", "No Goal", "Over 2.5", "Under 2.5"))
    out["combo_ou_linee"] = combo(("1", "X", "2"), LINEE_OU)
    out["combo_doppie_goal"] = combo(("1X", "X2", "12"), ("Goal", "No Goal"))
    out["combo_doppie_ou"] = combo(("1X", "X2", "12"), LINEE_OU)
    
    out["multigol_tot"] = pd.DataFrame([_riga_prob(f"{a}-{b} gol", M_gol[(_TOT >= a) & (_TOT <= b)].sum(), freq) for a, b in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (3, 6)]])
    larghi3 = {f"{a}-{b}": float(M_gol[(_TOT >= a) & (_TOT <= b)].sum()) for a, b in [(1, 3), (2, 4), (3, 5)]}
    best_key, best_p, best_f = _sopra_media(larghi3, freq, lambda k: f"{k} gol")
    out["multigol_best"] = best_key; out["multigol_best_p"] = best_p; out["multigol_best_f"] = best_f
    return out

# ============================================================
# TEMPI E TEMPI ESATTI
# ============================================================
def tabella_split(df, s=10.0):
    S = np.zeros((11, 11)); pi = 0.45; d = None
    if {"HTHG", "HTAG"}.issubset(df.columns):
        d = df.dropna(subset=["HTHG", "HTAG"])
        if len(d) < 100: d = None
    if d is not None:
        ft = (d["FTHG"] + d["FTAG"]).to_numpy(int); h1 = (d["HTHG"] + d["HTAG"]).to_numpy(int)
        pi = float(min(max(h1.sum() / max(ft.sum(), 1), 0.3), 0.6))
    for t in range(11):
        prior = np.array([math.comb(t, k) * pi ** k * (1 - pi) ** (t - k) for k in range(t + 1)])
        if d is None: S[t, :t + 1] = prior; continue
        m = ft == t
        counts = np.bincount(np.clip(h1[m], 0, t), minlength=t + 1)[:t + 1].astype(float)
        S[t, :t + 1] = (counts + s * prior) / (m.sum() + s)
    return S

def probs_tempi(M_gol, S):
    tot = _TOT
    pt = np.array([M_gol[tot == t].sum() for t in range(11)])
    pt = pt / pt.sum(); joint = np.zeros((11, 11))
    for t in range(11):
        for k in range(t + 1): joint[k, t - k] += pt[t] * S[t, k]
    return joint.sum(axis=1), joint.sum(axis=0), joint

def tabelle_tempi(M_gol, S, freq=None):
    p1, p2, pm = probs_tempi(M_gol, S); out = {}
    def righe(p, nome):
        rows = [_riga_prob(f"{nome} 0 gol", p[0], freq), _riga_prob(f"{nome} 1 gol", p[1], freq)]
        rows += [_riga_prob(f"{nome} {a}-{b} gol", p[a:b + 1].sum(), freq) for a, b in [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4)]]
        rows.append(_riga_prob(f"{nome} almeno 1 gol", 1 - p[0], freq)); rows.append(_riga_prob(f"{nome} almeno 2 gol", 1 - p[0] - p[1], freq))
        return pd.DataFrame(rows)
    out["t1"] = righe(p1, "1° tempo"); out["t2"] = righe(p2, "2° tempo")
    out["piu_gol"] = pd.DataFrame([_riga_prob("Primo tempo più gol", np.tril(pm, -1).sum(), freq), _riga_prob("Gol uguali nei due tempi", np.trace(pm), freq), _riga_prob("Secondo tempo più gol", np.triu(pm, 1).sum(), freq)])
    for chiave, p, nome_t in (("t1", p1, "1° tempo"), ("t2", p2, "2° tempo")):
        w2 = {f"{a}-{b}": float(p[a:b + 1].sum()) for a, b in [(0, 1), (1, 2), (2, 3)]}
        k, v, f = _sopra_media(w2, freq, lambda kk: f"{nome_t} {kk} gol")
        out[chiave + "_best"] = k; out[chiave + "_best_p"] = v; out[chiave + "_best_f"] = f
    piu = {"1° tempo": float(np.tril(pm, -1).sum()), "Pari": float(np.trace(pm)), "2° tempo": float(np.triu(pm, 1).sum())}
    out["piu_best"] = max(piu, key=piu.get); out["piu_best_p"] = max(piu.values())
    return out

def distribuzione_tempo(M, f_casa, f_trasf):
    n = M.shape[0]; k = np.arange(n)
    Bc = binom.pmf(k[None, :], k[:, None], f_casa); Bt = binom.pmf(k[None, :], k[:, None], f_trasf)
    return Bc.T @ M @ Bt

def esatti_per_tempo(M, df, quanti=4):
    if df is None or "HTHG" not in df.columns or "HTAG" not in df.columns: return None
    d = df.dropna(subset=["HTHG", "HTAG"])
    if len(d) < 100 or d["FTHG"].sum() == 0 or d["FTAG"].sum() == 0: return None
    fc = float(d["HTHG"].sum() / d["FTHG"].sum()); ft = float(d["HTAG"].sum() / d["FTAG"].sum())
    h1_hist = np.zeros((6, 6)); h2_hist = np.zeros((6, 6))
    for r in d.itertuples():
        a1, b1 = min(int(r.HTHG), 5), min(int(r.HTAG), 5); a2, b2 = min(int(r.FTHG - r.HTHG), 5), min(int(r.FTAG - r.HTAG), 5)
        h1_hist[a1, b1] += 1; h2_hist[a2, b2] += 1
    h1_hist = (h1_hist + 0.5) / (h1_hist.sum() + 0.5 * 36); h2_hist = (h2_hist + 0.5) / (h2_hist.sum() + 0.5 * 36)
    righe = []
    for nome, H, hist in (("1° tempo", distribuzione_tempo(M, fc, ft), h1_hist), ("2° tempo", distribuzione_tempo(M, 1 - fc, 1 - ft), h2_hist)):
        Hn = H / H.sum()
        ordine = sorted(((i, j, Hn[i, j]) for i in range(6) for j in range(6)), key=lambda z: -z[2])[:quanti]
        riga = {"Tempo": nome}
        for pos, (i, j, p) in enumerate(ordine): riga[["Più probabile", "Seconda", "Terza", "Quarta"][pos]] = f"{i}-{j} ({p * 100:.0f}%)"
        best, best_delta = None, 0.0
        for i in range(6):
            for j in range(6):
                if Hn[i, j] < 0.02 or hist[i, j] < 0.005: continue
                delta = Hn[i, j] - hist[i, j]
                if delta > best_delta: best_delta = delta; best = (i, j, Hn[i, j], hist[i, j])
        if best:
            i, j, p, ph = best
            riga["Sorpresa"] = f"{i}-{j} ({p*100:.0f}% vs {ph*100:.0f}% media)"
        righe.append(riga)
    return righe

# ============================================================
# FUNZIONI LIVE
# ============================================================
def stats_attese(df, casa, trasf, k=5):
    out = {}
    for nome, (h, a) in STAT_COLONNE.items():
        if h not in df.columns or a not in df.columns: continue
        d = df.dropna(subset=[h, a])
        if d.empty: continue
        lega_h, lega_a = d[h].mean(), d[a].mean()
        def sm(serie, lega): return (serie.sum() + k * lega) / (len(serie) + k)
        dc, dt = d[d["HomeTeam"] == casa], d[d["AwayTeam"] == trasf]
        out[nome] = (round((sm(dc[h], lega_h) + sm(dt[h], lega_h)) / 2, 1), round((sm(dt[a], lega_a) + sm(dc[a], lega_a)) / 2, 1))
    return out

def ultimi_5_risultati(df, squadra):
    d = df[(df["HomeTeam"] == squadra) | (df["AwayTeam"] == squadra)].sort_values("Date").tail(5)
    res = []
    for r in d.itertuples():
        if r.HomeTeam == squadra:
            esito = "V" if r.FTHG > r.FTAG else ("N" if r.FTHG == r.FTAG else "P")
        else:
            esito = "V" if r.FTAG > r.FTHG else ("N" if r.FTHG == r.FTAG else "P")
        res.append(esito)
    return res

def crea_heatmap(M):
    M_ridotta = M[:6, :6] * 100
    fig = go.Figure(data=go.Heatmap(
        z=M_ridotta, x=[str(i) for i in range(6)], y=[str(i) for i in range(6)],
        colorscale='Blues', text=np.round(M_ridotta, 1), texttemplate="%{text}%",
        hovertemplate="Risultato: %{y}-%{x}<br>Probabilità: %{z:.1f}%<extra></extra>"
    ))
    fig.update_layout(title="Matrice dei Risultati Esatti", xaxis_title="Gol Trasferta", yaxis_title="Gol Casa", yaxis_autorange="reversed", template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def tabella_classifica(df_stag):
    righe = {}
    for r in df_stag.itertuples():
        for sq, gf, gs in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            d = righe.setdefault(sq, {"G": 0, "Pt": 0, "GF": 0, "GS": 0})
            d["G"] += 1; d["GF"] += gf; d["GS"] += gs
            d["Pt"] += 3 if gf > gs else (1 if gf == gs else 0)
    if not righe: return pd.DataFrame()
    out = pd.DataFrame([{"Squadra": sq, "Giocate": v["G"], "Punti": v["Pt"], "Media GF": round(v["GF"] / v["G"], 2), "Media GS": round(v["GS"] / v["G"], 2), "DR": round((v["GF"] - v["GS"]) / v["G"], 2)} for sq, v in righe.items()])
    return out.sort_values(["Punti", "DR"], ascending=False).reset_index(drop=True)

def distribuzione_live(casa, trasf, modello, pesi, M_freq_live, quote_fixtures):
    lh, la = lambda_partita(modello, casa, trasf)
    Mb = matrice_punteggi(lh, la, modello["rho"])
    q = quote_fixtures.get((casa, trasf))
    if q is not None:
        Ma = matrice_ancorata(q, None, modello["rho"], (lh, la))
        return {"M_1x2": Ma, "M_gol": Ma, "M_es": Ma, "ancorata": True, "fonte": "ancorata alle quote reali"}
    Mg = pesi["gol"] * Mb + (1 - pesi["gol"]) * M_freq_live
    Me = pesi["esatto"] * Mb + (1 - pesi["esatto"]) * M_freq_live
    return {"M_1x2": Mb, "M_gol": Mg, "M_es": Me, "ancorata": False, "fonte": "modello + frequenze storiche"}

# ============================================================
# FUNZIONI DI SUPPORTO PER I PESI (Mancavano!)
# ============================================================
def pesi_ottimi(res):
    xs, ys = res["xs"], res["ys"]
    esito, outc = _esiti(xs, ys)
    Mb, Mf = res["Ms"]["Modello"], res["Ms"]["Frequenze"]
    gol, esatto = {}, {}
    for w in W_GRID:
        Mw = w * Mb + (1 - w) * Mf
        gol[w] = float(np.mean([_vettore_perdita(Mw, k_, xs, ys, outc, esito).mean() for k_ in ("over1.5", "over2.5", "over3.5", "btts")]))
        esatto[w] = float(_vettore_perdita(Mw, "exact", xs, ys, outc, esito).mean())
    return {"gol": min(gol, key=gol.get), "esatto": min(esatto, key=esatto.get), "n": len(xs), "auto": True}

def scegli_metodo_margine(df):
    metodi = ("proporzionale", "potenza", "shin")
    perdite = {m: {"ap": [], "ch": []} for m in metodi}
    for _, r in df.iterrows():
        es = 0 if r["FTHG"] > r["FTAG"] else (1 if r["FTHG"] == r["FTAG"] else 2)
        qa, qc = prima_tripla(r, COL_PRESA), prima_tripla(r, COL_CHIUSURA)
        for m in metodi:
            if qa: perdite[m]["ap"].append(-math.log(max(float(senza_margine(qa, m)[es]), 1e-9)))
            if qc: perdite[m]["ch"].append(-math.log(max(float(senza_margine(qc, m)[es]), 1e-9)))
    if len(perdite["proporzionale"]["ap"]) < 100: return "proporzionale", pd.DataFrame()
    righe = []
    base = np.array(perdite["proporzionale"]["ap"])
    for m in metodi:
        ap = np.array(perdite[m]["ap"]); ch = np.array(perdite[m]["ch"])
        righe.append({"Metodo": m, "Partite": len(ap), "Log loss apertura": round(float(ap.mean()), 4), "Differenza (± err.)": "" if m == "proporzionale" else _diff_se(ap, base), "Log loss chiusura": round(float(ch.mean()), 4) if len(ch) else None})
    best = min(metodi, key=lambda m: float(np.mean(perdite[m]["ap"])))
    d_best = np.array(perdite[best]["ap"]) - base
    se = d_best.std(ddof=1) / math.sqrt(len(d_best))
    scelto = best if (best != "proporzionale" and d_best.mean() < 0 and abs(d_best.mean()) >= 1.0 * se) else "proporzionale"
    tab = pd.DataFrame(righe)
    tab["Usato dall'app"] = ["sì" if r["Metodo"] == scelto else "" for r in righe]
    return scelto, tab

@st.cache_data(ttl=6 * 3600, show_spinner="Calcolo pesi dal backtest (solo la prima volta)...")
def pesi_blend_auto(div, ridge, halflife, giorno):
    df = carica_storico(div, N_STAGIONI_TEST, giorno)
    metodo, tab_margine = scegli_metodo_margine(df) if df is not None else ("proporzionale", pd.DataFrame())
    METODO_MARGINE["m"] = metodo
    res = esegui_test_completo(df, ridge, halflife, 28) if df is not None else None
    if res is None or len(res["xs"]) < 100:
        return {"gol": 0.5, "esatto": 0.75, "n": 0, "auto": False, "metodo_margine": metodo, "tab_margine": tab_margine}
    w = pesi_ottimi(res)
    w["metodo_margine"] = metodo; w["tab_margine"] = tab_margine
    return w

def _vettore_perdita(Mv, chiave, xs, ys, outc, esito):
    if chiave == "1x2":
        P = np.stack([_p_mercato(Mv, "p1"), _p_mercato(Mv, "px"), _p_mercato(Mv, "p2")], axis=1)
        return -np.log(np.clip(P[np.arange(len(esito)), esito], 1e-9, 1))
    if chiave == "exact":
        return -np.log(np.clip(Mv[np.arange(len(Mv)), xs, ys], 1e-12, 1))
    p = np.clip(_p_mercato(Mv, chiave), 1e-9, 1 - 1e-9)
    o = outc[chiave]
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))

def _p_mercato(Ms, chiave): return (Ms * _MASK[chiave]).sum(axis=(1, 2))

def esegui_test_completo(df, ridge, halflife, passo, inizio=None):
    stagioni = sorted(df["Stag"].unique())
    if len(stagioni) < 2: return None
    t = inizio if inizio is not None else df.loc[df["Stag"] == stagioni[1], "Date"].min()
    fine = df["Date"].max()
    rec = {"Partita": [], "xs": [], "ys": [], "ok": [], "ou_ok": [], "m1x2": [], "mou": [], "min_storia": [], "date": []}
    Ms = {"Modello": [], "Ancorato": [], "Frequenze": []}
    while t <= fine:
        blocco = df[(df["Date"] >= t) & (df["Date"] < t + pd.Timedelta(days=passo))]
        train = df[(df["Date"] < t) & (df["Date"] >= t - pd.Timedelta(days=FINESTRA_GIORNI))]
        if len(blocco) and len(train) >= 150:
            m = adatta_modello(train, t, ridge, halflife, True)
            cnt = pd.concat([train["HomeTeam"], train["AwayTeam"]]).value_counts()
            m_freq = matrice_frequenze(train)
            for _, r in blocco.iterrows():
                h, a = r["HomeTeam"], r["AwayTeam"]
                lh, la = lambda_partita(m, h, a)
                Mb = matrice_punteggi(lh, la, m["rho"])
                qp, qo = prima_tripla(r, COL_PRESA), prima_coppia(r, COL_OU_PRESA)
                if qp:
                    p_o = float(senza_margine(qo)[0]) if qo else None
                    Ma = matrice_ancorata(senza_margine(qp), p_o, m["rho"], (lh, la))
                else:
                    Ma = np.full((11, 11), np.nan)
                qc, qoc = prima_tripla(r, COL_CHIUSURA), prima_coppia(r, COL_OU_CHIUSURA)
                rec["Partita"].append(f"{h} - {a}")
                rec["date"].append(r["Date"])
                rec["xs"].append(min(int(r["FTHG"]), 10))
                rec["ys"].append(min(int(r["FTAG"]), 10))
                rec["ok"].append(bool(qp))
                rec["ou_ok"].append(bool(qp and qoc))
                rec["m1x2"].append(senza_margine(qc) if qc else np.full(3, np.nan))
                rec["mou"].append(float(senza_margine(qoc)[0]) if qoc else np.nan)
                rec["min_storia"].append(int(min(cnt.get(h, 0), cnt.get(a, 0))))
                for nome, Mv in {"Modello": Mb, "Ancorato": Ma, "Frequenze": m_freq}.items(): Ms[nome].append(Mv)
        t += pd.Timedelta(days=passo)
    if not rec["xs"]: return None
    out = {k: np.array(v) for k, v in rec.items() if k != "Partita"}
    out["Partita"] = rec["Partita"]
    out["Ms"] = {k: np.array(v) for k, v in Ms.items()}
    return out

# ============================================================
# UI PRINCIPALE
# ============================================================
try: _chiave_fd = st.secrets.get("FOOTBALL_DATA_API_KEY", "")
except Exception: _chiave_fd = ""

try: _chiave_odds = st.secrets.get("THE_ODDS_API_KEY", "")
except Exception: _chiave_odds = ""

st.sidebar.markdown("### ⚙ Configurazione API")
chiave_fd = st.sidebar.text_input("🔑 football-data.org", value=_chiave_fd, type="password")
chiave_odds = st.sidebar.text_input("🔑 The Odds API", value=_chiave_odds, type="password")
st.sidebar.caption("Le chiavi servono per le competizioni internazionali e le quote live.")
st.sidebar.markdown("---")

# Selezione Competizione
tipo_comp = st.sidebar.radio("Tipo di competizione", ["Campionati Nazionali", "Competizioni Internazionali"])
if tipo_comp == "Campionati Nazionali":
    campionato = st.selectbox("Scegli il torneo", list(LEAGUES.keys()), label_visibility="collapsed")
    div = LEAGUES[campionato]
    is_intl = False
else:
    campionato = st.selectbox("Scegli il torneo", list(INTL_COMPS.keys()), label_visibility="collapsed")
    div = campionato
    is_intl = True

giorno = date.today().isoformat()

# Caricamento Dati
if is_intl:
    if not chiave_fd:
        st.warning("Inserisci la chiave di football-data.org per caricare i dati delle competizioni internazionali.")
        st.stop()
    df_storico = carica_storico_intl(chiave_fd, INTL_COMPS[campionato]["fd_id"], N_STAGIONI, giorno)
    if df_storico is None or len(df_storico) < 50:
        st.warning(f"Dati insufficienti per {campionato}. Prova a cambiare competizione o stagione.")
        st.stop()
    quote_fixtures_raw = carica_odds_api(chiave_odds, INTL_COMPS[campionato]["odds_key"])
    squadre_modello = sorted(set(df_storico["HomeTeam"]) | set(df_storico["AwayTeam"]))
    all_odds_teams = set([k[0] for k in quote_fixtures_raw.keys()] + [k[1] for k in quote_fixtures_raw.keys()])
    mappa = mappa_nomi(list(all_odds_teams), squadre_modello)
    quote_fixtures = {}
    for (h, a), q in quote_fixtures_raw.items():
        mh, ma = mappa.get(h, h), mappa.get(a, a)
        if mh in squadre_modello and ma in squadre_modello:
            quote_fixtures[(mh, ma)] = q
else:
    df_storico = carica_storico(div, N_STAGIONI, giorno)
    if df_storico is None or len(df_storico) < 100:
        st.warning("Impossibile scaricare lo storico. Riprova tra poco.")
        st.stop()
    quote_fixtures = {}

# Addestramento Modello
par = parametri_auto(div, giorno) if not is_intl else {"ridge": 2.0, "halflife": 365}
modello = adatta_modello(df_storico, pd.Timestamp.now().normalize(), par["ridge"], par["halflife"], True)

# Costruzione Calendario
if is_intl:
    calendario = pd.DataFrame()
    if chiave_fd:
        url = f"https://api.football-data.org/v4/competitions/{INTL_COMPS[campionato]['fd_id']}/matches?status=SCHEDULED"
        res = requests.get(url, headers={"X-Auth-Token": chiave_fd}, timeout=TIMEOUT)
        if res.status_code == 200:
            righe = []
            for m in res.json().get("matches", []):
                ts = pd.to_datetime(m["utcDate"], utc=True).tz_convert("Europe/Rome")
                righe.append({"Data": ts.strftime("%Y-%m-%d"), "Ora": ts.strftime("%H:%M"), "Casa": m["homeTeam"]["name"], "Trasferta": m["awayTeam"]["name"], "Stato": m["status"]})
            calendario = pd.DataFrame(righe)
else:
    fx, fx_err = carica_fixtures(div)
    calendario, fonte_cal = costruisci_calendario(div, df_storico, fx, chiave_fd, modello["squadre"])

pesi = {"gol": 0.5, "esatto": 0.75} if is_intl else pesi_blend_auto(div, par["ridge"], par["halflife"], giorno)
M_freq_live = matrice_frequenze(df_storico)
freq_combo = frequenze_combo(df_storico)
split_live = tabella_split(df_storico)
oggi_ts = pd.Timestamp.now().normalize()

tabs = st.tabs(["📅 Calendario", "📋 Pronostici", "🏆 Classifica", "⚡ Value Finder", "🧮 Kelly"])

def mostra_analisi(casa, trasf, data, ora):
    d = distribuzione_live(casa, trasf, modello, pesi, M_freq_live, quote_fixtures)
    lh, la = gol_attesi(d["M_gol"])
    M = d["M_es"]; mk = mercati(d["M_1x2"]); mg = mercati(d["M_gol"])
    pr = tabelle_pronostico(d["M_1x2"], d["M_gol"], freq_combo, None)
    pt = tabelle_tempi(d["M_gol"], split_live, freq_combo)

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); border: 1px solid #312e81; border-radius: 20px; padding: 18px 25px; margin-top: 20px; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 800; font-size: 1.2rem; color: #f8fafc;">📋 {casa} vs {trasf}</span>
        <span style="color: #94a3b8; font-size: 0.95rem;">📅 {data} {ora}</span>
    </div>
    """, unsafe_allow_html=True)

    pp = pr["p"]; chiave_g, testo_g, motivo_g = giocata_consigliata(pr, None); top = pr["pronostico"]; p_top = pp[top]
    livello = "media"
    if p_top >= SOGLIA_FIDUCIA_ALTA and not chiave_g.startswith(("1X", "X2")): livello = "alta"
    elif chiave_g.startswith(("1X", "X2")): livello = "media"
    elif p_top < SOGLIA_FIDUCIA_MEDIA: livello = "bassa"
    senza_storico = casa not in modello["idx"] or trasf not in modello["idx"]
    if senza_storico: livello = "media" if livello == "alta" else "bassa"
    
    eq_1x2 = f"1 {pp['1']*100:.0f}% · X {pp['X']*100:.0f}% · 2 {pp['2']*100:.0f}%"
    emoji = {"alta": "🟢", "media": "🟡", "bassa": "🔴"}[livello]
    
    testo_alternativa = ""
    if livello == "bassa" or "debole" in motivo_g:
        alt_dc = pr["doppia_x_best"]
        p_alt_dc = pp[alt_dc]
        testo_alternativa = f"""
**💡 Qual è l'alternativa?** 
Dato che la favorita è debole, giocarla da sola è rischioso. Le alternative più solide sono:
*   **{NOMI_SCELTA[alt_dc]} al {p_alt_dc*100:.1f}%** (copre il pareggio ed è la doppia più probabile)
*   **X (Pareggio) al {pp['X']*100:.1f}%**
*   **12 (Non pareggia) al {pp['12']*100:.1f}%**
"""

    st.success(f"""**{emoji} GIOCATA CONSIGLIATA: {NOMI_SCELTA[chiave_g]}**

_{motivo_g}_

**Probabilità 1X2:** {eq_1x2} — pronostico secco del modello: **{NOMI_ESITO[top]} ({p_top*100:.1f}%)**

**Fiducia:** {livello.upper()}{" · ⚠ squadre senza storico" if senza_storico else ""}
{testo_alternativa}""")

    if st.button("🧮 Calcola Kelly per questa giocata"):
        st.session_state["kelly_p"] = pp[chiave_g] * 100
        st.session_state["kelly_q"] = round(1 / pp[chiave_g], 2)
        st.success("Dati inviati alla scheda Kelly!")

    st.markdown("#### 📈 Forma Recente (Ultime 5 partite)")
    f_casa = ultimi_5_risultati(df_storico, casa)
    f_trasf = ultimi_5_risultati(df_storico, trasf)
    def fmt_forma(lst):
        return " ".join([f"<span style='background:{'#22c55e' if x=='V' else ('#eab308' if x=='N' else '#ef4444')}; color:white; padding:2px 6px; border-radius:4px; font-weight:bold;'>{x}</span>" for x in lst])
    st.markdown(f"""
    <div style="display: flex; justify-content: space-around; text-align: center; background: #111827; padding: 15px; border-radius: 10px;">
        <div><b>{casa}</b><br>{fmt_forma(f_casa)}</div>
        <div style="color: #64748b;">VS</div>
        <div><b>{trasf}</b><br>{fmt_forma(f_trasf)}</div>
    </div>
    """, unsafe_allow_html=True)

    col_sq, col_st = st.columns([1.2, 1.8])
    with col_sq:
        st.markdown(f"""<div style="text-align: center; padding: 30px 20px; background: rgba(15,23,42,0.6); border-radius: 14px; border: 1px solid #1e293b;"><h2 style="color: {COL_CASA}; margin: 0; font-size: 1.4rem;">{casa}</h2><p style="color: #64748b; margin: 5px 0 20px 0; font-size: 0.85rem;">CASA</p><h3 style="color: {COL_ACCENTO}; margin: 0; font-size: 1.2rem;">VS</h3><h2 style="color: {COL_TRASF}; margin: 20px 0 0 0; font-size: 1.4rem;">{trasf}</h2><p style="color: #64748b; margin: 5px 0 0 0; font-size: 0.85rem;">TRASFERTA</p></div>""", unsafe_allow_html=True)
    with col_st:
        st.markdown("<p style='text-align: center; font-weight: 700; color: #94a3b8; letter-spacing: 0.5px;'>STATISTICHE ATTESE</p>", unsafe_allow_html=True)
        if is_intl:
            st.info("Statistiche dettagliate (Tiri, Corner) non disponibili per le competizioni internazionali tramite questa API.")
        else:
            righe_st = {"Gol Attesi": (round(lh, 2), round(la, 2))}; righe_st.update(stats_attese(df_storico, casa, trasf))
            for label, (vc, vt) in righe_st.items():
                st.markdown(f"""<div style="display: flex; justify-content: space-between; background: #111827; padding: 12px 18px; border-radius: 10px; border: 1px solid #1f2937; margin-bottom: 10px;"><span style="font-weight: 700; color: {COL_CASA}; width: 60px;">{vc}</span><span style="color: #94a3b8; text-align: center; flex-grow: 1;">{label}</span><span style="font-weight: 700; color: {COL_TRASF}; width: 60px; text-align: right;">{vt}</span></div>""", unsafe_allow_html=True)

    st.markdown(f"#### 📊 Mercati 1X2", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    a1.metric("1 (Casa)", f"{mk['p1']*100:.1f}%", f"Quota equa: {1/mk['p1']:.2f}", delta_color="off")
    a2.metric("X (Pareggio)", f"{mk['px']*100:.1f}%", f"Quota equa: {1/mk['px']:.2f}", delta_color="off")
    a3.metric("2 (Trasferta)", f"{mk['p2']*100:.1f}%", f"Quota equa: {1/mk['p2']:.2f}", delta_color="off")

    st.markdown(f"#### 🎰 Multigol", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Multigol sopra media (3 gol)", f"{pr['multigol_best']} gol", f"{pr['multigol_best_p']*100:.1f}%" + (f" · media {pr['multigol_best_f']*100:.1f}%" if pr.get('multigol_best_f') else ""), delta_color="off")
    
    st.markdown("#### 🎯 Heatmap Risultati Esatti Finali")
    st.plotly_chart(crea_heatmap(M), use_container_width=True)

with tabs[0]:
    if calendario.empty: st.info("Nessuna partita disponibile.")
    else:
        for idx, row in calendario.iterrows():
            if "Data" in row and pd.notna(row["Data"]):
                st.markdown(f"""<div style="background: #111827; padding: 18px; border-radius: 12px; border: 1px solid #1f2937; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;"><div><span style="font-weight: 700; font-size: 1.05rem;">{row['Casa']}</span><span style="color: #94a3b8; margin: 0 8px;">vs</span><span style="font-weight: 700; font-size: 1.05rem;">{row['Trasferta']}</span><br><span style="color: #64748b; font-size: 12px;">📅 {row['Data']} {row.get('Ora', '')}</span></div></div>""", unsafe_allow_html=True)
                if st.button("📊 Analizza Match", key=f"btn_{idx}"):
                    st.session_state["match_attivo"] = {"div": div, "casa": row["Casa"], "trasf": row["Trasferta"], "data": row["Data"], "ora": row.get("Ora", "")}
        att_m = st.session_state.get("match_attivo")
        if att_m and att_m.get("div") == div: mostra_analisi(att_m["casa"], att_m["trasf"], att_m["data"], att_m["ora"])

with tabs[1]:
    st.subheader("📋 Multipla Builder & Pronostici")
    righe_p = []
    for _, riga in calendario.iterrows():
        if "Data" not in riga or not riga["Data"]: continue
        c_, t_ = riga["Casa"], riga["Trasferta"]
        d_ = distribuzione_live(c_, t_, modello, pesi, M_freq_live, quote_fixtures)
        prp = tabelle_pronostico(d_["M_1x2"], d_["M_gol"], None, None)
        pp_ = prp["p"]; chiave_g, testo_g, motivo_g = giocata_consigliata(prp, None)
        quota_equa = round(1 / pp_[chiave_g], 2) if pp_[chiave_g] > 0 else 0
        righe_p.append({"Seleziona": False, "Data": riga["Data"], "Partita": f"{c_} - {t_}", "🎯 Giocata": NOMI_SCELTA[chiave_g], "Probabilità %": round(pp_[chiave_g]*100, 1), "Quota Equa": quota_equa})
    
    if not righe_p: st.info("Nessuna partita futura.")
    else: 
        df_p = pd.DataFrame(righe_p)
        edited_df = st.data_editor(df_p, column_config={"Seleziona": st.column_config.CheckboxColumn("Aggiungi", default=False), "Probabilità %": st.column_config.NumberColumn(format="%.1f%%")}, disabled=["Data", "Partita", "🎯 Giocata", "Probabilità %", "Quota Equa"], hide_index=True, use_container_width=True)
        selected = edited_df[edited_df["Seleziona"] == True]
        if not selected.empty:
            st.markdown("---")
            prob_comb = np.prod(selected["Probabilità %"] / 100); quota_tot = np.prod(selected["Quota Equa"]); ev = (prob_comb * quota_tot - 1) * 100
            m1, m2, m3 = st.columns(3)
            m1.metric("Quota Totale", f"{quota_tot:.2f}"); m2.metric("Probabilità Combinata", f"{prob_comb*100:.2f}%"); m3.metric("Valore Atteso (EV)", f"{ev:+.1f}%")

with tabs[2]:
    st.subheader("🏆 Classifica")
    df_cls = tabella_classifica(df_storico[df_storico["Stag"] == df_storico["Stag"].max()])
    if not df_cls.empty: st.dataframe(df_cls, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("⚡ Scanner Value Finder")
    if not quote_fixtures: st.info("Nessuna quota disponibile tramite The Odds API.")
    else:
        edge_min = st.slider("Edge Minimo (%)", 0.0, 25.0, 3.0)
        righe_v = []
        for (casa, trasf), q in quote_fixtures.items():
            lh, la = lambda_partita(modello, casa, trasf)
            mk = mercati(matrice_punteggi(lh, la, modello["rho"]))
            P = np.array([mk["p1"], mk["px"], mk["p2"]]); pm = senza_margine(q)
            nomi = [f"1 ({casa})", "X (Pareggio)", f"2 ({trasf})"]
            for k in range(3):
                edge = (P[k] * q[k] - 1) * 100
                if edge >= edge_min:
                    righe_v.append({"Partita": f"{casa} vs {trasf}", "Selezione": nomi[k], "Prob_Modello": round(P[k] * 100, 1), "Prob_Mercato": round(pm[k] * 100, 1), "Quota_Book": round(q[k], 2), "Edge": round(edge, 1)})
        if righe_v: st.dataframe(pd.DataFrame(righe_v).sort_values("Edge", ascending=False), use_container_width=True, hide_index=True)
        else: st.info("Nessuna selezione con questi filtri.")

with tabs[4]:
    st.subheader("🧮 Calcolatore Kelly")
    c1, c2, c3 = st.columns(3)
    p_default = st.session_state.get("kelly_p", 45.0); q_default = st.session_state.get("kelly_q", 2.30)
    p_stim = c1.slider("Probabilità Modello (%)", 1.0, 100.0, float(p_default))
    q_book = c2.number_input("Quota Bookmaker", 1.01, 50.0, float(q_default))
    bankroll = c3.number_input("Bankroll (€)", 10.0, 100000.0, 1000.0)
    b, p = q_book - 1, p_stim / 100; kelly_fraction = max(0.0, (b * p - (1 - p)) / b) * 0.25
    st.metric(f"Stake (Kelly 25%)", f"€{bankroll*kelly_fraction:.2f}")
