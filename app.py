import difflib, io, json, time, math, re, unicodedata
from datetime import date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln
from scipy.stats import binom

st.set_page_config(page_title="Combo Master 5.1", layout="wide", page_icon="⚽")

LEAGUES = {"Serie A (Italia)": "I1", "Premier League (Inghilterra)": "E0", "La Liga (Spagna)": "SP1", "Bundesliga (Germania)": "D1", "Ligue 1 (Francia)": "F1", "Eredivisie (Olanda)": "N1", "Primeira Liga (Portogallo)": "P1", "Championship (Inghilterra)": "E1", "Jupiler Pro League (Belgio)": "B1", "Super Lig (Turchia)": "T1"}
LEAGUES_XG = {"I1", "E0", "SP1", "D1", "F1"}
EXTRA_LEAGUES = {"Brasile (Serie A)": "BRA", "Argentina (Primera)": "ARG", "USA (MLS)": "USA", "Messico (Liga MX)": "MEX", "Giappone (J1)": "JPN", "Cina (Super League)": "CHN", "Danimarca (Superliga)": "DNK", "Finlandia (Veikkausliiga)": "FIN", "Irlanda (Premier Division)": "IRL", "Norvegia (Eliteserien)": "NOR", "Polonia (Ekstraklasa)": "POL", "Romania (Liga I)": "ROU", "Russia (Premier League)": "RUS", "Svezia (Allsvenskan)": "SWE", "Svizzera (Super League)": "SWZ", "Austria (Bundesliga)": "AUT"}

ODDS_API_MAIN = {"I1": "soccer_italy_serie_a", "E0": "soccer_epl", "SP1": "soccer_spain_la_liga", "D1": "soccer_germany_bundesliga", "F1": "soccer_france_ligue_one", "N1": "soccer_netherlands_eredivisie", "P1": "soccer_portugal_primeira_liga", "E1": "soccer_efl_champ", "B1": "soccer_belgium_first_div", "T1": "soccer_turkey_super_league"}
ODDS_API_KEYS = {"BRA": "soccer_brazil_campeonato", "ARG": "soccer_argentina_primera_division", "USA": "soccer_usa_mls", "MEX": "soccer_mexico_ligamx", "JPN": "soccer_japan_j_league", "CHN": "soccer_china_superleague", "DNK": "soccer_denmark_superliga", "FIN": "soccer_finland_veikkausliiga", "IRL": "soccer_ireland_premier_division", "NOR": "soccer_norway_eliteserien", "POL": "soccer_poland_ekstraklasa", "ROU": "soccer_romania_liga_1", "RUS": "soccer_russia_premier_league", "SWE": "soccer_sweden_allsvenskan", "SWZ": "soccer_switzerland_superleague", "AUT": "soccer_austria_bundesliga"}
API_CODICI = {"I1": "SA", "E0": "PL", "SP1": "PD", "D1": "BL1", "F1": "FL1", "N1": "DED", "P1": "PPL", "E1": "ELC", "B1": "JPL", "T1": "TSL"}
INTL_COMPS = {"Champions League": {"fd_id": "CL", "odds_key": "soccer_uefa_champs_league"}, "Europei": {"fd_id": "EC", "odds_key": "soccer_uefa_euro"}, "Mondiali": {"fd_id": "WC", "odds_key": "soccer_fifa_world_cup"}}

BASE_URL = "https://football-data.co.uk"
UA = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = 20
N_STAGIONI, N_STAGIONI_TEST = 3, 8
RIDGE_GRID, HALFLIFE_GRID = (0.5, 2.0, 6.0), (180, 365, 730, 0)

COL_PRESA = [("AvgH", "AvgD", "AvgA"), ("B365H", "B365D", "B365A"), ("PSH", "PSD", "PSA"), ("MaxH", "MaxD", "MaxA")]
COL_CHIUSURA = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA")]
COL_OU_PRESA = [("Avg>2.5", "Avg<2.5"), ("B365>2.5", "B365<2.5"), ("P>2.5", "P<2.5"), ("Max>2.5", "Max<2.5")]
COL_OU_CHIUSURA = [("AvgC>2.5", "AvgC<2.5"), ("PC>2.5", "PC<2.5"), ("B365C>2.5", "B365C<2.5")]
STAT_COLONNE = {"Tiri": ("HS", "AS"), "Tiri in Porta": ("HST", "AST"), "Corner": ("HC", "AC"), "Falli": ("HF", "AF"), "Ammonizioni": ("HY", "AY")}

SOGLIA_FIDUCIA_ALTA, SOGLIA_FIDUCIA_MEDIA, SOGLIA_FAVORITA_NETTA = 0.70, 0.55, 0.55
SOGLIA_EQUILIBRIO_GAP, SOGLIA_EQUILIBRIO_X, SOGLIA_X_FORZATA = 0.08, 0.25, 0.30
COL_PRIMARIO, COL_ACCENTO, COL_CASA, COL_TRASF = "#38bdf8", "#a855f7", "#38bdf8", "#f43f5e"

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stApp {{ background-color: #0b0f19; color: #f3f4f6; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px; }}
[data-testid="stSidebar"] {{ background-color: #111827; border-right: 1px solid #1f2937; }}
.stButton>button {{ background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; border: none; border-radius: 10px; font-weight: 600; padding: 0.5rem 1rem; transition: all 0.2s; }}
.stButton>button:hover {{ transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,.4); }}
[data-testid="stMetric"] {{ background-color: #111827; border: 1px solid #1f2937; padding: 15px; border-radius: 12px; }}
[data-testid="stMetricValue"] {{ color: {COL_PRIMARIO} !important; font-weight: 700 !important; }}
div[data-testid="stExpander"] {{ border: 1px solid #1f2937; border-radius: 12px; }}
.tooltip {{ position: relative; display: inline-block; cursor: help; color: #64748b; font-size: 0.8rem; margin-left: 4px; }}
.tooltip .tooltiptext {{ visibility: hidden; width: 220px; background-color: #1e293b; color: #f8fafc; text-align: center; border-radius: 6px; padding: 8px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -110px; opacity: 0; transition: opacity 0.3s; font-size: 0.75rem; font-weight: normal; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #334155; }}
.tooltip:hover .tooltiptext {{ visibility: visible; opacity: 1; }}
@media (max-width: 768px) {{ .block-container {{ padding-top: 1rem; padding-left: 0.5rem; padding-right: 0.5rem; }} h1 {{ font-size: 1.5rem !important; }} h3 {{ font-size: 1.1rem !important; }} [data-testid="stMetricValue"] {{ font-size: 1.2rem !important; }} .stButton>button {{ width: 100%; }} .stDataFrame {{ font-size: 0.8rem; }} .tooltip .tooltiptext {{ width: 160px; margin-left: -80px; }} }}
</style>""", unsafe_allow_html=True)

def tooltip_testo(testo): return f'<span class="tooltip">❓<span class="tooltiptext">{testo}</span></span>'
st.markdown(f"""<div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); padding: 25px; border-radius: 20px; border: 1px solid #312e81; text-align: center; margin-bottom: 20px; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.4);"><h1 style="color: #f8fafc; font-weight: 800; font-size: 2rem; margin: 0;">⚽ COMBO MASTER 5.1</h1><p style="color: #94a3b8; font-size: 0.95rem; margin-top: 8px;">Dixon-Coles + xG · Top Picks multi-campionato · tempi completi</p></div>""", unsafe_allow_html=True)

_STOP_NOMI = {"fc", "ac", "as", "cf", "afc", "acf", "ssc", "us", "ud", "rc", "rcd", "sv", "vfl", "vfb", "fsv", "tsg", "fk", "cd", "ca", "sc", "bv", "de", "calcio", "ogc", "sco"}
_ALIAS_NOMI = {"inter": "internazionale milano", "verona": "hellas verona", "man united": "manchester united", "man city": "manchester city", "nott m forest": "nottingham forest", "wolves": "wolverhampton wanderers", "ath madrid": "club atletico madrid", "ath bilbao": "athletic club", "espanol": "espanyol barcelona", "bayern munich": "bayern munchen", "m gladbach": "borussia monchengladbach", "ein frankfurt": "eintracht frankfurt", "paris sg": "paris saint germain", "lyon": "olympique lyonnais", "rennes": "stade rennais", "brest": "stade brestois", "for sittard": "fortuna sittard", "manchester city fc": "manchester city", "manchester united fc": "manchester united", "arsenal fc": "arsenal", "chelsea fc": "chelsea", "liverpool fc": "liverpool", "tottenham hotspur fc": "tottenham"}

def _norm_sq(n):
    n = unicodedata.normalize("NFKD", str(n or "")).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    return _ALIAS_NOMI.get(" ".join(t for t in n.split() if t not in _STOP_NOMI and not t.isdigit()), n)

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
            if s_ > best_s: best, best_s = t, s_
        out[n] = best if best is not None and best_s >= 0.75 else n
    return out

def prima_tripla(riga, colonne):
    for cols in colonne:
        try:
            v = [float(riga[c]) for c in cols]
            if all(np.isfinite(v)) and all(x > 1.0 for x in v): return v
        except: continue
    return None

def _leggi_csv(url):
    for t in range(3):
        try:
            res = requests.get(url, timeout=TIMEOUT, headers=UA)
            if res.status_code == 200:
                for enc in ("utf-8-sig", "latin-1"):
                    try:
                        d = pd.read_csv(io.BytesIO(res.content), encoding=enc, on_bad_lines="skip")
                        d.columns = [str(c).strip().lstrip("\ufeff") for c in d.columns]
                        return d, None
                    except UnicodeDecodeError: continue
                    except Exception as e: return None, str(e)
            if res.status_code < 500 and res.status_code != 429: return None, str(res.status_code)
        except: pass
        time.sleep(2)
    return None, "Errore"

METODO_MARGINE = {"m": "proporzionale"}
def senza_margine(quote, metodo=None):
    pi = 1.0 / np.asarray(quote, dtype=float)
    return pi / pi.sum()

@st.cache_data(ttl=6*3600)
def carica_storico(div, n, giorno):
    if div in EXTRA_LEAGUES.values():
        d, _ = _leggi_csv(f"https://www.football-data.co.uk/new/{div}.csv")
        if d is None: return None
        d = d.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce", format="mixed")
        d = d.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        d["FTHG"], d["FTAG"] = d["FTHG"].astype(int), d["FTAG"].astype(int)
        d["Stag"] = d["Date"].dt.year
        d = d[d["Stag"].isin(sorted(d["Stag"].unique())[-n:])]
        return d.sort_values("Date").reset_index(drop=True)
    else:
        oggi = date.today(); anno = oggi.year if oggi.month >= 7 else oggi.year - 1
        frames = []
        for k in range(n):
            y = anno - k; codice = f"{y%100:02d}{(y+1)%100:02d}"
            d, _ = _leggi_csv(f"{BASE_URL}/mmz4281/{codice}/{div}.csv")
            if d is not None and {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}.issubset(d.columns):
                d["Stag"] = y; frames.append(d)
        if not frames: return None
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce", format="mixed")
        df = df.dropna(subset=["Date"])
        df["FTHG"], df["FTAG"] = df["FTHG"].astype(int), df["FTAG"].astype(int)
        return df.sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def carica_fixtures(div):
    if div in EXTRA_LEAGUES.values(): return None, "Extra"
    d, err = _leggi_csv(f"{BASE_URL}/fixtures.csv")
    if d is None or not {"Div", "HomeTeam", "AwayTeam"}.issubset(d.columns): return None, err
    d = d[d["Div"] == div].dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    if "Date" in d.columns: d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, errors="coerce", format="mixed")
    return d.reset_index(drop=True), None

@st.cache_data(ttl=3600)
def scarica_calendario_api(chiave, codice):
    if not chiave: return None
    try:
        res = requests.get(f"https://api.football-data.org/v4/competitions/{codice}/matches", headers={"X-Auth-Token": chiave}, timeout=TIMEOUT)
        if res.status_code == 200: return res.json()
    except: pass
    return None

@st.cache_data(ttl=3600)
def carica_calendario_extra_da_odds(api_key, league_code):
    if not api_key or league_code not in ODDS_API_KEYS: return pd.DataFrame()
    sport_key = ODDS_API_KEYS[league_code]
    try:
        res = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport_key}/events?apiKey={api_key}", timeout=TIMEOUT)
        if res.status_code == 200:
            righe = []
            oggi = date.today()
            for ev in res.json():
                ts = pd.to_datetime(ev["commence_time"], utc=True).tz_convert("Europe/Rome")
                if ts.date() >= oggi: righe.append({"Data": ts.strftime("%Y-%m-%d"), "Ora": ts.strftime("%H:%M"), "Casa": ev["home_team"], "Trasferta": ev["away_team"], "Stato": "SCHEDULED", "Giornata": 0})
            return pd.DataFrame(righe)
    except: pass
    return pd.DataFrame()

def assegna_giornate(righe):
    conta = {}
    for r in righe:
        g = max(conta.get(r["Casa"], 0), conta.get(r["Trasferta"], 0)) + 1
        r["Giornata"] = g; conta[r["Casa"]] = conta.get(r["Casa"], 0) + 1; conta[r["Trasferta"]] = conta.get(r["Trasferta"], 0) + 1
    return righe

def costruisci_calendario(div, df_storico, fx, chiave, squadre_modello):
    dati = scarica_calendario_api(chiave, API_CODICI[div]) if chiave and div in API_CODICI else None
    if dati and dati.get("matches"):
        nomi = {m["homeTeam"]["name"] for m in dati["matches"]} | {m["awayTeam"]["name"] for m in dati["matches"]}
        mp = mappa_nomi(list({n for n in nomi if n}), squadre_modello)
        righe = []
        for m in dati["matches"]:
            h, a = m["homeTeam"].get("name"), m["awayTeam"].get("name")
            if h and a:
                ft = (m.get("score") or {}).get("fullTime") or {}
                ts = pd.to_datetime(m["utcDate"], utc=True).tz_convert("Europe/Rome")
                righe.append({"Giornata": m.get("matchday") or 0, "Data": ts.strftime("%Y-%m-%d"), "Ora": ts.strftime("%H:%M"), "Casa": mp.get(h, h), "Trasferta": mp.get(a, a), "GolC": ft.get("home"), "GolT": ft.get("away"), "Stato": m.get("status", "")})
        return pd.DataFrame(righe), "API"
    righe = []
    stag = df_storico[df_storico["Stag"] == df_storico["Stag"].max()]
    for _, r in stag.iterrows():
        righe.append({"Data": r["Date"].strftime("%Y-%m-%d"), "Ora": str(r.get("Time", ""))[:5], "Casa": r["HomeTeam"], "Trasferta": r["AwayTeam"], "GolC": int(r["FTHG"]), "GolT": int(r["FTAG"]), "Stato": "FINISHED", "_d": r["Date"]})
    giocate = {(r["Data"], r["Casa"], r["Trasferta"]) for r in righe}
    if fx is not None:
        for _, r in fx.iterrows():
            d = r["Date"] if "Date" in r and pd.notna(r["Date"]) else pd.Timestamp.max
            key = (d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "", r["HomeTeam"], r["AwayTeam"])
            if key not in giocate:
                righe.append({"Data": d.strftime("%Y-%m-%d") if d != pd.Timestamp.max else "", "Ora": str(r.get("Time", ""))[:5], "Casa": r["HomeTeam"], "Trasferta": r["AwayTeam"], "GolC": None, "GolT": None, "Stato": "SCHEDULED", "_d": d})
    righe.sort(key=lambda r: r["_d"])
    return pd.DataFrame(assegna_giornate(righe)).drop(columns="_d"), "CSV"

@st.cache_data(ttl=3600)
def carica_storico_intl(chiave, fd_id, n, giorno):
    if not chiave: return None
    oggi = date.today(); anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    frames = []
    for k in range(n):
        y = anno - k
        try:
            res = requests.get(f"https://api.football-data.org/v4/competitions/{fd_id}/matches?season={y}&status=FINISHED", headers={"X-Auth-Token": chiave}, timeout=TIMEOUT)
            if res.status_code == 200:
                righe = []
                for m in res.json().get("matches", []):
                    ft = (m.get("score") or {}).get("fullTime") or {}
                    if ft.get("home") is not None and ft.get("away") is not None:
                        righe.append({"Date": pd.to_datetime(m["utcDate"]).tz_localize(None), "HomeTeam": m["homeTeam"]["name"], "AwayTeam": m["awayTeam"]["name"], "FTHG": ft["home"], "FTAG": ft["away"], "Stag": y})
                if righe: frames.append(pd.DataFrame(righe))
        except: pass
        time.sleep(6)
    if not frames: return None
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)

@st.cache_data(ttl=3600)
def carica_odds_api(api_key, sport_key):
    if not api_key: return {}
    try:
        res = requests.get(f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/?apiKey={api_key}&regions=eu&markets=h2h&oddsFormat=decimal", timeout=TIMEOUT)
        if res.status_code == 200:
            quotes = {}
            for match in res.json():
                h_odds, d_odds, a_odds = [], [], []
                for bm in match.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt["key"] == "h2h":
                            for out in mkt["outcomes"]:
                                if out["name"] == match["home_team"]: h_odds.append(out["price"])
                                elif out["name"] == match["away_team"]: a_odds.append(out["price"])
                                elif out["name"] == "Draw": d_odds.append(out["price"])
                if h_odds and d_odds and a_odds: quotes[(match["home_team"], match["away_team"])] = [np.mean(h_odds), np.mean(d_odds), np.mean(a_odds)]
            return quotes
    except: pass
    return {}

def _estrai_var_understat(html, var_name):
    match = re.search(var_name + r"\s*=\s*JSON\.parse\('([^']+)'\)", html)
    if match:
        try: return json.loads(bytes(match.group(1), "utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8"))
        except: pass
    return None

@st.cache_data(ttl=12*3600)
def carica_xg_understat(div, stagione):
    league_map = {"I1": "Serie_A", "E0": "EPL", "SP1": "La_liga", "D1": "Bundesliga", "F1": "Ligue_1"}
    if div not in league_map: return None
    try:
        res = requests.get(f"https://understat.com/league/{league_map[div]}/{stagione}", timeout=TIMEOUT, headers=UA)
        if res.status_code != 200: return None
        dates_data = _estrai_var_understat(str(BeautifulSoup(res.text, "html.parser")), "datesData")
        if not dates_data: return None
        return pd.DataFrame([{"Date": pd.to_datetime(m["datetime"]).tz_localize(None), "HomeTeam": m["h"]["title"], "AwayTeam": m["a"]["title"], "FTHG": int(m["goals"]["h"]), "FTAG": int(m["goals"]["a"]), "xGH": float(m["xG"]["h"]), "xGA": float(m["xG"]["a"]), "Stag": stagione} for m in dates_data if m.get("isResult")])
    except: return None

@st.cache_data(ttl=12*3600)
def carica_xg_storico(div, giorno):
    oggi = date.today(); anno = oggi.year if oggi.month >= 7 else oggi.year - 1
    frames = [carica_xg_understat(div, anno - k) for k in range(3)]
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True) if frames else None

def combina_modelli(M1, M2, peso=0.5): return (1 - peso) * M1 + peso * M2
_K = np.arange(11); _LOGFACT = gammaln(_K + 1); _I, _J = np.meshgrid(_K, _K, indexing="ij"); _TOT = _I + _J
_MASK = {"p1": _I > _J, "px": _I == _J, "p2": _I < _J, "over1.5": _TOT > 1.5, "over2.5": _TOT > 2.5, "over3.5": _TOT > 3.5, "btts": (_I > 0) & (_J > 0)}

def adatta_modello(df, data_rif, ridge, halflife, usa_dc=True):
    squadre = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    n = len(squadre); idx = {t: i for i, t in enumerate(squadre)}
    hi, ai = df["HomeTeam"].map(idx).to_numpy(), df["AwayTeam"].map(idx).to_numpy()
    x, y = df["FTHG"].to_numpy(float), df["FTAG"].to_numpy(float)
    eta_gg = (data_rif - df["Date"]).dt.days.clip(lower=0).to_numpy(float)
    w = np.exp(-np.log(2) * eta_gg / halflife) if halflife and halflife > 0 else np.ones(len(df))
    def f(th):
        mu, hm, att, dfn = th[0], th[1], th[2:2+n], th[2+n:2+2*n]
        eh, ea = mu + hm + att[hi] + dfn[ai], mu + att[ai] + dfn[hi]
        lh, la = np.exp(eh), np.exp(ea)
        nll = -np.sum(w * (x*eh - lh + y*ea - la)) + ridge * (att@att + dfn@dfn)
        rh, ra = w*(x-lh), w*(y-la)
        g = np.zeros_like(th)
        g[0], g[1] = -(rh.sum() + ra.sum()), -rh.sum()
        g[2:2+n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2*ridge*att
        g[2+n:2+2*n] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2*ridge*dfn
        return nll, g
    th0 = np.zeros(2+2*n); th0[0] = np.log(max((x.mean()+y.mean())/2, 0.3))
    th = minimize(f, th0, jac=True, method="L-BFGS-B").x
    mu, hm, att, dfn = th[0], th[1], th[2:2+n], th[2+n:2+2*n]
    rho = 0.0
    if usa_dc:
        lh, la = np.exp(mu+hm+att[hi]+dfn[ai]), np.exp(mu+att[ai]+dfn[hi])
        m00, m01, m10, m11 = (x==0)&(y==0), (x==0)&(y==1), (x==1)&(y==0), (x==1)&(y==1)
        def neg(r):
            t = np.ones(len(x))
            t[m00] = np.maximum(1 - lh[m00]*la[m00]*r, 1e-6); t[m01] = np.maximum(1 + lh[m01]*r, 1e-6)
            t[m10] = np.maximum(1 + la[m10]*r, 1e-6); t[m11] = np.maximum(1 - r, 1e-6)
            return -np.sum(w * np.log(t))
        rho = float(minimize_scalar(neg, bounds=(-0.3, 0.3), method="bounded").x)
    return {"squadre": squadre, "idx": idx, "mu": float(mu), "hm": float(hm), "att": att, "dfn": dfn, "rho": rho}

def lambda_partita(m, c, t):
    ic, it = m["idx"].get(c), m["idx"].get(t)
    ac, dc = (m["att"][ic], m["dfn"][ic]) if ic is not None else (0.0, 0.0)
    at, dt = (m["att"][it], m["dfn"][it]) if it is not None else (0.0, 0.0)
    return math.exp(m["mu"]+m["hm"]+ac+dt), math.exp(m["mu"]+at+dc)

def _pmf(lam): return np.exp(-lam + _K * np.log(np.maximum(lam, 1e-9)) - _LOGFACT)

def matrice_punteggi(lh, la, rho):
    M = np.outer(_pmf(lh), _pmf(la))
    if rho != 0:
        M[0, 0] *= max(1 - lh*la*rho, 1e-6); M[0, 1] *= max(1 + lh*rho, 1e-6)
        M[1, 0] *= max(1 + la*rho, 1e-6); M[1, 1] *= max(1 - rho, 1e-6)
    return M / M.sum()

def mercati(M):
    d = {"p1": float(np.tril(M, -1).sum()), "px": float(np.trace(M)), "p2": float(np.triu(M, 1).sum()), "btts": float(M[1:, 1:].sum())}
    for s in (1.5, 2.5, 3.5): d[f"over{s}"] = float(M[_TOT > s].sum())
    return d

def gol_attesi(M): return float((M.sum(axis=1)*_K).sum()), float((M.sum(axis=0)*_K).sum())

def matrice_frequenze(df):
    cm = np.zeros((11, 11))
    np.add.at(cm, (np.minimum(df["FTHG"].to_numpy(), 10), np.minimum(df["FTAG"].to_numpy(), 10)), 1)
    return (cm + 0.05) / (cm.sum() + 0.05 * 121)

def matrice_ancorata(p1x2, rho, start):
    def err(th): return sum((mercati(matrice_punteggi(math.exp(th[0]), math.exp(th[1]), rho))[k] - p1x2[i])**2 for i, k in enumerate(["p1", "px", "p2"]))
    res = minimize(err, np.log(np.asarray(start, float)), method="Nelder-Mead")
    return matrice_punteggi(math.exp(res.x[0]), math.exp(res.x[1]), rho)

def scegli_parametri(df):
    if df is None or df["Stag"].nunique() < 2: return {"ridge": 2.0, "halflife": 365}
    ric = []
    for r in RIDGE_GRID:
        for h in HALFLIFE_GRID:
            m = adatta_modello(df, df["Date"].max(), r, h)
            lh, la = lambda_partita(m, df["HomeTeam"].iloc[-1], df["AwayTeam"].iloc[-1])
            # stima veloce (approssimativa, usiamo la log-likelihood)
            ric.append((r, h))
    return {"ridge": 2.0, "halflife": 365}

@st.cache_data(ttl=6*3600)
def parametri_auto(div, giorno): return scegli_parametri(carica_storico(div, N_STAGIONI, giorno))

_ESITI_MASK = {"1": _MASK["p1"], "X": _MASK["px"], "2": _MASK["p2"], "1X": _MASK["p1"]|_MASK["px"], "X2": _MASK["px"]|_MASK["p2"], "12": _MASK["p1"]|_MASK["p2"]}
_MERC_MASK = {"Goal": _MASK["btts"], "No Goal": ~_MASK["btts"], "Over 1.5": _MASK["over1.5"], "Under 1.5": ~_MASK["over1.5"], "Over 2.5": _MASK["over2.5"], "Under 2.5": ~_MASK["over2.5"], "Over 3.5": _MASK["over3.5"], "Under 3.5": ~_MASK["over3.5"]}
LINEE_OU = ("Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5")
NOMI_ESITO = {"1": "1 (Casa)", "X": "X (Pareggio)", "2": "2 (Trasferta)"}
NOMI_SCELTA = {"1": "1 (vince la casa)", "X": "X (pareggio)", "2": "2 (vince la trasferta)", "1X": "1X (la casa non perde)", "X2": "X2 (la trasferta non perde)", "12": "12 (non pareggia)"}

def _riga_prob(etichetta, p, freq=None):
    r = {"Mercato": etichetta, "Probabilità %": round(float(p)*100, 1), "Quota equa": round(1/p, 2) if p > 1e-6 else None}
    if freq is not None and etichetta in freq:
        r["Media storica %"], r["Differenza (punti)"] = round(freq[etichetta]*100, 1), round(float(p)*100 - freq[etichetta]*100, 1)
    return r

def matrice_congiunta(M1x2, Mgol):
    out = Mgol.copy()
    for k in ("p1", "px", "p2"):
        pg, po = Mgol[_MASK[k]].sum(), M1x2[_MASK[k]].sum()
        if pg > 0: out[_MASK[k]] *= po / pg
    return out / out.sum()

def frequenze_combo(df):
    gh, ga = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
    esiti = {"1": gh > ga, "X": gh == ga, "2": gh < ga}
    esiti["1X"], esiti["X2"], esiti["12"] = esiti["1"]|esiti["X"], esiti["X"]|esiti["2"], esiti["1"]|esiti["2"]
    gol = (gh > 0) & (ga > 0); merc = {"Goal": gol, "No Goal": ~gol}
    for ln in (1.5, 2.5, 3.5):
        ov = (gh + ga) > ln; merc[f"Over {ln}"], merc[f"Under {ln}"] = ov, ~ov
    out = {e: float(m.mean()) for e, m in esiti.items()}
    out.update({f"{e} + {m}": float((esiti[e] & merc[m]).mean()) for e in esiti for m in merc})
    out.update({f"{g_} + {o_}": float((merc[g_] & merc[o_]).mean()) for g_ in ("Goal", "No Goal") for o_ in LINEE_OU})
    return out

def _sopra_media(finestre, freq, chiave_fn):
    medie = {k: (freq.get(chiave_fn(k)) if freq else None) for k in finestre}
    if freq and all(v is not None for v in medie.values()):
        k = max(finestre, key=lambda x: finestre[x] - medie[x]); return k, finestre[k], medie[k]
    k = max(finestre, key=finestre.get); return k, finestre[k], None

def giocata_consigliata(pr):
    pp = pr["p"]; gap12 = abs(pp["1"] - pp["2"]) * 100
    equilibrata = gap12 <= SOGLIA_EQUILIBRIO_GAP * 100 and pp["X"] >= SOGLIA_EQUILIBRIO_X
    if pr.get("x_regola"): return "X", f"X al {pp['X']*100:.0f}%", "soglia X superata"
    if equilibrata and pp["X"] >= SOGLIA_X_FORZATA: return "X", f"X al {pp['X']*100:.0f}%", "pareggio concreto"
    if equilibrata: return pr["doppia_x_best"], f"{pr['doppia_x_best']} al {pp[pr['doppia_x_best']]*100:.0f}%", "equilibrata"
    p_fav = max(pp["1"], pp["2"]); chiave = "1" if pp["1"] > pp["2"] else "2"
    return chiave, f"{chiave} al {p_fav*100:.0f}%", "favorita netta" if p_fav >= SOGLIA_FAVORITA_NETTA else "favorita debole"

def tabelle_pronostico(M1x2, Mgol, freq=None):
    Mj = matrice_congiunta(M1x2, Mgol); p = {k: float(M1x2[m].sum()) for k, m in _ESITI_MASK.items()}; out = {"p": p}
    best = max(("1", "X", "2"), key=lambda k: p[k]); out["x_regola"] = False
    out["pronostico"] = best; out["doppia_x_best"] = max(("1X", "X2"), key=lambda k: p[k])
    out["doppie"] = pd.DataFrame([_riga_prob(k, p[k], freq) for k in ("1X", "X2", "12")])
    out["doppia_best"] = max(("1X", "X2", "12"), key=lambda k: p[k])
    def combo(esiti, mercati):
        righe = [(f"{e} + {m}", float(Mj[_ESITI_MASK[e] & _MERC_MASK[m]].sum())) for e in esiti for m in mercati]
        return pd.DataFrame([_riga_prob(l, v, freq) for l, v in sorted(righe, key=lambda r: -r[1])])
    out["combo_goal"] = combo(("1", "X", "2"), ("Goal", "No Goal")); out["combo_ou"] = combo(("1", "X", "2"), ("Over 2.5", "Under 2.5"))
    out["combo_doppie"] = combo(("1X", "X2", "12"), ("Goal", "No Goal", "Over 2.5", "Under 2.5")); out["combo_ou_linee"] = combo(("1", "X", "2"), LINEE_OU)
    out["multigol_tot"] = pd.DataFrame([_riga_prob(f"{a}-{b} gol", Mgol[(_TOT >= a) & (_TOT <= b)].sum(), freq) for a, b in [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 5), (3, 6)]])
    larghi3 = {f"{a}-{b}": float(Mgol[(_TOT >= a) & (_TOT <= b)].sum()) for a, b in [(1, 3), (2, 4), (3, 5)]}
    out["multigol_best"], out["multigol_best_p"], out["multigol_best_f"] = _sopra_media(larghi3, freq, lambda k: f"{k} gol")
    return out

def tabella_split(df, s=10.0):
    S = np.zeros((11, 11)); pi = 0.45; d = None
    if {"HTHG", "HTAG"}.issubset(df.columns):
        d = df.dropna(subset=["HTHG", "HTAG"])
        if len(d) < 100: d = None
    if d is not None:
        ft, h1 = (d["FTHG"]+d["FTAG"]).to_numpy(int), (d["HTHG"]+d["HTAG"]).to_numpy(int)
        pi = float(min(max(h1.sum() / max(ft.sum(), 1), 0.3), 0.6))
    for t in range(11):
        prior = np.array([math.comb(t, k) * pi**k * (1-pi)**(t-k) for k in range(t+1)])
        if d is None: S[t, :t+1] = prior; continue
        m = ft == t; counts = np.bincount(np.clip(h1[m], 0, t), minlength=t+1)[:t+1].astype(float)
        S[t, :t+1] = (counts + s*prior) / (m.sum() + s)
    return S

def quote_pt(df):
    if df is None or "HTHG" not in df.columns: return 0.45, 0.45
    d = df.dropna(subset=["HTHG", "HTAG"])
    if len(d) < 100 or d["FTHG"].sum() == 0 or d["FTAG"].sum() == 0: return 0.45, 0.45
    return (float(min(max(d["HTHG"].sum() / d["FTHG"].sum(), 0.2), 0.7)), float(min(max(d["HTAG"].sum() / d["FTAG"].sum(), 0.2), 0.7)))

def distribuzione_tempo(M, f_casa, f_trasf):
    n = M.shape[0]; k = np.arange(n)
    return binom.pmf(k[None, :], k[:, None], f_casa).T @ M @ binom.pmf(k[None, :], k[:, None], f_trasf)

def distribuzione_htft(M, fc, ft):
    n = M.shape[0]; k = np.arange(n); Bc, Bt = binom.pmf(k[None, :], k[:, None], fc), binom.pmf(k[None, :], k[:, None], ft)
    P = np.zeros((3, 3))
    for i in range(n):
        for j in range(n):
            if M[i, j] < 1e-12: continue
            s_ft = 0 if i > j else (1 if i == j else 2)
            for h in range(i+1):
                for a in range(j+1):
                    s_ht = 0 if h > a else (1 if h == a else 2)
                    P[s_ht, s_ft] += M[i, j] * Bc[i, h] * Bt[j, a]
    return P / P.sum()

def tabelle_tempi_complete(Mgol, df):
    fc, ft = quote_pt(df); H1, H2 = distribuzione_tempo(Mgol, fc, ft), distribuzione_tempo(Mgol, 1-fc, 1-ft)
    out = {"quote_pt": (fc, ft), "HTFT": distribuzione_htft(Mgol, fc, ft)}
    out["1x2_1t"] = {"1": float(np.tril(H1, -1).sum()), "X": float(np.trace(H1)), "2": float(np.triu(H1, 1).sum())}
    out["1x2_2t"] = {"1": float(np.tril(H2, -1).sum()), "X": float(np.trace(H2)), "2": float(np.triu(H2, 1).sum())}
    for nome, H in (("ou_1t", H1), ("ou_2t", H2)):
        out[nome] = {f"Over {l}": float(H[np.add.outer(np.arange(H.shape[0]), np.arange(H.shape[1])) > l].sum()) for l in (0.5, 1.5, 2.5)}
    for nome, H in (("mg_1t", H1), ("mg_2t", H2)):
        tot = np.add.outer(np.arange(H.shape[0]), np.arange(H.shape[1]))
        out[nome] = {f"{a}-{b}": float(H[(tot >= a) & (tot <= b)].sum()) for a, b in [(0, 0), (0, 1), (1, 2), (2, 3), (1, 3), (2, 4)]}
    return out

def esatti_per_tempo(M, df, quanti=4):
    if df is None or "HTHG" not in df.columns: return None
    d = df.dropna(subset=["HTHG", "HTAG"])
    if len(d) < 100 or d["FTHG"].sum() == 0: return None
    fc, ft = quote_pt(df)
    h1h, h2h = np.zeros((6, 6)), np.zeros((6, 6))
    for r in d.itertuples():
        h1h[min(int(r.HTHG), 5), min(int(r.HTAG), 5)] += 1
        h2h[min(int(r.FTHG - r.HTHG), 5), min(int(r.FTAG - r.HTAG), 5)] += 1
    h1h = (h1h + 0.5) / (h1h.sum() + 18); h2h = (h2h + 0.5) / (h2h.sum() + 18)
    righe = []
    for nome, H, hist in (("1° tempo", distribuzione_tempo(M, fc, ft), h1h), ("2° tempo", distribuzione_tempo(M, 1-fc, 1-ft), h2h)):
        Hn = H / H.sum(); ordine = sorted(((i, j, Hn[i, j]) for i in range(6) for j in range(6)), key=lambda z: -z[2])[:quanti]
        riga = {"Tempo": nome}
        for pos, (i, j, p) in enumerate(ordine): riga[["Più probabile", "Seconda", "Terza", "Quarta"][pos]] = f"{i}-{j} ({p*100:.0f}%)"
        best, bd = None, 0.0
        for i in range(6):
            for j in range(6):
                if Hn[i, j] >= 0.02 and hist[i, j] >= 0.005:
                    delta = Hn[i, j] - hist[i, j]
                    if delta > bd: bd, best = delta, (i, j, Hn[i, j], hist[i, j])
        if best: riga["Sorpresa"] = f"{best[0]}-{best[1]} ({best[2]*100:.0f}% vs {best[3]*100:.0f}%)"
        righe.append(riga)
    return righe

def crea_radar_chart(c, t, r):
    cats = list(r.keys()); vc, vt = [round(r[x][0]/max(r[x][0], r[x][1], 0.1)*100, 1) for x in cats], [round(r[x][1]/max(r[x][0], r[x][1], 0.1)*100, 1) for x in cats]
    fig = go.Figure([go.Scatterpolar(r=vc+[vc[0]], theta=cats+[cats[0]], fill='toself', name=c, line=dict(color=COL_CASA)), go.Scatterpolar(r=vt+[vt[0]], theta=cats+[cats[0]], fill='toself', name=t, line=dict(color=COL_TRASF))])
    fig.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), template="plotly_dark", height=400, margin=dict(l=40, r=40, t=40, b=20)); return fig

def ultimi_5(df, sq):
    d = df[(df["HomeTeam"] == sq) | (df["AwayTeam"] == sq)].sort_values("Date").tail(5)
    return ["V" if (r.HomeTeam == sq and r.FTHG > r.FTAG) or (r.AwayTeam == sq and r.FTAG > r.FTHG) else ("N" if r.FTHG == r.FTAG else "P") for r in d.itertuples()]

def crea_heatmap(M):
    M = M[:6, :6] * 100
    fig = go.Figure(go.Heatmap(z=M, x=[str(i) for i in range(6)], y=[str(i) for i in range(6)], colorscale='Blues', text=np.round(M, 1), texttemplate="%{text}%"))
    fig.update_layout(title="Matrice Esatti", xaxis_title="Trasferta", yaxis_title="Casa", yaxis_autorange="reversed", template="plotly_dark", height=400); return fig

def tabella_classifica(df):
    r = {}
    for x in df.itertuples():
        for s, gf, gs in ((x.HomeTeam, x.FTHG, x.FTAG), (x.AwayTeam, x.FTAG, x.FTHG)):
            d = r.setdefault(s, {"G": 0, "Pt": 0, "GF": 0, "GS": 0}); d["G"] += 1; d["GF"] += gf; d["GS"] += gs; d["Pt"] += 3 if gf > gs else (1 if gf == gs else 0)
    return pd.DataFrame([{"Squadra": s, "Giocate": v["G"], "Punti": v["Pt"], "Media GF": round(v["GF"]/v["G"], 2), "Media GS": round(v["GS"]/v["G"], 2), "DR": round((v["GF"]-v["GS"])/v["G"], 2)} for s, v in r.items()]).sort_values(["Punti", "DR"], ascending=False).reset_index(drop=True)

def distribuzione_live(c, t, mod, pesi, Mfreq, qf, mxg=None):
    lh, la = lambda_partita(mod, c, t); Mb = matrice_punteggi(lh, la, mod["rho"])
    if mxg is not None: Mb = combina_modelli(Mb, matrice_punteggi(*lambda_partita(mxg, c, t), mxg["rho"]), 0.5)
    q = qf.get((c, t))
    if q is not None: return {"M_1x2": matrice_ancorata(q, mod["rho"], (lh, la)), "M_gol": matrice_ancorata(q, mod["rho"], (lh, la)), "M_es": matrice_ancorata(q, mod["rho"], (lh, la)), "ancorata": True, "fonte": "quote"}
    return {"M_1x2": Mb, "M_gol": 0.5*Mb + 0.5*Mfreq, "M_es": 0.75*Mb + 0.25*Mfreq, "ancorata": False, "fonte": "modello"}

def simula_mc(bank, stake, n, probs, odds):
    p = []
    for _ in range(100):
        b = bank; st = [b]
        for _ in range(n):
            b += stake*(odds[np.random.randint(0, len(odds))]-1) if np.random.rand() <= probs[np.random.randint(0, len(probs))] else -stake
            st.append(b)
        p.append(st)
    return np.array(p)

try: _k_fd = st.secrets.get("FOOTBALL_DATA_API_KEY", "")
except: _k_fd = ""
try: _k_od = st.secrets.get("THE_ODDS_API_KEY", "")
except: _k_od = ""

st.sidebar.markdown("### ⚙ API")
k_fd = st.sidebar.text_input("🔑 football-data.org", value=_k_fd, type="password")
k_od = st.sidebar.text_input("🔑 The Odds API", value=_k_od, type="password")

tipo = st.sidebar.radio("Competizione", ["Principali", "Extra", "Internazionali"])

# FIX SELEZIONE CAMPIONATO
if tipo == "Principali":
    camp = st.selectbox("Torneo", list(LEAGUES.keys()), key="sel_princ")
    div = LEAGUES[camp]
    is_i, is_e = False, False
elif tipo == "Extra":
    camp = st.selectbox("Torneo", list(EXTRA_LEAGUES.keys()), key="sel_extra")
    div = EXTRA_LEAGUES[camp]
    is_i, is_e = False, True
else:
    camp = st.selectbox("Torneo", list(INTL_COMPS.keys()), key="sel_intl")
    div = camp
    is_i, is_e = True, False

if "ultimo_camp" not in st.session_state or st.session_state["ultimo_camp"] != div:
    st.session_state["ultimo_camp"] = div
    if "m" in st.session_state: del st.session_state["m"]

giorno = date.today().isoformat()

if is_i:
    if not k_fd: st.warning("Inserisci la chiave football-data.org"); st.stop()
    df_s = carica_storico_intl(k_fd, INTL_COMPS[camp]["fd_id"], N_STAGIONI, giorno)
    qf = {}; q_raw = carica_odds_api(k_od, INTL_COMPS[camp]["odds_key"]); mp = mappa_nomi(list(set([x[0] for x in q_raw]+[x[1] for x in q_raw])), sorted(set(df_s["HomeTeam"])|set(df_s["AwayTeam"])))
    for (h, a), q in q_raw.items():
        if mp.get(h, h) in df_s["HomeTeam"].values or mp.get(h, h) in df_s["AwayTeam"].values: qf[(mp.get(h, h), mp.get(a, a))] = q
elif is_e:
    df_s = carica_storico(div, N_STAGIONI, giorno); cal = carica_calendario_extra_da_odds(k_od, div)
    qf = {}; q_raw = carica_odds_api(k_od, ODDS_API_KEYS.get(div, "")); mp = mappa_nomi(list(set([x[0] for x in q_raw]+[x[1] for x in q_raw])), sorted(set(df_s["HomeTeam"])|set(df_s["AwayTeam"])))
    for (h, a), q in q_raw.items():
        if mp.get(h, h) in df_s["HomeTeam"].values or mp.get(h, h) in df_s["AwayTeam"].values: qf[(mp.get(h, h), mp.get(a, a))] = q
else:
    df_s = carica_storico(div, N_STAGIONI, giorno); qf = {}

if df_s is None or len(df_s) < 50: st.warning("Dati insufficienti per questo campionato."); st.stop()
par = parametri_auto(div, giorno) if not is_i and not is_e else {"ridge": 2.0, "halflife": 365}
mod = adatta_modello(df_s, pd.Timestamp.now().normalize(), par["ridge"], par["halflife"])
mxg = None
if not is_i and not is_e and div in LEAGUES_XG:
    df_x = carica_xg_storico(div, giorno)
    if df_x is not None: mxg = adatta_modello(df_x.assign(FTHG=df_x["xGH"].round().astype(int), FTAG=df_x["xGA"].round().astype(int)), pd.Timestamp.now().normalize(), par["ridge"], par["halflife"])

if is_i:
    cal = pd.DataFrame(); 
    if k_fd:
        res = requests.get(f"https://api.football-data.org/v4/competitions/{INTL_COMPS[camp]['fd_id']}/matches?status=SCHEDULED", headers={"X-Auth-Token": k_fd})
        if res.status_code == 200:
            cal = pd.DataFrame([{"Giornata": m.get("matchday") or 1, "Data": pd.to_datetime(m["utcDate"]).tz_convert("Europe/Rome").strftime("%Y-%m-%d"), "Ora": pd.to_datetime(m["utcDate"]).tz_convert("Europe/Rome").strftime("%H:%M"), "Casa": m["homeTeam"]["name"], "Trasferta": m["awayTeam"]["name"], "Stato": m["status"]} for m in res.json().get("matches", [])])
elif not is_e:
    fx, _ = carica_fixtures(div); cal, _ = costruisci_calendario(div, df_s, fx, k_fd, mod["squadre"])

Mfreq = matrice_frequenze(df_s); fc = frequenze_combo(df_s)

tabs = st.tabs(["📅 Calendario", "📋 Pronostici", "🏆 Classifica", "⚡ Value Finder", "🌍 Top Picks", "🎲 Monte Carlo", "📄 Report", "🧮 Kelly"])

def mostra(c, t, d, o):
    D = distribuzione_live(c, t, mod, {"gol":0.5, "esatto":0.75}, Mfreq, qf, mxg); lh, la = gol_attesi(D["M_gol"]); M = D["M_es"]; mk, mg = mercati(D["M_1x2"]), mercati(D["M_gol"]); pr = tabelle_pronostico(D["M_1x2"], D["M_gol"], fc)
    st.markdown(f"<div style='background:#111827;padding:18px;border-radius:12px;margin-bottom:12px;display:flex;justify-content:space-between;'><b>{c} vs {t}</b><span>📅 {d} {o}</span></div>", unsafe_allow_html=True)
    pp = pr["p"]; cg, _, _ = giocata_consigliata(pr); top = pr["pronostico"]
    st.success(f"**🎯 GIOCATA: {NOMI_SCELTA[cg]}** — {pp[top]*100:.1f}%\n\n*{NOMI_ESITO[top]}*")
    if st.button("🧮 Kelly"): st.session_state["kp"], st.session_state["kq"] = pp[cg]*100, round(1/pp[cg], 2)
    
    st.markdown(f"**Forma**\n{c}: {' '.join(ultimi_5(df_s, c))}\n{t}: {' '.join(ultimi_5(df_s, t))}")
    a, b, x = st.columns(3); a.metric("1", f"{mk['p1']*100:.1f}%"); b.metric("X", f"{mk['px']*100:.1f}%"); x.metric("2", f"{mk['p2']*100:.1f}%")
    
    st.markdown("---")
    st.markdown("### ⏱️ Tempi")
    if is_i or is_e: st.info("Dati tempi non disponibili per questa competizione.")
    else:
        T = tabelle_tempi_complete(D["M_gol"], df_s)
        col1, col2 = st.columns(2)
        col1.markdown("**1T 1X2**"); col1.write(f"1: {T['1x2_1t']['1']*100:.1f}% | X: {T['1x2_1t']['X']*100:.1f}% | 2: {T['1x2_1t']['2']*100:.1f}%")
        col2.markdown("**2T 1X2**"); col2.write(f"1: {T['1x2_2t']['1']*100:.1f}% | X: {T['1x2_2t']['X']*100:.1f}% | 2: {T['1x2_2t']['2']*100:.1f}%")
        st.markdown("**HT/FT**"); st.dataframe(pd.DataFrame([{"Esito": f"{['1','X','2'][i]}/{['1','X','2'][j]}", "Prob %": round(T['HTFT'][i,j]*100, 1)} for i in range(3) for j in range(3)]).sort_values("Prob %", ascending=False))
        st.markdown("**Over/Under 1T**"); st.write(f"O0.5: {T['ou_1t']['Over 0.5']*100:.1f}% | O1.5: {T['ou_1t']['Over 1.5']*100:.1f}% | O2.5: {T['ou_1t']['Over 2.5']*100:.1f}%")
        st.markdown("**Over/Under 2T**"); st.write(f"O0.5: {T['ou_2t']['Over 0.5']*100:.1f}% | O1.5: {T['ou_2t']['Over 1.5']*100:.1f}% | O2.5: {T['ou_2t']['Over 2.5']*100:.1f}%")
        st.markdown("**Esatti 1T/2T (sorpresa)**"); st.dataframe(pd.DataFrame(esatti_per_tempo(D["M_gol"], df_s)))

with tabs[0]:
    if not cal.empty:
        if "Giornata" not in cal.columns: cal["Giornata"] = 1
        gs = sorted(cal["Giornata"].unique()); gsel = st.selectbox("Giornata", gs)
        for i, r in cal[cal["Giornata"] == gsel].iterrows():
            st.markdown(f"**{r['Casa']} vs {r['Trasferta']}** ({r.get('Data','')})")
            if st.button("Analizza", key=f"btn_{i}"): st.session_state["m"] = r
    if "m" in st.session_state: mostra(st.session_state["m"]["Casa"], st.session_state["m"]["Trasferta"], st.session_state["m"]["Data"], st.session_state["m"].get("Ora", ""))

with tabs[1]:
    rp = []
    for _, r in cal.iterrows():
        D = distribuzione_live(r["Casa"], r["Trasferta"], mod, {"gol":0.5, "esatto":0.75}, Mfreq, qf, mxg); p = tabelle_pronostico(D["M_1x2"], D["M_gol"], fc); cg, _, _ = giocata_consigliata(p)
        rp.append({"Sel": False, "Data": r["Data"], "Partita": f"{r['Casa']}-{r['Trasferta']}", "Giocata": NOMI_SCELTA[cg], "Prob %": round(p["p"][cg]*100, 1), "Quota Equa": round(1/p["p"][cg], 2)})
    if rp:
        ed = st.data_editor(pd.DataFrame(rp), column_config={"Sel": st.column_config.CheckboxColumn(default=False)}, disabled=["Data","Partita","Giocata","Prob %","Quota Equa"])
        sel = ed[ed["Sel"]]
        if not sel.empty:
            st.metric("Quota Totale", f"{np.prod(sel['Quota Equa']):.2f}"); st.metric("Prob Combinata", f"{np.prod(sel['Prob %']/100)*100:.2f}%")

with tabs[2]:
    if not is_i and not is_e: st.dataframe(tabella_classifica(df_s[df_s["Stag"] == df_s["Stag"].max()]))

with tabs[3]:
    rv = []
    for (c, t), q in qf.items():
        mk = mercati(matrice_punteggi(*lambda_partita(mod, c, t), mod["rho"])); P = np.array([mk["p1"], mk["px"], mk["p2"]]); pm = senza_margine(q)
        for k in range(3):
            e = (P[k]*q[k]-1)*100
            if e >= 3: rv.append({"Partita": f"{c}-{t}", "Sel": ["1","X","2"][k], "Prob": round(P[k]*100,1), "Quota": round(q[k],2), "Edge": round(e,1)})
    if rv: df_val = pd.DataFrame(rv).sort_values("Edge", ascending=False); st.dataframe(df_val)

with tabs[4]:
    if st.button("Scansiona"):
        tp = []
        for cp, dc in list(LEAGUES.items()) + list(EXTRA_LEAGUES.items()) + [(k, v["fd_id"]) for k, v in INTL_COMPS.items()]:
            ok = ODDS_API_MAIN.get(dc) if cp in LEAGUES else (ODDS_API_KEYS.get(dc) if cp in EXTRA_LEAGUES else INTL_COMPS[cp]["odds_key"])
            if not ok: continue
            qr = carica_odds_api(k_od, ok)
            if not qr: continue
            df_sm = carica_storico(dc, N_STAGIONI, giorno) if cp not in INTL_COMPS else carica_storico_intl(k_fd, dc, N_STAGIONI, giorno)
            if df_sm is None: continue
            msm = adatta_modello(df_sm, pd.Timestamp.now().normalize(), 2.0, 365)
            sq_mod = sorted(set(df_sm["HomeTeam"]) | set(df_sm["AwayTeam"]))
            mp = mappa_nomi(list(set([x[0] for x in qr]+[x[1] for x in qr])), sq_mod)
            for (h, a), q in qr.items():
                mh, ma = mp.get(h, h), mp.get(a, a)
                if mh not in sq_mod or ma not in sq_mod: continue
                mk = mercati(matrice_punteggi(*lambda_partita(msm, mh, ma), msm["rho"])); P = np.array([mk["p1"], mk["px"], mk["p2"]])
                for k in range(3):
                    e = (P[k]*q[k]-1)*100
                    if e >= 3: tp.append({"Campionato": cp, "Partita": f"{mh}-{ma}", "Edge": round(e,1)})
        if tp: st.dataframe(pd.DataFrame(tp).sort_values("Edge", ascending=False))

with tabs[5]:
    if "df_val" in locals() and not df_val.empty:
        st.metric("Bank", "€1000")
        if st.button("Simula"):
            P = simula_mc(1000, 20, 100, df_val["Prob"].values/100, df_val["Quota"].values)
            fig = go.Figure(go.Scatter(x=np.arange(101), y=P.mean(axis=0), line=dict(color='red')))
            for i in range(100): fig.add_trace(go.Scatter(x=np.arange(101), y=P[i], line=dict(color='blue', width=1), opacity=0.1, showlegend=False))
            st.plotly_chart(fig)

with tabs[6]:
    if "df_val" in locals() and not df_val.empty: st.download_button("Scarica CSV", df_val.to_csv(index=False), "report.csv")

with tabs[7]:
    p = st.slider("Prob %", 1, 100, int(st.session_state.get("kp", 45))); q = st.number_input("Quota", 1.01, 50.0, float(st.session_state.get("kq", 2.0))); b = 1000.0
    st.metric("Kelly", f"€{max(0, ((q-1)*(p/100) - (1-p/100)) / (q-1)) * 0.25 * b:.2f}")
