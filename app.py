import streamlit as st
import pandas as pd
import penaltyblog as pb
import numpy as np
from scipy.stats import poisson
import soccerdata as sd
from datetime import datetime

# --- CALCOLO DINAMICO DELLA STAGIONE (Siamo nel 2026) ---
anno_corrente = datetime.now().year
anno_inizio_stagione = anno_corrente if datetime.now().month >= 7 else anno_corrente - 1
stagione_fbref = f"{anno_inizio_stagione}-{anno_inizio_stagione+1}"
codice_football_data = f"{str(anno_inizio_stagione)[-2:]}{str(anno_inizio_stagione+1)[-2:]}"

st.set_page_config(page_title="Calcio Betting Model 2026", page_icon="⚽", layout="wide")

# --- BARRA LATERALE ---
st.sidebar.header("⚙️ Impostazioni Campionato")
campionato = st.sidebar.selectbox(
    "Scegli il campionato", 
    ["Premier League", "Serie A", "La Liga", "Bundesliga", "Ligue 1"]
)

mappa_campionati = {
    "Premier League": {"fd": "E0", "fbref": "ENG-Premier League"},
    "Serie A": {"fd": "I1", "fbref": "ITA-Serie A"},
    "La Liga": {"fd": "SP1", "fbref": "ESP-La Liga"},
    "Bundesliga": {"fd": "D1", "fbref": "GER-Bundesliga"},
    "Ligue 1": {"fd": "F1", "fbref": "FRA-Ligue 1"}
}
codice_lega_fd = mappa_campionati[campionato]["fd"]
nome_lega_fbref = mappa_campionati[campionato]["fbref"]

# --- DATI MODELLO (Gol, Angoli, Cartellini) ---
@st.cache_data
def carica_dati_modello(lega_fd, stagione_codice):
    stagione_prec = f"{int(stagione_codice[:2])-1:02d}{int(stagione_codice[2:])-1:02d}"
    url_corrente = f"https://www.football-data.co.uk/mmz4281/{stagione_codice}/{lega_fd}.csv"
    url_precedente = f"https://www.football-data.co.uk/mmz4281/{stagione_prec}/{lega_fd}.csv"
    try:
        df_corrente = pd.read_csv(url_corrente)
        df_precedente = pd.read_csv(url_precedente)
        data = pd.concat([df_precedente, df_corrente], ignore_index=True)
    except:
        data = pd.read_csv(url_precedente)
    return data.dropna(subset=["FTHG", "FTAG", "HC", "AC", "HY", "AY"])

@st.cache_resource
def allena_modello(data):
    model = pb.models.DixonColesGoalModel(
        goals_home=data["FTHG"], goals_away=data["FTAG"],
        teams_home=data["HomeTeam"], teams_away=data["AwayTeam"],
    )
    model.fit()
    return model

# --- DATI GIOCATORI (FBref tramite soccerdata) ---
@st.cache_data(show_spinner=False)
def carica_stats_giocatori(lega_fbref, stagione):
    try:
        # Percorso di Chromium installato tramite packages.txt su Streamlit Cloud
        fbref = sd.FBref(leagues=lega_fbref, seasons=stagione, path_to_browser='/usr/bin/chromium')
        
        df_standard = fbref.read_player_season_stats(stat_type="standard")
        df_shooting = fbref.read_player_season_stats(stat_type="shooting")
        
        df_merged = pd.merge(
            df_standard[['player', 'team', 'Gls', 'Min', 'CrdY']],
            df_shooting[['player', 'team', 'Sh', 'SoT']],
            on=['player', 'team'], how='left'
        ).rename(columns={
            'player': 'Giocatore', 'team': 'Squadra', 'Gls': 'Gol',
            'Min': 'Minuti', 'CrdY': 'Cartellini', 'Sh': 'Tiri Totali', 'SoT': 'Tiri in Porta'
        }).fillna(0)
        
        df_merged['Minuti'] = df_merged['Minuti'].replace(0, np.nan)
        df_merged['Tiri/90'] = (df_merged['Tiri Totali'] / df_merged['Minuti'] * 90).fillna(0)
        df_merged['Tiri in Porta/90'] = (df_merged['Tiri in Porta'] / df_merged['Minuti'] * 90).fillna(0)
        df_merged['Gol/90'] = (df_merged['Gol'] / df_merged['Minuti'] * 90).fillna(0)
        df_merged['Cartellini/90'] = (df_merged['Cartellini'] / df_merged['Minuti'] * 90).fillna(0)
        
        return df_merged[['Giocatore', 'Squadra', 'Minuti', 'Gol', 'Tiri Totali', 'Tiri in Porta', 'Cartellini', 'Tiri/90', 'Tiri in Porta/90', 'Gol/90', 'Cartellini/90']]
    except Exception as e:
        return f"ERRORE: {e}"

def filtra_giocatori(df, squadra_casa, squadra_trasferta):
    if isinstance(df, str): return df # Restituisce l'errore
    
    mappatura = {
        "Man United": "Manchester United", "Man City": "Manchester City",
        "Nott'm Forest": "Nottingham Forest", "Sheffield United": "Sheffield United",
        "West Ham": "West Ham United", "Newcastle": "Newcastle United",
        "Tottenham": "Tottenham Hotspur", "Wolves": "Wolverhampton Wanderers",
        "Brighton": "Brighton & Hove Albion", "Inter": "Inter", "Milan": "AC Milan",
        "Roma": "AS Roma", "Lazio": "Lazio", "Juventus": "Juventus", "Napoli": "Napoli"
    }
    sq_casa = mappatura.get(squadra_casa, squadra_casa)
    sq_trasferta = mappatura.get(squadra_trasferta, squadra_trasferta)
    
    df_filtrato = df[df['Squadra'].isin([sq_casa, sq_trasferta])].copy()
    if df_filtrato.empty:
        return "Nessun giocatore trovato per queste squadre nel database di FBref."
    return df_filtrato.sort_values(by='Tiri/90', ascending=False)

# --- INTERFACCIA PRINCIPALE ---
st.title(f"⚽ Pronostici Calcio Avanzati ({anno_inizio_stagione}/{anno_inizio_stagione+1})")
st.write(f"Stai analizzando: **{campionato}**")

with st.spinner("Caricamento dati e addestramento del modello..."):
    data = carica_dati_modello(codice_lega_fd, codice_football_data)
    model = allena_modello(data)

squadre = sorted(data['HomeTeam'].unique())
col1, col2 = st.columns(2)
with col1:
    casa = st.selectbox("Squadra in Casa", squadre)
with col2:
    trasferta = st.selectbox("Squadra in Trasferta", squadre, index=1 if len(squadre)>1 else 0)

if "pred" not in st.session_state: st.session_state.pred = None
if "match" not in st.session_state: st.session_state.match = None

if st.button("Calcola Pronostico", type="primary"):
    if casa == trasferta: 
        st.error("Seleziona due squadre diverse!")
    else:
        st.session_state.pred = model.predict(casa, trasferta)
        st.session_state.match = (casa, trasferta)

if st.session_state.pred is not None:
    pred = st.session_state.pred
    casa_match, trasferta_match = st.session_state.match
    
    st.header(f"{casa_match} vs {trasferta_match}")
    
    # Sezione 1: Mercati Base
    st.subheader("1️⃣ Pronostico di Base")
    c1, c2, c3 = st.columns(3)
    c1.metric("Vittoria Casa (1)", f"{pred.home_win:.1%}")
    c2.metric("Pareggio (X)", f"{pred.draw:.1%}")
    c3.metric("Vittoria Trasferta (2)", f"{pred.away_win:.1%}")
    c4, c5 = st.columns(2)
    c4.metric("Over 2.5", f"{pred.total_goals('over', 2.5):.1%}")
    c5.metric("Gol (BTTS Sì)", f"{pred.btts_yes:.1%}")
    st.divider()
    
    # Sezione 2: Risultati Esatti
    st.subheader("🎯 Risultati Esatti più probabili")
    scores = []
    for h in range(5):
        for a in range(5):
            prob = pred.exact_score(h, a)
            scores.append({"Punteggio": f"{h}-{a}", "Probabilità": prob, "Quota Minima": 1/prob})
    df_scores = pd.DataFrame(scores).sort_values(by="Probabilità", ascending=False).head(5)
    styled_scores = df_scores.style.format({'Probabilità': '{:.1%}', 'Quota Minima': '{:.2f}'}).background_gradient(cmap="Greens", subset=['Probabilità'])
    st.dataframe(styled_scores, use_container_width=True, hide_index=True)
    p_top3 = df_scores['Probabilità'].head(3).sum()
    st.info(f"💡 **Consiglio Multipla:** Probabilità totale dei primi 3: **{p_top3:.1%}** (Quota minima: {1/p_top3:.2f}).")
    st.divider()
    
    # Sezione 3: Combo HT/FT
    st.subheader("🔄 Combo: Esito 1° Tempo / Esito Finale")
    lambda_home_ht = pred.home_goal_expectation * 0.45
    lambda_away_ht = pred.away_goal_expectation * 0.45
    lambda_home_st = pred.home_goal_expectation * 0.55
    lambda_away_st = pred.away_goal_expectation * 0.55
    combinazioni = []
    nomi_esiti = {"1": "Casa", "X": "Pareggio", "2": "Trasferta"}
    for h_ht in range(5):
        for a_ht in range(5):
            p_ht = poisson.pmf(h_ht, lambda_home_ht) * poisson.pmf(a_ht, lambda_away_ht)
            if p_ht < 0.0001: continue
            esito_ht = "1" if h_ht > a_ht else ("X" if h_ht == a_ht else "2")
            for h_st in range(5):
                for a_st in range(5):
                    p_st = poisson.pmf(h_st, lambda_home_st) * poisson.pmf(a_st, lambda_away_st)
                    if p_st < 0.0001: continue
                    ft_h, ft_a = h_ht + h_st, a_ht + a_st
                    esito_ft = "1" if ft_h > ft_a else ("X" if ft_h == ft_a else "2")
                    combinazioni.append({"Combo": f"{esito_ht}/{esito_ft}", "Probabilità": p_ht * p_st})
    df_combo = pd.DataFrame(combinazioni).groupby("Combo")["Probabilità"].sum().reset_index().sort_values(by="Probabilità", ascending=False)
    df_combo["Quota Minima"] = 1 / df_combo["Probabilità"]
    df_combo["Significato"] = df_combo["Combo"].apply(lambda x: f"1°T: {nomi_esiti[x.split('/')[0]]} -> Finale: {nomi_esiti[x.split('/')[1]]}")
    styled_combo = df_combo[["Combo", "Significato", "Probabilità", "Quota Minima"]].style.format({'Probabilità': '{:.1%}', 'Quota Minima': '{:.2f}'}).background_gradient(cmap="Oranges", subset=['Probabilità'])
    st.dataframe(styled_combo, use_container_width=True, hide_index=True)
    st.divider()
    
    # Sezione 4: Angoli e Cartellini
    col_angoli, col_cards = st.columns(2)
    with col_angoli:
        st.subheader("🚩 Angoli")
        avg_home_c = data[data['HomeTeam'] == casa_match]['HC'].mean()
        avg_away_c = data[data['AwayTeam'] == trasferta_match]['AC'].mean()
        lambda_c = avg_home_c + avg_away_c
        p_over_95 = 1 - poisson.cdf(9, lambda_c)
        p_over_105 = 1 - poisson.cdf(10, lambda_c)
        st.metric("Over 9.5", f"{p_over_95:.1%}", help=f"Quota: {1/p_over_95:.2f}")
        st.metric("Over 10.5", f"{p_over_105:.1%}", help=f"Quota: {1/p_over_105:.2f}")
    with col_cards:
        st.subheader("🟨 Cartellini")
        avg_home_cd = data[data['HomeTeam'] == casa_match]['HY'].mean()
        avg_away_cd = data[data['AwayTeam'] == trasferta_match]['AY'].mean()
        lambda_cd = avg_home_cd + avg_away_cd
        p_over_35 = 1 - poisson.cdf(3, lambda_cd)
        p_over_45 = 1 - poisson.cdf(4, lambda_cd)
        st.metric("Over 3.5", f"{p_over_35:.1%}", help=f"Quota: {1/p_over_35:.2f}")
        st.metric("Over 4.5", f"{p_over_45:.1%}", help=f"Quota: {1/p_over_45:.2f}")
    st.divider()
    
    # Sezione 5: Mercati Giocatori
    st.subheader(f"👤 Mercati Giocatori (Dati Stagione {stagione_fbref})")
    if st.button("Carica Statistiche Giocatori"):
        with st.spinner(f"Download dati {campionato} {stagione_fbref} da FBref... (può richiedere 1-2 minuti la prima volta)"):
            df_stats = carica_stats_giocatori(nome_lega_fbref, stagione_fbref)
            df_filtrato = filtra_giocatori(df_stats, casa_match, trasferta_match)
            
            if isinstance(df_filtrato, str):
                st.error(df_filtrato)
            else:
                st.success("✅ Dati caricati con successo!")
                st.dataframe(df_filtrato, use_container_width=True, hide_index=True)
                st.markdown("#### 🎯 Top 3 per Tiri in Porta / 90 min")
                top3 = df_filtrato.nlargest(3, 'Tiri in Porta/90')[['Giocatore', 'Squadra', 'Tiri in Porta/90', 'Gol/90']]
                st.dataframe(top3, use_container_width=True, hide_index=True)
