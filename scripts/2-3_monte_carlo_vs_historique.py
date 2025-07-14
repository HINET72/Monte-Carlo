import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from matplotlib.widgets import Cursor

def calcul_rsi(close_prices, window=14):
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def telecharger_donnees(ticker, date_debut, date_fin):
    data = yf.download(ticker, start=date_debut, end=date_fin, auto_adjust=True, group_by='ticker')
    if data.empty:
        raise ValueError("❌ Aucune donnée reçue.")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    
    close_col = next((col for col in data.columns if col.endswith('_Close') or col == 'Close'), None)
    if not close_col:
        raise KeyError("❌ Colonne 'Close' introuvable.")
    
    return data, close_col

def sauvegarder_parametres_simulation(S0, mu, sigma, T, chemin_fichier="parametres_monte_carlo.json"):
    parametres = {
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "T": T
    }
    with open(chemin_fichier, "w") as f:
        json.dump(parametres, f)
    print(f"✅ Paramètres sauvegardés dans {chemin_fichier}")

def simuler_trajectoires_monte_carlo(S0, mu, sigma, T, N, M):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    trajectoires = np.zeros((N + 1, M), dtype=np.float64)
    trajectoires[0, :] = S0
    for i in range(1, N + 1):
        Z = np.random.standard_normal(M)
        trajectoires[i, :] = trajectoires[i - 1, :] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        )
    return t, trajectoires

def afficher_tous_graphiques(ticker="TTE.PA", date_debut="2020-01-01", date_fin="2025-07-01"):
    data, close_col = telecharger_donnees(ticker, date_debut, date_fin)

    data['EMA20'] = data[close_col].ewm(span=20, adjust=False).mean()
    data['RSI'] = calcul_rsi(data[close_col])
    data['LogReturn'] = np.log(data[close_col] / data[close_col].shift(1))
    data = data.dropna(subset=['LogReturn'])

    # **Utilisation des paramètres fixes au lieu de calcul dynamique**
    S0 = 52.84
    mu = 0.07236838706371665
    sigma = 0.315083249588379
    T = 5.515068493150685

    # Sauvegarde des paramètres (optionnel)
    sauvegarder_parametres_simulation(S0, mu, sigma, T)

    N = 252 * int(np.ceil(T))
    M = 10

    t, trajectoires = simuler_trajectoires_monte_carlo(S0, mu, sigma, T, N, M)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

    ax1.plot(data.index, data[close_col], label='Cours ajusté', color='blue')
    ax1.plot(data.index, data['EMA20'], label='EMA 20 jours', color='red', linestyle='--')
    ax1.set_ylabel('Prix (€)')
    ax1.set_title(f"{ticker} — Prix, EMA20, RSI et Log Returns ({date_debut} à {date_fin})")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(data.index, data['RSI'], label='RSI 14 jours', color='purple')
    ax2.axhline(70, color='red', linestyle='--', label='Surachat (70)')
    ax2.axhline(30, color='green', linestyle='--', label='Survente (30)')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(data.index, data['LogReturn'], label='Log Returns', color='darkgreen')
    ax3.set_ylabel('Log Return')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)

    # Affichage des trajectoires Monte Carlo sur le graphique prix
    debut_simulation = data.index[-1]
    temps_sim = np.linspace(0, T, N + 1)

    # On convertit le temps simulé en dates pour l'axe X
    dates_sim = pd.date_range(start=debut_simulation, periods=N+1, freq='B')  # B = jours ouvrés

    for i in range(M):
        ax1.plot(dates_sim, trajectoires[:, i], alpha=0.3, linestyle='--')

    from matplotlib.widgets import Cursor
    cursor = Cursor(ax1, useblit=True, horizOn=True, vertOn=True, color='gray', linewidth=1)
    fig.canvas.mpl_connect('motion_notify_event', lambda event: fig.canvas.draw_idle())

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    afficher_tous_graphiques()
