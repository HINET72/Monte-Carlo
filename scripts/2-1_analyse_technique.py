import os
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

def afficher_tous_graphiques(ticker="TTE.PA", date_debut="2020-01-01", date_fin=None):
    if date_fin is None:
        # Calcul dynamique de la date de fin = jour ouvré précédent
        date_fin = pd.Timestamp.today()
        while date_fin.weekday() >= 5:  # 5 = samedi, 6 = dimanche
            date_fin -= pd.Timedelta(days=1)
        date_fin = date_fin.strftime("%Y-%m-%d")

    data, close_col = telecharger_donnees(ticker, date_debut, date_fin)

    data['EMA20'] = data[close_col].ewm(span=20, adjust=False).mean()
    data['RSI'] = calcul_rsi(data[close_col])
    data['LogReturn'] = np.log(data[close_col] / data[close_col].shift(1))
    data = data.dropna(subset=['LogReturn'])

    log_returns = data['LogReturn']
    mu = log_returns.mean() * 252
    sigma = log_returns.std() * np.sqrt(252)
    S0 = data[close_col].iloc[-1]
    nb_jours = (data.index[-1] - data.index[0]).days
    T = nb_jours / 365  # en années

    sauvegarder_parametres_simulation(S0=S0.item(), mu=mu.item(), sigma=sigma.item(), T=T)

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

    cursor = Cursor(ax1, useblit=True, horizOn=True, vertOn=True, color='gray', linewidth=1)
    fig.canvas.mpl_connect('motion_notify_event', lambda event: fig.canvas.draw_idle())

    plt.tight_layout()

    # Création du dossier s'il n'existe pas
    output_dir = "Inputs-Outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Nom et chemin du fichier image
    filepath = os.path.join(output_dir, f"{ticker}_graphique_{date_debut}_to_{date_fin}.png")

    # Sauvegarde de la figure
    fig.savefig(filepath)
    plt.close(fig)  # ferme la figure pour libérer la mémoire

    print(f"✅ Graphique sauvegardé dans {filepath}")

if __name__ == "__main__":
    afficher_tous_graphiques()
