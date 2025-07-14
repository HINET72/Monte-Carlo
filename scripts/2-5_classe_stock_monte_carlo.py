import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class Stock:
    def __init__(self, ticker, date_debut="2020-01-01", date_fin=None):
        self.ticker = ticker
        self.date_debut = date_debut
        self.date_fin = date_fin
        self.data = None
        self.close_col = None
        self.S0 = None
        self.mu = None
        self.sigma = None
        self.T = None
        self.log_returns = None

    def telecharger_donnees(self):
        if self.date_fin is None:
            from datetime import datetime, timedelta
            yesterday = datetime.now() - timedelta(days=1)
            self.date_fin = yesterday.strftime("%Y-%m-%d")
        data = yf.download(self.ticker, start=self.date_debut, end=self.date_fin, auto_adjust=True, group_by='ticker')
        if data.empty:
            raise ValueError(f"Aucune donnée reçue pour {self.ticker} entre {self.date_debut} et {self.date_fin}.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        close_col = next((col for col in data.columns if col.endswith('_Close') or col == 'Close'), None)
        if not close_col:
            raise KeyError("Colonne 'Close' introuvable dans les données téléchargées.")
        self.data = data
        self.close_col = close_col
        return data

    @staticmethod
    def calcul_rsi(close_prices, window=14):
        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculer_indicateurs(self):
        if self.data is None:
            raise ValueError("Les données ne sont pas chargées.")
        self.data['EMA20'] = self.data[self.close_col].ewm(span=20, adjust=False).mean()
        self.data['RSI'] = self.calcul_rsi(self.data[self.close_col])
        self.data['LogReturn'] = np.log(self.data[self.close_col] / self.data[self.close_col].shift(1))
        self.data.dropna(subset=['LogReturn'], inplace=True)
        self.log_returns = self.data['LogReturn']

    def calculer_parametres(self):
        if self.log_returns is None:
            raise ValueError("Les log returns n'ont pas été calculés.")
        self.mu = float(self.log_returns.mean() * 252)
        self.sigma = float(self.log_returns.std() * np.sqrt(252))
        self.S0 = float(self.data[self.close_col].iloc[-1])
        nb_jours = (self.data.index[-1] - self.data.index[0]).days
        self.T = nb_jours / 365

    def sauvegarder_parametres(self, chemin_fichier="parametres_monte_carlo.json"):
        parametres = {
            "S0": self.S0,
            "mu": self.mu,
            "sigma": self.sigma,
            "T": self.T
        }
        with open(chemin_fichier, "w") as f:
            json.dump(parametres, f)
        print(f"✅ Paramètres sauvegardés dans {chemin_fichier}")

    def simuler_monte_carlo(self, N=252, M=100, seed=None):
        if self.S0 is None or self.mu is None or self.sigma is None or self.T is None:
            raise ValueError("Les paramètres ne sont pas initialisés. Exécutez calculer_parametres() d'abord.")
        if seed is not None:
            np.random.seed(seed)
        dt = self.T / N
        t = np.linspace(0, self.T, N + 1)
        trajectoires = np.zeros((N + 1, M))
        trajectoires[0, :] = self.S0
        for i in range(1, N + 1):
            Z = np.random.standard_normal(M)
            trajectoires[i, :] = trajectoires[i - 1, :] * np.exp(
                (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z
            )
        return t, trajectoires

    def afficher_tous_les_graphiques(self):
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Inputs-Outputs")
        os.makedirs(output_dir, exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
        ax1.plot(self.data.index, self.data[self.close_col], label='Prix ajusté', color='blue')
        ax1.plot(self.data.index, self.data['EMA20'], label='EMA 20 jours', color='red', linestyle='--')
        ax1.set_ylabel('Prix (€)')
        ax1.set_title(f"{self.ticker} — Prix ajusté, EMA20, RSI et Log Returns\n({self.date_debut} à {self.date_fin})")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.data.index, self.data['RSI'], label='RSI 14 jours', color='purple')
        ax2.axhline(70, color='red', linestyle='--', label='Surachat (70)')
        ax2.axhline(30, color='green', linestyle='--', label='Survente (30)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True)

        ax3.plot(self.data.index, self.data['LogReturn'], label='Log Returns', color='darkgreen')
        ax3.set_ylabel('Log Return')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        path1 = os.path.join(output_dir, f"graphique_btc_analyse_technique.png")
        plt.savefig(path1)
        plt.close()
        print(f"✅ Graphique technique sauvegardé dans : {path1}")

if __name__ == "__main__":
    action = Stock(ticker="BTC-USD", date_debut="2020-01-01")
    action.telecharger_donnees()
    action.calculer_indicateurs()
    action.calculer_parametres()
    action.sauvegarder_parametres()
    action.afficher_tous_les_graphiques()

    t, trajectoires = action.simuler_monte_carlo(N=252, M=10, seed=42)

    # Sauvegarde du graphique Monte Carlo
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Inputs-Outputs")
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for i in range(trajectoires.shape[1]):
        plt.plot(t, trajectoires[:, i], alpha=0.4)
    plt.title(f"Simulation Monte Carlo – {action.ticker}\n"
              f"S0={action.S0:.2f}, mu={action.mu:.4f}, sigma={action.sigma:.4f}, T={action.T:.2f}")
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix simulé (€)")
    plt.grid(True)
    plt.tight_layout()
    path2 = os.path.join(output_dir, f"graphique_btc_monte_carlo.png")
    plt.savefig(path2)
    plt.close()
    print(f"✅ Graphique Monte Carlo sauvegardé dans : {path2}")
