import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def telecharger_donnees(ticker, date_debut, date_fin):
    data = yf.download(ticker, start=date_debut, end=date_fin, auto_adjust=True)
    if data.empty:
        raise ValueError("❌ Aucune donnée reçue.")
    return data

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

def exemple_monte_carlo_totalenergies():
    # Paramètres fixes fournis
    S0 = 52.84
    mu = 0.07236838706371665
    sigma = 0.315083249588379
    T = 5.515068493150685  # années

    N = 252 * int(np.ceil(T))  # approx. nombre de pas (jours de trading)
    M = 10                    # nombre de trajectoires simulées

    t, trajectoires = simuler_trajectoires_monte_carlo(S0, mu, sigma, T, N, M)

    plt.figure(figsize=(12, 6))
    for i in range(min(M, 10)):
        plt.plot(t, trajectoires[:, i], alpha=0.3)
    plt.title(f"Simulation Monte Carlo GBM - Paramètres fixes (10 trajectoires)\n"
              f"S0 = {S0:.2f}, μ = {mu:.4f}, σ = {sigma:.4f}, T = {T:.2f} ans")
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix simulé (€)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    exemple_monte_carlo_totalenergies()
