import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def telecharger_donnees(ticker, date_debut, date_fin):
    data = yf.download(ticker, start=date_debut, end=date_fin, auto_adjust=True)
    if data.empty:
        raise ValueError("❌ Aucune donnée reçue.")
    return data

def calculer_parametres(data):
    close_prices = data['Close'].dropna()
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    mu = log_returns.mean() * 252
    sigma = log_returns.std() * np.sqrt(252)
    S0 = close_prices.iloc[-1]
    return float(S0), float(mu), float(sigma)

def simuler_trajectoires_monte_carlo(S0, mu, sigma, T, N, M):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    trajectoires = np.zeros((N + 1, M))
    trajectoires[0, :] = S0

    for i in range(1, N + 1):
        Z = np.random.standard_normal(M)
        trajectoires[i, :] = trajectoires[i - 1, :] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return t, trajectoires

def comparer_simulations(ticker="TTE.PA"):
    # Périodes
    periode_longue = ("2020-01-01", "2025-07-01")
    periode_courte = ("2024-01-01", "2025-07-01")

    # Télécharger données longues et courtes
    data_longue = telecharger_donnees(ticker, *periode_longue)
    data_courte = telecharger_donnees(ticker, *periode_courte)

    # Calculer paramètres
    S0_long, mu_long, sigma_long = calculer_parametres(data_longue)
    S0_court, mu_court, sigma_court = calculer_parametres(data_courte)

    print(f"Période longue ({periode_longue[0]} à {periode_longue[1]}) : mu={mu_long:.4f}, sigma={sigma_long:.4f}")
    print(f"Période courte ({periode_courte[0]} à {periode_courte[1]}) : mu={mu_court:.4f}, sigma={sigma_court:.4f}")

    # Simulation
    T = 1      # horizon de simulation : 1 an
    N = 252    # nombre de pas (jours de bourse)
    M = 20     # nombre de trajectoires

    np.random.seed(42)  # pour la reproductibilité

    t_long, traj_long = simuler_trajectoires_monte_carlo(S0_long, mu_long, sigma_long, T, N, M)
    t_court, traj_court = simuler_trajectoires_monte_carlo(S0_court, mu_court, sigma_court, T, N, M)

    # Plot
    plt.figure(figsize=(14,7))

    # Longue période - en bleu
    for i in range(M):
        plt.plot(t_long, traj_long[:, i], color='blue', alpha=0.3)
    plt.plot(t_long, np.mean(traj_long, axis=1), color='blue', label='Moyenne trajectoires (période longue)', linewidth=2)

    # Courte période - en rouge
    for i in range(M):
        plt.plot(t_court, traj_court[:, i], color='red', alpha=0.3)
    plt.plot(t_court, np.mean(traj_court, axis=1), color='red', label='Moyenne trajectoires (période courte)', linewidth=2)

    plt.title(f"Simulation Monte Carlo - {ticker}\nComparaison périodes longue vs courte")
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix simulé (€)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    comparer_simulations()
