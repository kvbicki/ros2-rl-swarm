import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

HOME_DIR = os.path.expanduser('~')

PATHS_TO_CHECK = [
    'src/ros2-rl-swarm/swarm_manager/models/test.csv',
    os.path.join(HOME_DIR, 'ros2_ws/src/ros2-rl-swarm/swarm_manager/models/test.csv'),
    'test.csv'
]

csv_path = None
for path in PATHS_TO_CHECK:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    print("\n" + "="*60)
    print("BŁĄD KRYTYCZNY: Nie znaleziono pliku z danymi!")
    print("Sprawdzono lokalizacje:")
    for p in PATHS_TO_CHECK:
        print(f" - {os.path.abspath(p)}")
    print("\nROZWIĄZANIE: Uruchom najpierw trening, aby utworzyć plik.")
    print("="*60 + "\n")
    sys.exit(1)

print(f"--> Wczytywanie danych z: {csv_path}")

try:
    df = pd.read_csv(csv_path)
except pd.errors.EmptyDataError:
    print("BŁĄD: Plik csv istnieje, ale jest pusty.")
    sys.exit(1)

if df.empty:
    print("BŁĄD: Dataframe jest pusty.")
    sys.exit(1)

# Dopasowanie nazw kolumn do nowych danych
df.rename(columns={
    'steps': 'total_steps',
    'avg_reward': 'avg_reward_100',
    'collisions': 'collision_count'
}, inplace=True)

# Dodanie globalnego indeksu
df['global_index'] = range(len(df))

def plot_learning_curve(data):
    plt.figure()
    plt.plot(data['global_index'], data['reward'], color='lightgray', alpha=0.6, label='Nagroda (epizod)')
    plt.plot(data['global_index'], data['avg_reward_100'], color='crimson', linewidth=2, label='Średnia krocząca (100 ep.)')
    
    plt.title("Przebieg procesu uczenia algorytmu PPO")
    plt.xlabel("Liczba pomiarów (kolejne epizody)")
    plt.ylabel("Całkowita nagroda")
    plt.legend()
    plt.tight_layout()
    plt.savefig("wykres_1_krzywa_uczenia.png")
    print("Generowano: wykres_1_krzywa_uczenia.png")

def plot_episode_length(data):
    plt.figure()
    sns.scatterplot(data=data, x='global_index', y='total_steps', alpha=0.5, s=15, color='steelblue')
    
    if len(data) > 1:
        window = min(50, len(data))
        trend = data['total_steps'].rolling(window=window, min_periods=1).mean()
        plt.plot(data['global_index'], trend, color='navy', linewidth=2, label=f'Średnia ({window} pomiarów)')
    
    plt.title("Efektywność czasowa misji")
    plt.xlabel("Liczba pomiarów (kolejne epizody)")
    plt.ylabel("Liczba kroków symulacyjnych")
    plt.legend()
    plt.tight_layout()
    plt.savefig("wykres_2_dlugosc_epizodu.png")
    print("Generowano: wykres_2_dlugosc_epizodu.png")

def plot_collision_rate(data):
    if 'collision_count' not in data.columns:
        print("POMINIĘTO: Wykres 3 (brak kolumny 'collision_count').")
        return

    plt.figure()
    
    bin_size = max(1, int(len(data) / 10))
    
    data = data.copy()
    data['group'] = (data['global_index'] // bin_size) * bin_size
    grouped = data.groupby('group')['collision_count'].mean().reset_index()
    
    sns.barplot(data=grouped, x='group', y='collision_count', color='orange', alpha=0.8)
    
    plt.title("Średnia liczba kolizji w procesie uczenia")
    plt.xlabel(f"Zakres pomiarów (agregacja co {bin_size})")
    plt.ylabel("Średnia liczba kolizji na epizod")
    
    locs, labels = plt.xticks()
    for label in labels:
        label.set_rotation(45)
    
    plt.tight_layout()
    plt.savefig("wykres_3_wskaznik_kolizji.png")
    print("Generowano: wykres_3_wskaznik_kolizji.png")

def plot_reward_distribution(data):
    plt.figure()
    n = len(data)
    if n < 5:
        print("POMINIĘTO: Wykres 4 (zbyt mało danych).")
        return

    data = data.copy()
    data['Phase'] = pd.cut(data['global_index'], 
                           bins=[0, n*0.33, n*0.66, n], 
                           labels=['Faza początkowa', 'Faza środkowa', 'Faza końcowa'])
    
    data = data.dropna(subset=['Phase'])
    
    if data.empty:
         print("POMINIĘTO: Wykres 4 (błąd danych).")
         return

    sns.boxplot(x='Phase', y='reward', data=data, palette="Blues")
    plt.title("Rozkład nagród w poszczególnych fazach treningu")
    plt.xlabel("Etap uczenia")
    plt.ylabel("Całkowita nagroda")
    plt.tight_layout()
    plt.savefig("wykres_4_rozklad_nagrod.png")
    print("Generowano: wykres_4_rozklad_nagrod.png")

print(f"Przetwarzanie {len(df)} wierszy danych...")
plot_learning_curve(df)
plot_episode_length(df)
plot_collision_rate(df)
plot_reward_distribution(df)
print("\n--- Zakończono sukcesem ---")
