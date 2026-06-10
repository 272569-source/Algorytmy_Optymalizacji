from __future__ import annotations

import csv
import itertools
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class City:
    name: str
    x: float
    y: float


@dataclass
class ACOResult:
    best_tour: List[int]
    best_length: float
    history: List[float]


class AntColonyTSP:
    def __init__(
        self,
        distance_matrix: np.ndarray,
        ants: int = 30,
        iterations: int = 200,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.5,
        q: float = 100.0,
        elitist_weight: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.dist = np.asarray(distance_matrix, dtype=float)
        self.n = self.dist.shape[0]

        if self.dist.shape[0] != self.dist.shape[1]:
            raise ValueError("Macierz odległości musi być kwadratowa.")
        if self.n < 3:
            raise ValueError("TSP wymaga co najmniej 3 miast.")
        if np.any(self.dist < 0):
            raise ValueError("Odległości nie mogą być ujemne.")

        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.elitist_weight = elitist_weight
        self.verbose = verbose

        self.tau_min = 0.05
        self.tau_max = 10.0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.pheromone = np.ones((self.n, self.n), dtype=float)

        with np.errstate(divide="ignore"):
            self.visibility = 1.0 / self.dist
        self.visibility[np.isinf(self.visibility)] = 0.0
        np.fill_diagonal(self.visibility, 0.0)

    def solve(self) -> ACOResult:
        best_tour: Optional[List[int]] = None
        best_length = math.inf
        history: List[float] = []

        stagnation_counter = 0
        stagnation_limit = 20

        for iteration in range(self.iterations):
            all_tours: List[List[int]] = []
            all_lengths: List[float] = []

            improvement_found = False

            for _ in range(self.ants):
                tour = self._construct_tour()
                length = self._tour_length(tour)

                all_tours.append(tour)
                all_lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_tour = tour[:]
                    improvement_found = True

            if improvement_found:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            self._evaporate_pheromone()
            self._deposit_pheromone(all_tours, all_lengths)

            if best_tour is not None:
                self._deposit_elitist_pheromone(best_tour, best_length)

            self._limit_pheromone()

            if stagnation_counter >= stagnation_limit:
                if self.verbose:
                    print(
                        f"Iteracja {iteration + 1}: "
                        f"stagnacja -> częściowy reset feromonów"
                    )

                self._reset_pheromone_partially(0.5)
                stagnation_counter = 0

            history.append(best_length)

            step = max(1, self.iterations // 10)
            if self.verbose and ((iteration + 1) % step == 0 or iteration == 0):
                print(
                    f"Iteracja {iteration + 1:4d}/{self.iterations}: "
                    f"najlepsza długość = {best_length:.4f}"
                )

        if best_tour is None:
            raise RuntimeError("Nie udało się utworzyć żadnej trasy.")

        return ACOResult(best_tour=best_tour, best_length=best_length, history=history)

    def _construct_tour(self) -> List[int]:
        start = random.randrange(self.n)
        tour = [start]

        unvisited = set(range(self.n))
        unvisited.remove(start)

        current = start

        while unvisited:
            next_city = self._select_next_city(current, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        return tour

    def _select_next_city(self, current: int, unvisited: set[int]) -> int:
        candidates = np.array(list(unvisited), dtype=int)

        pheromone_values = self.pheromone[current, candidates] ** self.alpha
        visibility_values = self.visibility[current, candidates] ** self.beta
        desirability = pheromone_values * visibility_values

        total = desirability.sum()

        if total <= 0 or not np.isfinite(total):
            return int(random.choice(list(unvisited)))

        probabilities = desirability / total
        return int(np.random.choice(candidates, p=probabilities))

    def _evaporate_pheromone(self) -> None:
        self.pheromone *= 1.0 - self.evaporation_rate
        self.pheromone = np.maximum(self.pheromone, 1e-12)

    def _deposit_pheromone(
        self,
        tours: Sequence[Sequence[int]],
        lengths: Sequence[float],
    ) -> None:
        for tour, length in zip(tours, lengths):
            delta = self.q / length
            for a, b in edges_of_tour(tour):
                self.pheromone[a, b] += delta
                self.pheromone[b, a] += delta

    def _deposit_elitist_pheromone(
        self,
        tour: Sequence[int],
        length: float,
    ) -> None:
        delta = self.elitist_weight * self.q / length
        for a, b in edges_of_tour(tour):
            self.pheromone[a, b] += delta
            self.pheromone[b, a] += delta

    def _reset_pheromone_partially(self, reset_factor: float = 0.5) -> None:
        initial_pheromone = np.ones((self.n, self.n), dtype=float)

        self.pheromone = (
            (1.0 - reset_factor) * self.pheromone
            + reset_factor * initial_pheromone
        )

        np.fill_diagonal(self.pheromone, 0.0)

    def _limit_pheromone(self) -> None:
        self.pheromone = np.clip(
            self.pheromone,
            self.tau_min,
            self.tau_max,
        )

        np.fill_diagonal(self.pheromone, 0.0)

    def _tour_length(self, tour: Sequence[int]) -> float:
        return tour_length(self.dist, tour)


def edges_of_tour(tour: Sequence[int]) -> Iterable[Tuple[int, int]]:
    for i in range(len(tour)):
        yield tour[i], tour[(i + 1) % len(tour)]


def tour_length(distance_matrix: np.ndarray, tour: Sequence[int]) -> float:
    return sum(distance_matrix[a, b] for a, b in edges_of_tour(tour))


def euclidean_distance_matrix(cities: Sequence[City]) -> np.ndarray:
    n = len(cities)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            dx = cities[i].x - cities[j].x
            dy = cities[i].y - cities[j].y
            d = math.hypot(dx, dy)

            matrix[i, j] = d
            matrix[j, i] = d

    return matrix


def load_cities_from_csv(path: str | Path) -> List[City]:
    cities: List[City] = []

    with open(path, "r", encoding="utf-8-sig") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        raise ValueError("Plik CSV jest pusty.")

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        cleaned_lines.append(line)

    header = cleaned_lines[0]

    if ";" in header:
        delimiter = ";"
    elif "," in header:
        delimiter = ","
    else:
        raise ValueError("Nie wykryto separatora.")

    columns = [col.strip().lower() for col in header.split(delimiter)]

    if columns != ["name", "x", "y"]:
        raise ValueError(
            f"CSV musi mieć nagłówek: name{delimiter}x{delimiter}y. "
            f"Wykryty nagłówek: {columns}"
        )

    for line in cleaned_lines[1:]:
        parts = [part.strip() for part in line.split(delimiter)]

        if len(parts) != 3:
            raise ValueError(f"Nieprawidłowy wiersz CSV: {line}")

        name = parts[0]
        x = float(parts[1].replace(",", "."))
        y = float(parts[2].replace(",", "."))

        cities.append(City(name, x, y))

    if len(cities) < 3:
        raise ValueError("Podaj co najmniej 3 miasta.")

    return cities


def generate_random_cities(
    n: int,
    width: int = 100,
    height: int = 100,
    seed: Optional[int] = None,
) -> List[City]:
    rng = random.Random(seed)

    return [
        City(
            name=f"C{i}",
            x=rng.uniform(0, width),
            y=rng.uniform(0, height),
        )
        for i in range(n)
    ]


def print_solution(cities: Sequence[City], result: ACOResult) -> None:
    route_names = [cities[i].name for i in result.best_tour]
    route_names.append(cities[result.best_tour[0]].name)

    print("\nNajlepsza znaleziona trasa:")
    print(" -> ".join(route_names))
    print(f"Długość trasy: {result.best_length:.4f}")


def plot_solution(cities: Sequence[City], result: ACOResult) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Brak biblioteki matplotlib.") from exc

    xs = [city.x for city in cities]
    ys = [city.y for city in cities]

    tour = result.best_tour + [result.best_tour[0]]
    tour_x = [cities[i].x for i in tour]
    tour_y = [cities[i].y for i in tour]

    plt.figure()
    plt.scatter(xs, ys)
    plt.plot(tour_x, tour_y, marker="o")

    for city in cities:
        plt.text(city.x, city.y, city.name, fontsize=8)

    plt.title(f"Najlepsza trasa ACO, długość = {result.best_length:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    plt.figure()
    plt.plot(result.history)
    plt.title("Zbieżność algorytmu")
    plt.xlabel("Iteracja")
    plt.ylabel("Najlepsza długość trasy")
    plt.tight_layout()

    plt.show()


# -------------------------
# Menu tekstowe
# -------------------------


def ask_int(prompt: str, default: int, minimum: Optional[int] = None) -> int:
    while True:
        text = input(f"{prompt} [{default}]: ").strip()

        if text == "":
            value = default
        else:
            try:
                value = int(text)
            except ValueError:
                print("Błąd: wpisz liczbę całkowitą.")
                continue

        if minimum is not None and value < minimum:
            print(f"Błąd: wartość musi być >= {minimum}.")
            continue

        return value


def ask_float(
    prompt: str,
    default: float,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    while True:
        text = input(f"{prompt} [{default}]: ").strip().replace(",", ".")

        if text == "":
            value = default
        else:
            try:
                value = float(text)
            except ValueError:
                print("Błąd: wpisz liczbę, np. 0.5 albo 2.")
                continue

        if minimum is not None and value < minimum:
            print(f"Błąd: wartość musi być >= {minimum}.")
            continue

        if maximum is not None and value > maximum:
            print(f"Błąd: wartość musi być <= {maximum}.")
            continue

        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    default_text = "t" if default else "n"

    while True:
        text = input(f"{prompt} [t/n, domyślnie {default_text}]: ").strip().lower()

        if text == "":
            return default

        if text in {"t", "tak", "y", "yes"}:
            return True

        if text in {"n", "nie", "no"}:
            return False

        print("Błąd: wpisz t albo n.")


def show_main_menu() -> None:
    print("\n" + "=" * 60)
    print(" ALGORYTM MRÓWKOWY DLA PROBLEMU KOMIWOJAŻERA ")
    print("=" * 60)
    print("1. Wygeneruj losowe miasta")
    print("2. Wczytaj miasta z pliku CSV")
    print("3. Wyjdź")
    print("=" * 60)


def get_cities_from_menu() -> Optional[List[City]]:
    while True:
        show_main_menu()
        choice = input("Wybierz opcję: ").strip()

        if choice == "1":
            n = ask_int("Podaj liczbę miast", default=40, minimum=3)
            seed = ask_int("Podaj ziarno losowości", default=42)
            return generate_random_cities(n, seed=seed)

        if choice == "2":
            path = input("Podaj ścieżkę do pliku CSV: ").strip().strip('"')

            try:
                return load_cities_from_csv(path)
            except Exception as exc:
                print(f"Nie udało się wczytać pliku: {exc}")
                print("Spróbuj ponownie.")

        if choice == "3":
            return None

        print("Nieprawidłowa opcja. Wybierz 1, 2 albo 3.")


def get_algorithm_parameters_from_menu() -> dict:
    print("\nParametry algorytmu")
    print("Możesz nacisnąć Enter, żeby zostawić wartość domyślną.\n")

    ants = ask_int("Liczba mrówek", default=40, minimum=1)
    iterations = ask_int("Liczba iteracji", default=200, minimum=1)
    alpha = ask_float("Alpha, czyli wpływ feromonu", default=1.0, minimum=0.0)
    beta = ask_float("Beta, czyli wpływ odległości", default=2.0, minimum=0.0)
    evaporation = ask_float(
        "Parowanie feromonu",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
    )
    q = ask_float("Stała Q odkładania feromonu", default=100.0, minimum=0.000001)
    elitist = ask_float("Wzmocnienie najlepszej trasy", default=2.0, minimum=0.0)
    plot = ask_yes_no("Czy pokazać wykres trasy i zbieżności?", default=True)

    return {
        "ants": ants,
        "iterations": iterations,
        "alpha": alpha,
        "beta": beta,
        "evaporation_rate": evaporation,
        "q": q,
        "elitist_weight": elitist,
        "plot": plot,
    }


def run_once() -> None:
    cities = get_cities_from_menu()

    if cities is None:
        print("Zakończono program.")
        return

    params = get_algorithm_parameters_from_menu()

    distance_matrix = euclidean_distance_matrix(cities)

    solver = AntColonyTSP(
        distance_matrix=distance_matrix,
        ants=params["ants"],
        iterations=params["iterations"],
        alpha=params["alpha"],
        beta=params["beta"],
        evaporation_rate=params["evaporation_rate"],
        q=params["q"],
        elitist_weight=params["elitist_weight"],
        seed=42,
        verbose=True,
    )

    print("\nRozpoczynam obliczenia...\n")

    result = solver.solve()
    print_solution(cities, result)

    if params["plot"]:
        plot_solution(cities, result)


def main() -> None:
    while True:
        run_once()

        again = ask_yes_no("\nCzy uruchomić program jeszcze raz?", default=False)

        if not again:
            print("Koniec.")
            break


# -------------------------
# Funkcje pomocnicze do testów / eksperymentów
# -------------------------


def print_test_header(name: str) -> None:
    print("\n" + "=" * 70)
    print(f"START TESTU: {name}")
    print("=" * 70)


def print_progress(test_name: str, done: int, total: int) -> None:
    percent_done = (done / total) * 100
    percent_left = 100 - percent_done

    print(
        f"[{test_name}] "
        f"{done}/{total} | "
        f"wykonano: {percent_done:6.2f}% | "
        f"pozostało: {percent_left:6.2f}%"
    )


def save_results(filename: str, results: List[dict]) -> None:
    if not results:
        print(f"Brak wyników do zapisania w pliku {filename}.")
        return

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Zapisano wyniki do pliku: {filename}")


def run_experiment(
    city_count: int,
    ants: int,
    iterations: int,
    seed: int,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation_rate: float = 0.5,
    q: float = 100.0,
    elitist_weight: float = 2.0,
) -> dict:
    cities = generate_random_cities(city_count, seed=seed)
    distance_matrix = euclidean_distance_matrix(cities)

    start = time.perf_counter()

    solver = AntColonyTSP(
        distance_matrix=distance_matrix,
        ants=ants,
        iterations=iterations,
        alpha=alpha,
        beta=beta,
        evaporation_rate=evaporation_rate,
        q=q,
        elitist_weight=elitist_weight,
        seed=seed,
        verbose=False,
    )

    result = solver.solve()

    elapsed = time.perf_counter() - start

    return {
        "cities": city_count,
        "ants": ants,
        "iterations": iterations,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "evaporation_rate": evaporation_rate,
        "q": q,
        "elitist_weight": elitist_weight,
        "best_length": result.best_length,
        "time_sec": elapsed,
    }


# -------------------------
# Dodatkowe algorytmy porównawcze
# -------------------------


def brute_force_tsp(distance_matrix: np.ndarray) -> Tuple[List[int], float]:
    """
    Dokładne rozwiązanie TSP metodą pełnego przeglądu.

    Uwaga:
    Ta metoda nadaje się tylko dla małej liczby miast, np. 8-10,
    ponieważ liczba permutacji rośnie bardzo szybko.
    """
    n = len(distance_matrix)
    best_tour: Optional[List[int]] = None
    best_length = math.inf

    # Zaczynamy zawsze od miasta 0.
    # Dzięki temu nie liczymy wielokrotnie tych samych cykli.
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm)
        length = tour_length(distance_matrix, tour)

        if length < best_length:
            best_length = length
            best_tour = tour

    if best_tour is None:
        raise RuntimeError("Brute force nie znalazł żadnej trasy.")

    return best_tour, best_length


def nearest_neighbor_tsp(
    distance_matrix: np.ndarray,
    start: int = 0,
) -> Tuple[List[int], float]:
    """
    Prosta heurystyka najbliższego sąsiada.

    W każdym kroku wybiera najbliższe jeszcze nieodwiedzone miasto.
    """
    n = len(distance_matrix)

    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)

    current = start

    while unvisited:
        next_city = min(
            unvisited,
            key=lambda city: distance_matrix[current, city],
        )

        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    length = tour_length(distance_matrix, tour)

    return tour, length


# -------------------------
# Dotychczasowe testy / eksperymenty
# -------------------------


def test_ants() -> None:
    test_name = "test_ants - wpływ liczby mrówek"
    print_test_header(test_name)

    results = []
    ants_values = [10, 20, 40, 80]

    for index, ants in enumerate(ants_values, start=1):
        row = run_experiment(
            city_count=50,
            ants=ants,
            iterations=200,
            seed=42,
        )

        results.append(row)
        print_progress(test_name, index, len(ants_values))

    save_results("test_ants.csv", results)


def test_iterations() -> None:
    test_name = "test_iterations - wpływ liczby iteracji"
    print_test_header(test_name)

    results = []
    iteration_values = [50, 100, 200, 500]

    for index, iterations in enumerate(iteration_values, start=1):
        row = run_experiment(
            city_count=50,
            ants=40,
            iterations=iterations,
            seed=42,
        )

        results.append(row)
        print_progress(test_name, index, len(iteration_values))

    save_results("test_iterations.csv", results)


def test_cities() -> None:
    test_name = "test_cities - wpływ liczby miast"
    print_test_header(test_name)

    results = []

    city_counts = [20, 50, 100, 200]
    seeds = list(range(10))
    total = len(city_counts) * len(seeds)
    done = 0

    for city_count in city_counts:
        for seed in seeds:
            results.append(
                run_experiment(
                    city_count=city_count,
                    ants=40,
                    iterations=200,
                    seed=seed,
                )
            )

            done += 1
            print_progress(test_name, done, total)

    save_results("test_cities.csv", results)


# -------------------------
# NOWE TESTY: punkt 1, 2, 3
# -------------------------


def test_against_optimum() -> None:
    """
    Punkt 1:
    Porównanie ACO z dokładnym optimum dla małej liczby miast.
    """
    test_name = "test_against_optimum - porównanie z optimum"
    print_test_header(test_name)

    results = []

    city_count = 8
    seeds = list(range(10))
    total = len(seeds)

    for index, seed in enumerate(seeds, start=1):
        cities = generate_random_cities(city_count, seed=seed)
        distance_matrix = euclidean_distance_matrix(cities)

        brute_start = time.perf_counter()
        _, optimum_length = brute_force_tsp(distance_matrix)
        brute_time = time.perf_counter() - brute_start

        aco_start = time.perf_counter()

        solver = AntColonyTSP(
            distance_matrix=distance_matrix,
            ants=40,
            iterations=200,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.5,
            q=100.0,
            elitist_weight=2.0,
            seed=seed,
            verbose=False,
        )

        result = solver.solve()
        aco_time = time.perf_counter() - aco_start

        error_percent = (
            (result.best_length - optimum_length) / optimum_length
        ) * 100

        results.append(
            {
                "cities": city_count,
                "seed": seed,
                "aco_length": result.best_length,
                "optimum_length": optimum_length,
                "error_percent": error_percent,
                "aco_time_sec": aco_time,
                "brute_force_time_sec": brute_time,
            }
        )

        print_progress(test_name, index, total)

    save_results("test_against_optimum.csv", results)


def test_vs_nearest_neighbor() -> None:
    """
    Punkt 2:
    Porównanie ACO z prostą heurystyką najbliższego sąsiada.
    """
    test_name = "test_vs_nearest_neighbor - porównanie z najbliższym sąsiadem"
    print_test_header(test_name)

    results = []

    city_count = 50
    seeds = list(range(20))
    total = len(seeds)

    for index, seed in enumerate(seeds, start=1):
        cities = generate_random_cities(city_count, seed=seed)
        distance_matrix = euclidean_distance_matrix(cities)

        nn_start = time.perf_counter()
        _, nn_length = nearest_neighbor_tsp(distance_matrix, start=0)
        nn_time = time.perf_counter() - nn_start

        aco_start = time.perf_counter()

        solver = AntColonyTSP(
            distance_matrix=distance_matrix,
            ants=40,
            iterations=200,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.5,
            q=100.0,
            elitist_weight=2.0,
            seed=seed,
            verbose=False,
        )

        result = solver.solve()
        aco_time = time.perf_counter() - aco_start

        improvement_percent = (
            (nn_length - result.best_length) / nn_length
        ) * 100

        results.append(
            {
                "cities": city_count,
                "seed": seed,
                "nearest_neighbor_length": nn_length,
                "aco_length": result.best_length,
                "improvement_percent": improvement_percent,
                "nearest_neighbor_time_sec": nn_time,
                "aco_time_sec": aco_time,
            }
        )

        print_progress(test_name, index, total)

    save_results("test_vs_nearest_neighbor.csv", results)


def test_alpha() -> None:
    """
    Punkt 3a:
    Badanie wpływu parametru alpha, czyli wpływu feromonu.
    """
    test_name = "test_alpha - wpływ parametru alpha"
    print_test_header(test_name)

    results = []
    alpha_values = [0.0, 0.5, 1.0, 2.0, 5.0]

    for index, alpha in enumerate(alpha_values, start=1):
        row = run_experiment(
            city_count=50,
            ants=40,
            iterations=200,
            seed=42,
            alpha=alpha,
            beta=2.0,
            evaporation_rate=0.5,
        )

        results.append(row)
        print_progress(test_name, index, len(alpha_values))

    save_results("test_alpha.csv", results)


def test_beta() -> None:
    """
    Punkt 3b:
    Badanie wpływu parametru beta, czyli wpływu odległości.
    """
    test_name = "test_beta - wpływ parametru beta"
    print_test_header(test_name)

    results = []
    beta_values = [0.0, 1.0, 2.0, 5.0, 10.0]

    for index, beta in enumerate(beta_values, start=1):
        row = run_experiment(
            city_count=50,
            ants=40,
            iterations=200,
            seed=42,
            alpha=1.0,
            beta=beta,
            evaporation_rate=0.5,
        )

        results.append(row)
        print_progress(test_name, index, len(beta_values))

    save_results("test_beta.csv", results)


def test_evaporation() -> None:
    """
    Punkt 3c:
    Badanie wpływu parametru evaporation_rate, czyli parowania feromonu.
    """
    test_name = "test_evaporation - wpływ parowania feromonu"
    print_test_header(test_name)

    results = []
    evaporation_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    for index, evaporation_rate in enumerate(evaporation_values, start=1):
        row = run_experiment(
            city_count=50,
            ants=40,
            iterations=200,
            seed=42,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=evaporation_rate,
        )

        results.append(row)
        print_progress(test_name, index, len(evaporation_values))

    save_results("test_evaporation.csv", results)


def run_all_tests() -> None:
    print("\nUruchomiono tryb testowy.")
    print("Wyniki zostaną zapisane do plików CSV w katalogu programu.")

    # Testy, które były już wcześniej w kodzie.
    test_ants()
    test_iterations()
    test_cities()

    # Nowe testy dodane do rozbudowy projektu.
    test_against_optimum()
    test_vs_nearest_neighbor()
    test_alpha()
    test_beta()
    test_evaporation()

    print("\nWszystkie eksperymenty zakończone.")


if __name__ == "__main__":
    mode = input("Tryb [menu/test]: ").strip().lower()

    if mode == "test":
        run_all_tests()
    else:
        main()
