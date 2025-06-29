
# benchmark.py
import logging
import time
import random
from data_loader import load_student_data
from aco import AntColonyOptimizer
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')

def random_assignment(students, num_teams, team_size):
    students_copy = students.copy()
    random.shuffle(students_copy)
    teams = [[] for _ in range(num_teams)]
    for i, student in enumerate(students_copy):
        teams[i % num_teams].append(student)
    return teams

def simulated_self_selection(students, num_teams, team_size):
    prefs_dict = defaultdict(list)
    for s in students:
        prefs_dict[s['preferences'][0]].append(s)
    teams = [[] for _ in range(num_teams)]
    i = 0
    for group in prefs_dict.values():
        for student in group:
            teams[i % num_teams].append(student)
            i += 1
    return teams

def evaluate_teams(teams):
    total_div = 0
    total_pref = 0
    for team in teams:
        roles = set()
        prefs = 0
        for s in team:
            roles.add(s['belbin_role'])
            try:
                prefs += 5 - s['preferences'].index(s['preferences'][0])
            except ValueError:
                pass
        total_div += len(roles) / 9.0
        total_pref += prefs / (5 * len(team)) if team else 0
    avg_div = total_div / len(teams)
    avg_pref = total_pref / len(teams)
    return avg_div, avg_pref

def run_all_methods(students):
    num_teams = 42
    team_size = 6

    # ACO
    aco = AntColonyOptimizer(
        students, num_teams, team_size,
        max_iter=50, num_ants=20,
        alpha=1.0, beta=3.0, rho=0.3,
        preference_weight=0.5
    )
    start = time.time()
    aco_teams, _ = aco.run()
    aco_time = time.time() - start
    aco_div, aco_pref = evaluate_teams(aco_teams)

    # Random
    start = time.time()
    rand_teams = random_assignment(students, num_teams, team_size)
    rand_time = time.time() - start
    rand_div, rand_pref = evaluate_teams(rand_teams)

    # Self-selection simulation
    start = time.time()
    self_teams = simulated_self_selection(students, num_teams, team_size)
    self_time = time.time() - start
    self_div, self_pref = evaluate_teams(self_teams)

    # Report
    print("\nResults Summary:")
    print(f"{'Method':<15} {'Diversity':<10} {'Preference':<12} {'Runtime (s)'}")
    print(f"{'ACO':<15} {aco_div:<10.2f} {aco_pref:<12.2f} {aco_time:.2f}")
    print(f"{'Random':<15} {rand_div:<10.2f} {rand_pref:<12.2f} {rand_time:.2f}")
    print(f"{'Self-selection':<15} {self_div:<10.2f} {self_pref:<12.2f} {self_time:.2f}")

if __name__ == '__main__':
    students = load_student_data(
        students_csv='Module 4 Research Overview 2023.csv',
        prefs_csv='Module 4 Research 2023 Student Preferences.csv'
    )
    run_all_methods(students)
