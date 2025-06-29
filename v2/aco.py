import random
import logging
from collections import defaultdict

class AntColonyOptimizer:
    def __init__(self, students, num_teams, team_size, max_iter, num_ants,
                 alpha, beta, rho, preference_weight):
        self.students = students
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_iter = max_iter
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.preference_weight = preference_weight

        self.pheromones = defaultdict(lambda: 1.0)
        self.project_list = list(set(p for s in students for p in s['preferences']))

    def run(self):
        best_solution = None
        best_fitness = float('-inf')

        for it in range(self.max_iter):
            all_solutions = []
            for _ in range(self.num_ants):
                solution = self.construct_solution()
                fitness = self.evaluate(solution)
                all_solutions.append((solution, fitness))

                if fitness > best_fitness:
                    best_solution = solution
                    best_fitness = fitness

            self.evaporate_pheromones()
            self.update_pheromones(all_solutions)
            logging.info(f"Iteration {it+1}: Best fitness = {best_fitness:.4f}")

        return best_solution, best_fitness

    def construct_solution(self):
        students = self.students.copy()
        random.shuffle(students)
        teams = [[] for _ in range(self.num_teams)]

        for student in students:
            feasible_teams = [t for t in teams if len(t) < self.team_size]
            scores = []
            for team in feasible_teams:
                idx = teams.index(team)
                score = self.pheromones[(student['student_id'], idx)] ** self.alpha
                score *= self.heuristic(student, team) ** self.beta
                scores.append(score)

            if not scores:
                continue

            total = sum(scores)
            probs = [s / total for s in scores]
            chosen_team = random.choices(feasible_teams, weights=probs, k=1)[0]
            chosen_idx = teams.index(chosen_team)
            teams[chosen_idx].append(student)

        return teams

    def heuristic(self, student, team):
        if not team:
            return 1.0
        roles = set(s['belbin_role'] for s in team)
        div_score = 1.0 if student['belbin_role'] not in roles else 0.5

        prefs = student['preferences']
        team_projects = [p for s in team for p in s['preferences'][:1]]
        pref_score = sum((5 - prefs.index(p)) for p in team_projects if p in prefs) / (5 * len(team_projects) + 1e-5)

        return self.preference_weight * pref_score + (1 - self.preference_weight) * div_score

    def evaluate(self, teams):
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
        return self.preference_weight * avg_pref + (1 - self.preference_weight) * avg_div

    def evaporate_pheromones(self):
        for key in self.pheromones:
            self.pheromones[key] *= (1 - self.rho)

    def update_pheromones(self, solutions):
        for teams, fitness in solutions:
            for idx, team in enumerate(teams):
                for student in team:
                    self.pheromones[(student['student_id'], idx)] += fitness
