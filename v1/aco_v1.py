import random
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

class Student:
    def __init__(self, sid: int, prefs: List[int], belbin: str, nationality: str):
        self.id = sid
        self.prefs = prefs # list of project indices ordered by preference (length = number of projects)
        self.belbin = belbin
        self.nationality = nationality

    def satisfaction(self, project: int) -> float:
        """returns satisfaction score on 0<>1 scale (1 for first choice, 0.5 for mid, 0 for last)"""
        rank = self.prefs.index(project)
        return (len(self.prefs) - 1 - rank) / (len(self.prefs) - 1)

class Team:
    def __init__(self, tid: int, project: int, capacity: int = 6):
        self.id = tid
        self.project = project
        self.capacity = capacity
        self.students: List[Student] = []

    # ----- constraint checks -----
    def can_add(self, student: Student, max_same_nat: int = 2, unlimited_nat: str = "Dutch") -> bool:
        if len(self.students) >= self.capacity:
            return False
        if student.nationality == unlimited_nat:
            return True
        n_same = sum(1 for s in self.students if s.nationality == student.nationality)
        return n_same < max_same_nat

    def add(self, student: Student):
        self.students.append(student)

    # ----- scoring -----
    def _satisfaction_score(self) -> float:
        if not self.students:
            return 0.0
        return sum(s.satisfaction(self.project) for s in self.students) / len(self.students)

    def _belbin_diversity(self) -> float:
        if not self.students:
            return 0.0
        unique_roles = {s.belbin for s in self.students}
        return len(unique_roles) / len(self.students)

    def fitness(self) -> float:
        # basic objective function
        return 0.5 * self._satisfaction_score() + 0.5 * self._belbin_diversity()

    def __len__(self):
        return len(self.students)

    def __repr__(self):
        sids = [s.id for s in self.students]
        return f"Team{self.id}(proj={self.project}, students={sids}, fitness={self.fitness():.3f})"

class ACO:
    def __init__(
        self,
        students: List[Student],
        n_projects: int = 6,
        teams_per_project: int = 7,
        team_size: int = 6,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        n_ants: int = 30,
        n_iter: int = 200,
        q: float = 1.0,
        seed: int = 42
    ):
        random.seed(seed)
        self.students = students
        self.n_projects = n_projects
        self.teams_per_project = teams_per_project
        self.team_size = team_size
        self.n_teams = n_projects * teams_per_project
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.q = q
        # pheromone matrix tau[i][j] where i student index, j team index
        self.tau = [[1.0 for _ in range(self.n_teams)] for _ in range(len(students))]
        # precreate team objects (template); each ant will clone these
        self.base_teams = [
            Team(tid=j, project=j // teams_per_project, capacity=team_size)
            for j in range(self.n_teams)
        ]

    def heuristic(self, student: Student, team: Team) -> float:
        """returns eta value for assigning student to team (>= small positive)"""
        sat = student.satisfaction(team.project)
        roles = {s.belbin for s in team.students}
        roles.add(student.belbin)
        belbin = len(roles) / self.team_size
        predicted = 0.5 * sat + 0.5 * belbin
        return predicted + 1e-6

    def construct_solution(self):
        teams = [Team(tid=t.id, project=t.project, capacity=self.team_size) for t in self.base_teams]
        unassigned = self.students.copy()
        random.shuffle(unassigned)
        for s in unassigned:
            feasible = [t for t in teams if t.can_add(s)]
            if not feasible:
                feasible = [t for t in teams if len(t) < t.capacity]
            probabilities = []
            for t in feasible:
                tau_ij = self.tau[s.id][t.id] ** self.alpha
                eta_ij = self.heuristic(s, t) ** self.beta
                probabilities.append(tau_ij * eta_ij)
            total = sum(probabilities)
            if total == 0:
                chosen_team = random.choice(feasible)
            else:
                r = random.random() * total
                cumulative = 0
                for t, p in zip(feasible, probabilities):
                    cumulative += p
                    if r <= cumulative:
                        chosen_team = t
                        break
            chosen_team.add(s)
        fitness = sum(t.fitness() for t in teams) / self.n_teams
        return teams, fitness

    def update_pheromones(self, ants_solutions):
        for i in range(len(self.students)):
            for j in range(self.n_teams):
                self.tau[i][j] *= (1 - self.rho)
                if self.tau[i][j] < 1e-6:
                    self.tau[i][j] = 1e-6
        best_teams, best_fit = max(ants_solutions, key=lambda x: x[1])
        for t in best_teams:
            for s in t.students:
                self.tau[s.id][t.id] += self.q * best_fit

    # ---------- main ----------
    def run(self):
        global_best_fit = -math.inf
        global_best_teams = None
        for it in range(self.n_iter):
            ants_solutions = [self.construct_solution() for _ in range(self.n_ants)]
            self.update_pheromones(ants_solutions)
            iter_best_fit = max(f for _, f in ants_solutions)
            if iter_best_fit > global_best_fit:
                global_best_fit = iter_best_fit
                global_best_teams = max(ants_solutions, key=lambda x: x[1])[0]
            if (it + 1) % 10 == 0:
                print(f"[iter {it+1}] bvest fitness so far: {global_best_fit:.4f}")
        return global_best_teams, global_best_fit

def generate_sample_data(
    n_students: int = 250,
    n_projects: int = 6,
    belbin_roles: List[str] = None,
    nationalities: List[str] = None,
    pref_noise: float = 0.2,
    seed: int = 1,
):
    random.seed(seed)
    belbin_roles = belbin_roles or [
        "Plant",
        "Resource Investigator",
        "Co-ordinator",
        "Shaper",
        "Monitor Evaluator",
        "Teamworker",
        "Implementer",
        "Completer Finisher",
        "Specialist",
    ]
    nationalities = nationalities or [
        "Dutch",
        "German",
        "Italian",
        "Spanish",
        "Polish",
        "French",
        "Chinese",
        "Indian",
    ]
    students = []
    for sid in range(n_students):
        prefs = list(range(n_projects))
        random.shuffle(prefs)
        if random.random() < pref_noise:
            prefs.insert(0, random.choice(prefs))
        belbin = random.choice(belbin_roles)
        nationality = random.choice(nationalities)
        students.append(Student(sid, prefs, belbin, nationality))
    return students

if __name__ == "__main__":
    students = generate_sample_data()
    aco = ACO(students, n_projects=6, teams_per_project=7)
    best_teams, best_fit = aco.run()
    print(f"Global best fitness: {best_fit:.4f}")
    for t in best_teams:
        print(t)
