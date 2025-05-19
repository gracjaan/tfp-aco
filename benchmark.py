import random
import statistics
import time
from aco_v1 import Student, Team, ACO, generate_sample_data
from collections import Counter, defaultdict

def random_assignment(students, n_projects=6, teams_per_project=7, team_size=6):
    """randomly assign students to teams while respecting constraints."""
    teams = [Team(tid=j, project=j // teams_per_project, capacity=team_size) 
             for j in range(n_projects * teams_per_project)]
    
    unassigned = students.copy()
    random.shuffle(unassigned)
    
    for student in unassigned:
        feasible = [t for t in teams if t.can_add(student)]
        if not feasible:
            feasible = [t for t in teams if len(t) < t.capacity]
        
        if feasible:
            chosen_team = random.choice(feasible)
            chosen_team.add(student)
    
    fitness = sum(t.fitness() for t in teams) / len(teams)
    return teams, fitness

def run_benchmark(n_runs=10, n_students=250, n_projects=6, teams_per_project=7, 
                 team_size=6, aco_iterations=100, aco_ants=30, verbose=True):
    """run multiple trials comparing ACO vs random assignment."""
    random_results = []
    aco_results = []
    random_times = []
    aco_times = []
    random_team_metrics = []
    aco_team_metrics = []
    unassigned_counts_random = []
    unassigned_counts_aco = []
    
    if verbose:
        print(f"Running {n_runs} benchmarks...")
        print(f"{'Run':^5}|{'Random Fitness':^15}|{'ACO Fitness':^15}|{'Improvement':^15}|{'Random Time':^15}|{'ACO Time':^15}")
        print("-" * 85)
    
    for i in range(n_runs):
        seed = i + 1
        students = generate_sample_data(n_students=n_students, n_projects=n_projects, seed=seed)
        
        start_time = time.time()
        random_teams, random_fitness = random_assignment(
            students, n_projects, teams_per_project, team_size)
        random_time = time.time() - start_time
        random_results.append(random_fitness)
        random_times.append(random_time)
        
        assigned_students_random = sum(len(t.students) for t in random_teams)
        unassigned_random = n_students - assigned_students_random
        unassigned_counts_random.append(unassigned_random)
        
        random_team_metrics.append(analyze_team_metrics(random_teams))
        
        start_time = time.time()
        aco = ACO(students, n_projects=n_projects, teams_per_project=teams_per_project, 
                  team_size=team_size, n_iter=aco_iterations, n_ants=aco_ants, seed=seed)
        aco_teams, aco_fitness = aco.run()
        aco_time = time.time() - start_time
        aco_results.append(aco_fitness)
        aco_times.append(aco_time)
        
        assigned_students_aco = sum(len(t.students) for t in aco_teams)
        unassigned_aco = n_students - assigned_students_aco
        unassigned_counts_aco.append(unassigned_aco)
        
        aco_team_metrics.append(analyze_team_metrics(aco_teams))
        
        improvement = ((aco_fitness - random_fitness) / random_fitness) * 100 if random_fitness > 0 else float('inf')
        
        if verbose:
            print(f"{i+1:^5}|{random_fitness:^15.4f}|{aco_fitness:^15.4f}|{improvement:^15.2f}%|{random_time:^15.2f}s|{aco_time:^15.2f}s")
    
    avg_random = statistics.mean(random_results)
    avg_aco = statistics.mean(aco_results)
    avg_improvement = ((avg_aco - avg_random) / avg_random) * 100 if avg_random > 0 else float('inf')
    
    avg_metrics_random = {
        k: statistics.mean([m[k] for m in random_team_metrics]) 
        for k in random_team_metrics[0]
    }
    avg_metrics_aco = {
        k: statistics.mean([m[k] for m in aco_team_metrics]) 
        for k in aco_team_metrics[0]
    }
    
    if verbose:
        print("\nSummary Statistics:")
        print(f"Average Random Fitness: {avg_random:.4f}")
        print(f"Average ACO Fitness:    {avg_aco:.4f}")
        print(f"Average Improvement:    {avg_improvement:.2f}%")
        print(f"Average Random Time:    {statistics.mean(random_times):.2f}s")
        print(f"Average ACO Time:       {statistics.mean(aco_times):.2f}s")
        
        print("\nTeam Metrics Comparison:")
        print(f"{'Metric':<25}|{'Random':^15}|{'ACO':^15}|{'Improvement':^15}")
        print("-" * 73)
        
        for k in avg_metrics_random:
            rand_val = avg_metrics_random[k]
            aco_val = avg_metrics_aco[k]
            improv = ((aco_val - rand_val) / rand_val) * 100 if rand_val > 0 else float('inf')
            print(f"{k:<25}|{rand_val:^15.4f}|{aco_val:^15.4f}|{improv:^15.2f}%")
        
        avg_unassigned_random = statistics.mean(unassigned_counts_random)
        avg_unassigned_aco = statistics.mean(unassigned_counts_aco)
        print(f"\nAvg Unassigned Students (Random): {avg_unassigned_random:.2f}")
        print(f"Avg Unassigned Students (ACO):    {avg_unassigned_aco:.2f}")
    
    return {
        "random_fitness": random_results,
        "aco_fitness": aco_results,
        "random_times": random_times,
        "aco_times": aco_times,
        "unassigned_random": unassigned_counts_random,
        "unassigned_aco": unassigned_counts_aco,
        "avg_metrics_random": avg_metrics_random,
        "avg_metrics_aco": avg_metrics_aco,
    }

def analyze_team_metrics(teams):
    """analyze detailed metrics about teams."""
    team_sizes = [len(t) for t in teams]
    satisfaction_scores = [t._satisfaction_score() for t in teams]
    belbin_scores = [t._belbin_diversity() for t in teams]
    
    project_counts = Counter([t.project for t in teams])
    students_per_project = defaultdict(int)
    for team in teams:
        students_per_project[team.project] += len(team.students)
    
    project_balance = 0
    if students_per_project:
        avg_students = sum(students_per_project.values()) / len(students_per_project)
        project_balance = 1 - (statistics.pstdev(students_per_project.values()) / avg_students if avg_students > 0 else 0)
    
    return {
        "avg_team_size": statistics.mean(team_sizes) if team_sizes else 0,
        "min_team_size": min(team_sizes) if team_sizes else 0,
        "max_team_size": max(team_sizes) if team_sizes else 0,
        "team_size_std": statistics.pstdev(team_sizes) if len(team_sizes) > 1 else 0,
        "avg_satisfaction": statistics.mean(satisfaction_scores) if satisfaction_scores else 0,
        "avg_belbin_diversity": statistics.mean(belbin_scores) if belbin_scores else 0,
        "project_balance": project_balance,
    }

def detailed_single_run_comparison(n_students=250, n_projects=6, teams_per_project=7, 
                                 team_size=6, aco_iterations=100, aco_ants=30, seed=42):
    """Run a detailed single comparison with more metrics and analysis."""
    students = generate_sample_data(n_students=n_students, n_projects=n_projects, seed=seed)
    
    print("Running random assignment...")
    random_teams, random_fitness = random_assignment(
        students, n_projects, teams_per_project, team_size)
    
    print(f"Running ACO algorithm (iterations={aco_iterations}, ants={aco_ants})...")
    aco = ACO(students, n_projects=n_projects, teams_per_project=teams_per_project, 
              team_size=team_size, n_iter=aco_iterations, n_ants=aco_ants, seed=seed)
    aco_teams, aco_fitness = aco.run()
    
    random_metrics = analyze_team_metrics(random_teams)
    aco_metrics = analyze_team_metrics(aco_teams)
    
    assigned_random = sum(len(t.students) for t in random_teams)
    assigned_aco = sum(len(t.students) for t in aco_teams)
    
    print("\nDetailed Comparison:")
    print(f"{'Metric':<25}|{'Random':^15}|{'ACO':^15}|{'Difference':^15}")
    print("-" * 73)
    print(f"{'Overall Fitness':<25}|{random_fitness:^15.4f}|{aco_fitness:^15.4f}|{aco_fitness-random_fitness:^15.4f}")
    print(f"{'Assigned Students':<25}|{assigned_random:^15}|{assigned_aco:^15}|{assigned_aco-assigned_random:^15}")
    print(f"{'Unassigned Students':<25}|{n_students-assigned_random:^15}|{n_students-assigned_aco:^15}|{(assigned_aco-assigned_random):^15}")
    
    for k in random_metrics:
        print(f"{k:<25}|{random_metrics[k]:^15.4f}|{aco_metrics[k]:^15.4f}|{aco_metrics[k]-random_metrics[k]:^15.4f}")
    
    return {
        "random_teams": random_teams,
        "aco_teams": aco_teams,
        "random_fitness": random_fitness,
        "aco_fitness": aco_fitness,
        "random_metrics": random_metrics,
        "aco_metrics": aco_metrics,
    }

if __name__ == "__main__":
    results = run_benchmark(
        n_runs=5,             # Number of benchmark runs
        n_students=250,       # Number of students
        n_projects=6,         # Number of projects
        teams_per_project=7,  # Teams per project
        team_size=6,          # Maximum team size
        aco_iterations=100,   # Number of ACO iterations
        aco_ants=20           # Number of ants per iteration
    )
    
    print("\n\nRunning detailed single comparison...")
    detailed_results = detailed_single_run_comparison(
        n_students=250,
        n_projects=6,
        teams_per_project=7,
        team_size=6,
        aco_iterations=100,
        aco_ants=20,
        seed=42
    ) 