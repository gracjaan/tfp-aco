from aco import AntColonyOptimizer
from data_loader import load_student_data
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')

if __name__ == '__main__':
    # Load data
    students = load_student_data(
        students_csv='Module 4 Research Overview 2023.csv',
        prefs_csv='Module 4 Research 2023 Student Preferences.csv'
    )

    # Initialize optimizer
    aco = AntColonyOptimizer(
        students,
        num_teams=42,
        team_size=6,
        max_iter=50,
        num_ants=20,
        alpha=1.0,
        beta=3.0,
        rho=0.3,
        preference_weight=0.5
    )

    # Run optimization
    logging.info("Starting ACO optimization...")
    start_time = time.time()
    best_solution, best_fitness = aco.run()
    duration = time.time() - start_time

    # Report
    logging.info("\nBest fitness: %.4f" % best_fitness)
    logging.info("Time elapsed: %.2f seconds" % duration)

    # Output teams
    for idx, team in enumerate(best_solution):
        logging.info(f"Team {idx+1}: {[s['student_id'] for s in team]}")
