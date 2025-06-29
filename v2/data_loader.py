import pandas as pd

def load_student_data(students_csv, prefs_csv):
    students_df = pd.read_csv(students_csv, header=[0, 1])
    students_df.columns = [' '.join([str(a), str(b)]).strip() for a, b in students_df.columns]
    students_df = students_df.rename(columns={
        "STUDENT INFO S-Number": "student_id",
        "Unnamed: 7_level_0 Nationality": "nationality",
        "Unnamed: 42_level_0 Dom 1": "belbin_role"
    })

    prefs_df = pd.read_csv(prefs_csv)
    students_df["student_id"] = students_df["student_id"].astype(str).str.strip()
    prefs_df["Student"] = prefs_df["Student"].astype(str).str.strip()

    merged = pd.merge(prefs_df, students_df, left_on="Student", right_on="student_id", how="inner")
    merged = merged[[
        "student_id", "Option 1", "Option 2", "Option 3", "Option 4", "Option 5", "belbin_role", "nationality"
    ]].dropna()

    students = []
    for _, row in merged.iterrows():
        students.append({
            "student_id": row["student_id"],
            "preferences": [row[f"Option {i+1}"] for i in range(5)],
            "belbin_role": row["belbin_role"],
            "nationality": row["nationality"]
        })

    return students
