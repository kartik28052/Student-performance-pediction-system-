"""Step 1: Create a sample student dataset (students.csv).

Real projects use real school data. Since you may not have any yet, this
script makes realistic fake data so you can build and demo the system.
Later, replace students.csv with your real data (same column names).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

attendance = np.random.uniform(40, 100, N).round(1)        # % of classes attended
study_hours = np.random.uniform(0, 8, N).round(1)          # hours studied per day
prev_marks = np.random.uniform(30, 100, N).round(1)        # previous exam marks (%)
assignments = np.random.uniform(0, 100, N).round(1)        # assignment completion (%)
sleep_hours = np.random.uniform(4, 9, N).round(1)          # hours of sleep per night

# Final marks depend on the above factors (plus some random noise)
final_marks = (
    0.30 * prev_marks
    + 0.25 * attendance
    + 0.15 * assignments
    + 4.0 * study_hours
    + 1.0 * sleep_hours
    + np.random.normal(0, 5, N)
)
final_marks = np.clip(final_marks, 0, 100).round(1)


def to_category(m):
    if m >= 75:
        return "High"
    if m >= 55:
        return "Medium"
    return "Low"


df = pd.DataFrame(
    {
        "attendance": attendance,
        "study_hours": study_hours,
        "prev_marks": prev_marks,
        "assignments": assignments,
        "sleep_hours": sleep_hours,
        "final_marks": final_marks,
    }
)
df["performance"] = df["final_marks"].apply(to_category)
df.to_csv("students.csv", index=False)
print("Created students.csv with", len(df), "students")
print(df["performance"].value_counts())