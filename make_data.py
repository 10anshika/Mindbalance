import pandas as pd
import numpy as np

np.random.seed(2024)
n = 1100

data = {
    'anxiety_level':          np.random.randint(0, 21, n),
    'self_esteem':            np.random.randint(0, 31, n),
    'mental_health_history':  np.random.randint(0, 2, n),
    'depression':             np.random.randint(0, 28, n),
    'headache':               np.random.randint(0, 6, n),
    'blood_pressure':         np.random.randint(1, 4, n),
    'sleep_quality':          np.random.randint(0, 6, n),
    'breathing_problem':      np.random.randint(0, 6, n),
    'noise_level':            np.random.randint(0, 6, n),
    'living_conditions':      np.random.randint(0, 6, n),
    'safety':                 np.random.randint(0, 6, n),
    'basic_needs':            np.random.randint(0, 6, n),
    'academic_performance':   np.random.randint(0, 6, n),
    'study_load':             np.random.randint(0, 6, n),
    'teacher_student_rel':    np.random.randint(0, 6, n),
    'future_career_concerns': np.random.randint(0, 6, n),
    'social_support':         np.random.randint(0, 4, n),
    'peer_pressure':          np.random.randint(0, 6, n),
    'extracurricular_act':    np.random.randint(0, 6, n),
    'bullying':               np.random.randint(0, 6, n),
    'yoga_days_per_week':     np.random.randint(0, 8, n),
    'yoga_duration_mins':     np.where(np.random.randint(0,8,n)>0, np.random.randint(15,75,n), 0),
    'mindfulness_mins_day':   np.random.randint(0, 61, n),
    'pranayama_practice':     np.random.randint(0, 2, n),
}

df = pd.DataFrame(data)

def label(row):
    neg = (row['anxiety_level']/20 + row['depression']/27 +
           row['study_load']/5 + row['peer_pressure']/5 +
           row['blood_pressure']/3 + row['headache']/5 +
           (5-row['sleep_quality'])/5 + row['breathing_problem']/5 +
           row['noise_level']/5 + row['bullying']/5)
    pos = (row['self_esteem']/30 + row['social_support']/3 +
           row['living_conditions']/5 + row['safety']/5 +
           row['basic_needs']/5 + row['academic_performance']/5 +
           row['teacher_student_rel']/5 +
           row['yoga_days_per_week']/7*1.5 +
           row['mindfulness_mins_day']/60 +
           row['pranayama_practice'])
    score = neg - pos * 0.7 + np.random.normal(0, 0.4)
    if score < 2.2:   return 0
    elif score < 3.8: return 1
    else:             return 2

df['stress_level'] = df.apply(label, axis=1)
df.to_csv('/home/claude/mindbalance_v2/data/StressLevelDataset.csv', index=False)
print("Shape:", df.shape)
print(df['stress_level'].value_counts().sort_index())
