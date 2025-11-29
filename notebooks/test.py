import pandas as pd
df = pd.read_csv("data/clean_tickets.csv")
print(df['Ticket Type'].value_counts())
