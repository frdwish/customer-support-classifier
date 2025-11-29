import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Title
st.title("📊 Exploratory Data Analysis")

# Import dataset
df = pd.read_csv("data/clean_tickets.csv", encoding="ISO-8859-1")

# Dataset preview
st.subheader("Dataset Preview")
st.write(df.head())

# Ticket Type distribution
st.subheader("Ticket Type Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(y=df["Ticket Type"], order=df["Ticket Type"].value_counts().index, ax=ax)
st.pyplot(fig)

# WordCloud of Ticket Descriptions
st.subheader("WordCloud of Ticket Descriptions")
text = " ".join(df["Ticket Description"].astype(str))
wordcloud = WordCloud(background_color="black").generate(text)

fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation="bilinear")
ax_wc.axis("off")
st.pyplot(fig_wc)
