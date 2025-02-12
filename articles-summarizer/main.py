import tkinter as tk
from tkinter.constants import DISABLED

import nltk
from textblob import TextBlob
from newspaper import Article

def summarize():

    url = uText.get('1.0', tk.END)

    article = Article(url)

    article.download()
    article.parse()

    article.nlp()

    title.config(state=tk.NORMAL)
    author.config(state=tk.NORMAL)
    publication.config(state=tk.NORMAL)
    summary.config(state=tk.NORMAL)
    sentiment.config(state=tk.NORMAL)

    title.delete('1.0', tk.END)
    title.insert('1.0', article.title)

    author.delete('1.0', tk.END)
    author.insert('1.0', article.authors)

    publication.delete('1.0', tk.END)
    publication.insert('1.0', article.publish_date)

    summary.delete('1.0', tk.END)
    summary.insert('1.0', article.summary)

    analysis = TextBlob(article.text)
    sentiment.delete('1.0', tk.END)
    sentiment.insert('1.0',
                     f'Polarity: {analysis.sentiment.polarity}, Sentiment: {analysis.sentiment.subjectivity} {"positive" if analysis.polarity > 0 else "negative" if analysis.polarity < 0 else "neutral"}')

    title.config(state="disabled")
    author.config(state="disabled")
    publication.config(state="disabled")
    summary.config(state="disabled")
    sentiment.config(state="disabled")

    print(f'Title: {article.title}')
    print(f'Authors: {article.authors}')
    print(f'Publication Date: {article.publish_date}')
    print(f'Summary: {article.summary}')

root = tk.Tk()
root.title("News Summarizer")
root.geometry('1200x600')

tLabel = tk.Label(root, text="Title")
tLabel.pack()

title = tk.Text(root, height=1, width=140)
title.config(state='disabled', bg='#dddddd')
title.pack()

aLabel = tk.Label(root, text="Author")
aLabel.pack()

author = tk.Text(root, height=1, width=140)
author.config(state='disabled', bg='#dddddd')
author.pack()

pLabel = tk.Label(root, text="Publishing Date")
pLabel.pack()

publication = tk.Text(root, height=1, width=140)
publication.config(state='disabled', bg='#dddddd')
publication.pack()

sLabel = tk.Label(root, text="Summary")
sLabel.pack()

summary = tk.Text(root, height=20, width=140)
summary.config(state='disabled', bg='#dddddd')
summary.pack()

seLabel = tk.Label(root, text="Sentiment Analysis")
seLabel.pack()

sentiment = tk.Text(root, height=1, width=140)
sentiment.config(state="disabled", bg='#dddddd')
sentiment.pack()

uLabel = tk.Label(root, text="URL")
uLabel.pack()

uText = tk.Text(root, height=1, width=140)
uText.pack()

btn = tk.Button(root, text="Summarize", command=summarize)
btn.pack()

root.mainloop()
