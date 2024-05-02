# Sentiment Analysis of Bible translations

This repo contains the source code for our analysis of 5 bible translations: American Standard Version (ASV), Free Bible Version (FBV), King James Version (KJV), World English Bible (WEB), World Messianic Bible (WMB).

### Overview

1. [Data](#data)
   1. In [api_pipeline.ipynb](data/bibles/api_pipeline.ipynb) we collect bible data from [API.Bible](https://scripture.api.bible/)
   2. In [chunk_verses.ipynb](data/bibles_chunked/chunk_verses.ipynb) we chunk the verses in groups of 3
   3. In [get_random_verses.ipynb](data/random_veres/get_random_verses.ipynb) we get random verses from each bible (not used. Chunks instesad.)
   4. In [get_random_chunks.ipynb](data/random_chunks/get_random_chunks.ipynb) we get random chunks of verses from each bible to annotate (us and the model)
2. [Manual Annotation](#manual-annotation)
   1. In [manual_annotation](manual_annotation/anno_instructions.md) we manually annotate the random chunks
   2. [anno.py](manual_annotation/anno.py) is a script to help with manual annotation
   3. In [agreement.ipynb](annotation_analysis/agreement.ipynb) we compile labels from each annotator and calculate agreement with various metrics
3. [Model Annotation](#model-annotation)
   1. In [bible.ipynb](model_annotation/bible.ipynb) we run our model on the bible data
4. [Model Accuracy](#model-accuracy)
   1. In [accuracy.ipynb](annotation_analysis/accuracy.ipynb) we compare the model's sentiment to the manual annotations and plot
5. [Translation Comparisons](#translation-comparisons)
   1. In [bible_comparison.ipynb](annotation_analysis/bible_comparison.ipynb) we compare the sentiment distribution across translations
6. [Sentiment by Character](#sentiment-by-character)
   1. In [characters.ipynb](char_sent_analysis/characters.ipynb) we analyze the sentiment distribution by character
7. [Low Frequency Token Analysis](#low-frequency-token-analysis)
   1. In [Low_freq.ipynb](Low_Freq_analysis/Low_freq.ipynb) we analyze the model's accuracy with the least frequent tokens

### Link to data:

https://drive.google.com/drive/folders/1fpad5m010h4UXQhI0J7hEprQimdGIeXy?usp=sharing

# Where to put data from Google Drive

- All folders with student names (cameron, gerardo, nick, olivia, river) go into the `manual_annotation` folder
- All csv's in the low_freq_analysis folder on Drive should go into the `Low_Freq_analysis` folder in the codebase
- `data` folder on Drive should just be placed in the root folder of the codebase, at the same level as this README.md
- `*_out.csv` files in the Drive should all go into `model_annotation`
- `annotator_labels.csv` and `translation_agreement.csv` should go into `annotation_analysis` folder
- `char_analysis_confidence.csv` and `top10conflict_confidence.csv` go into `char_sent_analysis` folder

# Setup

Note: you must have a Python environment and a way to run .ipynb files.

1. Clone the repo and cd into it
2. ```
   pip install -r requirements.txt
   ```

# Data

### Collecting the data

Bible data is easy to come by, however we needed our data to be uniform across translations, which is not as easy. Our requirements led us to use https://scripture.api.bible/.

API.Bible is an api that provides access to ~2500 bible translations in a uniform format.\
To collect the data we did the following:

1. Make an API.Bible account to get a key
2. Run [api_pipeline.ipynb](data/bibles/api_pipeline.ipynb) in `data/bibles/`

Here is a brief overview of the function that does the heavy lifting. The actual code is more robust with exceptions for failed requests so only actual verses are written to the csv, this is simplified for readability.

> [data/bibles/api_pipeline.ipynb](data/bibles/api_pipeline.ipynb)

```python
def pipeline(bibleID, bibleName):

    # the most verses you can get at once is all verses of a chapter
    # so here we get the chapter IDs for each chapter in each book (excluding intros)
    # we use these IDs to request the verses for each chapter
    url = f"{baseURL}/bibles/{bibleID}/books"
    response = requests.get(url, headers=headers, params={"include-chapters": True}).json()
    chapIDs = jmespath.search("data[].chapters[?number!='intro'].id[]", response)

    # make a csv
    with open(f"{bibleName}.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["citation", "verse"])
        writer.writeheader()

        # get verses for each chapter
        for chapID in chapIDs:

            # request chapter info
            url = f"{baseURL}/bibles/{bibleID}/chapters/{chapID}"
            response = requests.get(url, headers=headers, params={"content-type": "text"}).json()
            print(response)

            # str of all verses in the chapter
            verses = response["data"]["content"]

            # split into individual verses
            verseList = re.split(r"\s*\[\d+\]\s*", verses.strip())
            verseList = [v.strip() for v in verseList if v]

            # write a line to the csv for each verse
            for i, verse in enumerate(verseList):
                writer.writerow({"citation": f"{response['data']['id']}.{i+1}", "verse": verse})
                print(f"wrote: {verse}")
```

### Chunking the Data

For reasons we will discuss in the following section, it was benificial to chunk the verses into groups of 3.

> [data/bibles_chunked/chunk_verses.ipynb](data/bibles_chunked/chunk_verses.ipynb)

```python
 def chunker(df: DataFrame) -> DataFrame:
     new_df = (
         # group by every 3 verses
         df.groupby(df.index // 3)
         # citiation: first cit. in group
         # verse: join all verses
         .agg({"citation": "first", "verse": " ".join})
         # rename cols
         .rename(columns={"citation": "start_citation", "verse": "text"})
     )
     # rename index
     new_df.index.name = "chunk"
     return new_df
```

# Manual Annotation

In order to evaluate our model's accuracy we need labeled data. Because we don't have access to sentiment labels for all of these bibles, we needed to do our own annotation.

Initially we were going to annotate by verse, but we learned that bible verses are quite short (many just a sentence fragment), making them challenging to annotate alone.\
Because of this we decided to annotate in chunks of 3 verses.

We chose to anotate

- 10 chunks of ASV, FBV, WEB, & WMB
- 50 chunks of KJV

We chose to do 50 of KJV instead of 10 like the others because we were particularly worried about the model's accuracy on the old style of english KJV uses. Looking back it would've been better to do 50 for all of the bibles; 10 was too few.

## Getting the random chunks

In `data/random_chunks/`\
Get random chunks from each bible

> [data/random_chunks/get_random_chunks.ipynb](data/random_chunks/get_random_chunks.ipynb)

```python
 # make csv with 10 random chunks for each modern bible
 for bible in ["asv.csv", "fbv.csv", "web.csv", "wmb.csv"]:
     df = pd.read_csv(f"../bibles_chunked/{bible}")
     df.sample(10).to_csv(f"{bible}", index=False)

 # 50 chunks for the KJV
 df = pd.read_csv("../bibles_chunked/kjv.csv")
 df.sample(50).to_csv("kjv.csv", index=False)
```

## Annotating the chunks

in `manual_annotation/`

We each added a 'sentiment' column to each random chunk, either by manually typing the numbers or using the [anno.py](manual_annotation/anno.py) script.

Each of us followed the instructions in

> [manual_annotation/anno_instructions.md](manual_annotation/anno_instructions.md)

> ## 1. Copy the random verses
>
> Make a copy of each csv file in `data/random_chunks` and put it in a folder named with your name like
>
> ```
> manual_annotation
> └── John
>     ├── asv.csv
>     ├── fbv.csv
>     ├── kjv.csv
>     ├── web.csv
>     └── wmb.csv
> ```
>
> ## 2. Annotate
>
> ### Option A - Use Olivia's script
>
> 1. Copy `./anno.py` into your folder with your csv files.
> 2. Go to the main function and change it to your name
> 3. Run the file
>
> ### Option B - Do it manually
>
> To each file, add a `sentiment` label to each row like this
>
> ```
> chunk,start_citation,text,sentiment
> 2556,1SA.17.50,"So David prevailed...", 1
> ```
>
> CLASSIFICATIONS\
> **1**: Neutral\
> **2**: Positive\
> **3**: Negative

## Annotator agreement

in `annotation_analysis/`

The annotated chunks are compiled into one file [annotator_labels.csv](annotation_analysis/annotator_labels.csv) using the script [agreement.ipynb](annotation_analysis/agreement.ipynb)\
[annotator_labels.csv](annotation_analysis/annotator_labels.csv) looks like this

    bible,chunk,olivia,river,gerardo,nick,cameron
    asv,2556,3,2,1,3,3
    asv,10127,3,3,3,3,3
    asv,9900,2,2,1,2,2

Then we calculate Fleiss' Kappa and a few other numbers and plot them.
Methodology is well documented there in [agreement.ipynb](annotation_analysis/agreement.ipynb)

# Model Annotation

In `model_annotation/`

In [bible.ipynb](model_annotation/bible.ipynb) we run our model on our bible data.

Moving through each chunk (row of a particular translation's csv file), we parse the model's output to grab the sentiment score with the highest relative confidence score and the confidence score itself.

> [bible.ipynb](model_annotation/bible.ipynb)

```python
classifier = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    return_all_scores=False,
)

def process(row):
    '''parses model output to fill in sentiment/confidence scores for each row'''

    # run model
    text = row["text"]
    result = classifier(text)

    # parse result
    sentiment = result[0]["label"]
    confidence = result[0]["score"]
    sentiment_value = label_to_sentiment(sentiment) # num to pos neg neu

    return {
        "chunk": row["chunk"],
        "start_citation": row["start_citation"],
        "text": text,
        "sentiment": sentiment_value,
        "confidence": confidence,
    }

def annotate(infile):
    # read the verses
    with open(infile, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        # open file to write to
        with open(f"{infile[:-4]}_out.csv", mode='w', newline='', encoding='utf-8') as out_csv:
            fields = ['chunk', 'start_citation', 'text', 'sentiment', 'confidence']
            writer = csv.DictWriter(out_csv, fieldnames=fields)
            writer.writeheader()
            # run model on each verse and write to file
            for row in reader:
                writer.writerow(process(row))
```

Since we wanted our model annotations to be in the same format as our manual annotations, we generated similar columns in our output csvs.

```
Model output:
    [[{'label': 'negative', 'score': 0.8345967531204224},
    {'label': 'neutral', 'score': 0.1521468460559845},
    {'label': 'positive', 'score': 0.013256409205496311}]]

Sample annotated csv:
    chunk,start_citation,text,sentiment,confidence
    0,GEN.1.1,"In the beginning God created...",1,0.5875061750411987
    1,GEN.1.4,"And God saw the light, that it...",2,0.6193588972091675
```

Ultimately, this ran fine, but we did run into a few runtime errors related to the length of text being input into our classifier. Tensorflow has a maximum input tensor size, and we found that a few lines in the WEB translation were causing issues. To remedy this, we implemented a try-except clause to handle these exceptions, annotating these lines with a sentiment score of "0" to filter them out from the rest of our data. Since we only encountered this problem with <10 chunks, we considered it to have a negligable impact on our overall analysis.

# Model Accuracy

In `annotation_analysis`

To test the accuracy of our model, we compared our manually annotated data to the model annotations.

We generated plots to visualize sentiment distribution across translations and finding the similarity between each set of annotations for every translation.

> [accuracy.ipynb](annotation_analysis/accuracy.ipynb)

```python
# model compared to mode of annotator labels
df["annotator_sentiment_mode"] = df.iloc[:, 2:-1].apply(lambda x: x.mode()[0], axis=1)
accuracies_mode = []
for bible in bibles:
    accuracies_mode.append(
        accuracy_score(
            df[df["bible"] == bible]["bible_sentiment"],
            df[df["bible"] == bible]["annotator_sentiment_mode"],
        )
    )

# model compared to mean of annotator labels
df["annotator_sentiment_avg"] = df.iloc[:, 2:-2].mean(axis=1).round().astype("int64")
accuracies_mean = []
for bible in bibles:
    accuracies_mean.append(
        accuracy_score(
            df[df["bible"] == bible]["bible_sentiment"],
            df[df["bible"] == bible]["annotator_sentiment_avg"],
        )
    )

# accuracy for each annotator
annotator_accuracies = []
for annotator in df.columns[2:-3]:
    annotator_accuracies.append(accuracy_score(df["bible_sentiment"], df[annotator]))
```

From this plot, which utilizes the mode of our manual annotations as a baseline, we found that our model classified the `FBV` tranlation with the highest accuracy, of around 80%:
![acc_mode](readme_plots/acc_mode.png)

It is important to note that our annotator agreement scores are fairly low -- as we see in the kappa scores -- and the amount of labeled samples we have are very low.

# Translation Comparisons

In `annotation_analysis`

The overall goal of our project was to identify if there were any discrepancies in sentiment across different translations of the Bible.

In [bible_comparison.ipynb](annotation_analysis/bible_comparison.ipynb) we analyze the sentiment distribution across each translation and compare them.

```python
for bible in bibles:
    bible_df = pd.read_csv(bibles[bible])
    # Get the total count of each sentiment
    sentiment_counts = bible_df["sentiment"].value_counts()
    print(sentiment_counts)

    # percent distribution of each sentiment
    bible_sentiments[bible]["positive"] = sentiment_counts[2] / sentiment_counts.sum()
    bible_sentiments[bible]["negative"] = sentiment_counts[3] / sentiment_counts.sum()
    bible_sentiments[bible]["neutral"] = sentiment_counts[1] / sentiment_counts.sum()

    total_sentiment[bible] = (sentiment_counts[2]) / (sentiment_counts[2] + sentiment_counts[3])
```

One of of visualizations shows the percent of each classification per translation. See [bible_comparison.ipynb](annotation_analysis/bible_comparison.ipynb) for more.

![bib_sent2](readme_plots/bib_sent2.png)

**Overall, we found that the most neutral translation seems to be the `ASV`. The translation with the highest variance is the `FBV`, with the highest percentage of positive AND negative sentiment scores across all translations, also featuring a relatively low level of neutrality**

# Sentiment by Character

In `char_sent_analysis`

For this portion of our project, we were interested in finding whether or not different translations portray characters differently.

By filtering our model-annotated data, we could isolate chunks of verses that explicitly mentioned certain characters.

In [characters.ipynb](char_sent_analysis/characters.ipynb) we filter our model-annotated data by character.

First we define a list of 100 characters that appear in most translations of the Bible. We then find the most frequent sentiment and average confidence for verse chunks containing those characters.

```python
# init empty dict with character names as keys (set up framework for storing sentiment data)
result_data = {name: {} for name in character_names}

# finding most frequent sentiment AND average confidence for that sentiment for each character in each translation
for bible, path in bible_paths.items():
    df = pd.read_csv(path)
    bible_sentiment_count = {name: {1: [], 2: [], 3: []} for name in character_names} # storing sentiment counts and confidence values here
    for index, row in df.iterrows():
        sentiment = int(row['sentiment'])
        confidence = float(row['confidence'])
        if sentiment in [1, 2, 3]:  # disregard the sentiments labeled as 0 from model annotations (runtime error with web.csv)
            # iterate through all names, if the name appears in the row, append the confidence value to the character/sentiment pair
            for name in character_names:
                if name in row['text']:
                    bible_sentiment_count[name][sentiment].append(confidence)
    # find the most frequently occuring sentiment for each mention of a character and
    # calculate average confidence for that sentiment score
    for name in character_names:
        max_sentiment = most_frequent_sentiment(bible_sentiment_count[name])
        if max_sentiment is not None:
            result_data[name][bible + ' sent'] = max_sentiment
            result_data[name][bible + ' conf'] = sum(bible_sentiment_count[name][max_sentiment]) / len(
                bible_sentiment_count[name][max_sentiment])
```

**Overall, we found that the `ASV` seems to portray characters with the most neutral sentiment, which coincides with our findings from the previous section**

In order to identify possible discrepancies, we identified characters that were labeled with neutral, positive AND negative sentiment in one or more translations. Out of this list, we gathered the top 10 characters based on variance within the average confidence scores for each translation. Results from this step are exported as `top10conflict_confidence.csv`

```python
df = pd.read_csv('char_analysis_confidence.csv', index_col=0)
df = df.dropna().replace('N/A', pd.NA)
conflict_characters = []

# check for if character has been labeled as neu, pos AND neg in one or more translations
for index, row in df.iterrows():
    unique_sentiments = row.nunique()
    if unique_sentiments >= 3:
        conflict_characters.append((index, unique_sentiments))

# grab top 10 based on variance in confidence scores
conflict_characters.sort(key=lambda x: x[1], reverse=True)
top10 = conflict_characters[:10]

# generate new df and export to csv for visualization
conflicting_df = df.loc[[character[0] for character in top10]]
conflicting_df.to_csv('top10conflict_confidence.csv')
conflicting_df
```

Our results indicated that **Adam** was the most volatile character when it comes to sentiment across translations, with the following results:

    name,asv sent,asv conf,fbv sent,fbv conf,web sent,web conf,wmb sent,wmb conf,kjv sent,kjv conf
    Adam,Neutral (1),0.76,Negative (3),0.74,Positive (2),0.81,Negative (3),0.59,Neutral (1),0.74

# Low Frequency Token Analysis

In [Low_freq.ipynb](Low_Freq_analysis/Low_freq.ipynb) we analyze the model's accuracy with the least frequent tokens.

First we obtained the 200 least frequent words in each bible translation. We then compared the average confidence and sentiment with and without the bible verses that include these words.

```python
def get_lowest_frequency(df, n=200):
    # gets the least frequent words from each text
    frequency = {}
    texts = df["text"]
    for text in texts:
        # removes unnecessary characters
        words = re.sub(r'[^\w\s]', '', text.lower()).split()
        for word in words:
            if word in frequency:
                frequency[word] += 1
            else:
                frequency[word] = 1
```

The metrics from this analysis were saved in [least_frequent_results.csv](Low_Freq_analysis/least_frequent_results.csv).

We found that there was no significant difference between the accuracy of the model with the least frequent tokens and without them.

# Contributions

**Project maintenence**

- Collect the data
  - Olivia
  - Cameron
- Coordinate and analyze manual annotations
  - Olivia
- Run the model on the bibles
  - Nick
- Compare model to manual annotations (accuracy)
  - Cameron
  - Gerardo
- Compare sentiment between characters
  - Nick
- Low frequency token analysis
  - Gerardo
  - River
- Lead making the slides
  - Everyone
- Lead writing the README and organize the repo/drive/requirements for deliverables
  - Olivia
  - Nick
- Maintaining repo
  - Cameron

<br>

**File authors**

- [original api pipeline](data/bibles/old_api_pipeline/pipeline.ipynb) - Cameron
- [api_pipeline.ipynb](data/bibles/api_pipeline.ipynb) - Olivia
- [get_random_verses.ipynb](data/random_verses/get_random_verses.ipynb) - Cameron
- [get_random_chunks.ipynb](data/random_chunks/get_random_chunks.ipynb) - Olivia
- [anno_instructions.md](manual_annotation/anno_instructions.md) - Olivia
- [anno.py](manual_annotation/anno.py) - Olivia
- [agreement.ipynb](annotation_analysis/agreement.ipynb) - Olivia
- [bible.ipynb](model_annotation/bible.ipynb) - Nick
- [bible_comparison.ipynb](annotation_analysis/bible_comparison.ipynb) - Gerardo, Nick
- [characters.ipynb](char_sent_analysis/characters.ipynb) - Nick
- [Low_freq.ipynb](Low_Freq_analysis/Low_freq.ipynb) - Gerardo, River
- [accuracy.ipynb](annotation_analysis/accuracy.ipynb) - Cameron
- [bible_comparison.ipynb](annotation_analysis/bible_comparison.ipynb) - Cameron, Gerardo
