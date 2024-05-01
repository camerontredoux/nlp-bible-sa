# Sentiment Analysis of Bible translations

This repo contains the source code for our analysis of 5 bible translations: American Standard Version (ASV), Free Bible Version (FBV), King James Version (KJV), World English Bible (WEB), World Messianic Bible (WMB).

# Setup
Note: you must have a Python environment and a way to run .ipynb files.

1. clone the repo
2. cd into it
3. run
```pip install -r requirements.txt```

# Data

Bible data is easy to come by, however we needed our data to be uniform across translations, which is not as easy. Our requirements led us to use https://scripture.api.bible/.

API.Bible is an api that provides access to ~2500 bible translations in a uniform format.\
To collect the data we did the following:
1. Make an API.Bible account to get a key
2. Run [api_pipeline.ipynb](data/bibles/api_pipeline.ipynb) in `data/bibles/`

Here is a brief overview of the function that does the heavy lifting. The actual code is more robust with exceptions for failed requests so only actual verses are written to the csv, this is simplified for readability.

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

# Manual Annotation

In order to evaluate our model's accuracy we need labeled data. Because we don't have access to sentiment labels for all of these bibles, we needed to do our own annotation.

Initially we were going to annotate by verse, but we learned that bible verses are quite short (many just a sentence fragment), making them challenging to annotate alone.\
Because of this we decided to annotate in chunks of 3 verses.

We chose to anotate
- 10 chunks of ASV, FBV, WEB, & WMB
- 50 chunks of KJV

We chose to do 50 of KJV instead of 10 like the others because we were particularly worried about the model's accuracy on the old style of english KJV uses. Looking back it would've been better to do 50 for all of the bibles; 10 was too few.

Getting the chunks
---
1. `data/bibles_chunked/`\
Chunk the data into groups of 3 verses in [chunk_verses.ipynb](data/bibles_chunked/chunk_verses.ipynb)
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
2. `data/random_chunks/`\
Get random chunks from each bible in [get_random_chunks.ipynb](data/random_chunks/get_random_chunks.ipynb)
   ```python
    # make csv with 10 random chunks for each modern bible
    for bible in ["asv.csv", "fbv.csv", "web.csv", "wmb.csv"]:
        df = pd.read_csv(f"../bibles_chunked/{bible}")
        df.sample(10).to_csv(f"{bible}", index=False)

    # 50 chunks for the KJV
    df = pd.read_csv("../bibles_chunked/kjv.csv")
    df.sample(50).to_csv("kjv.csv", index=False)
   ```

   
Annotating the chunks
---
in `manual_annotation/`

We each added a 'sentiment' column to each random chunk, either by manually typing the numbers or using the [anno.py](manual_annotation/anno.py) script.

Each of us followed the instructions in [anno_instructions.md](manual_annotation/anno_instructions.md)

> ## 1. Copy the random verses
> Make a copy of each csv file in `data/random_chunks` and put it in a folder named with your name like
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
> ### Option A - Use Olivia's script
> 1. Copy `./anno.py` into your folder with your csv files.
> 2. Go to the main function and change it to your name
> 3. Run the file
> 
> ### Option B - Do it manually
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


Annotator agreement
---
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

# Model Accuracy

# Translation Comparisons

# Sentiment by Character

# Low Frequency Token Analysis
We made a Low Frequency Token Analysis so as to assess our model's accuracy given the occurance of unusual tokens. Given that we could not obtain the tokens our model was trained on, we obtained the 200 least frequent words in each bible translation and compared the average confidence and sentiment averages with and without the bible verses that include these words. We included plots to visualise our results using matplotlib. We also included a csv that contains all the resulting metrics from this analysis, which are located in `Low_Freq_analysis/least_frequent_results.csv`. The metrics we used to assess our model's accuracy were neutral, postive, and negative averages, along with the average confidence. We used pandas for data manipulation and to remove the rows with the lowest frequency tokens. We found that there was no significant difference between the accuracy of the model with the least frequent tokens and without them.

The Low Frequency token Analysis is located in `Low_freq.ipynb`. Make sure to download full `Low_Freq_analysis` folder and install all dependencies.

# Contributions

**Project maintenence**

!! EVERYONE LOOK AT THIS AND CORRECT IT
!! I DON'T KNOW WHAT EVERYONE DID IN THE END!!
- Collect the data
  - Olivia
  - Cameron
- Coordinate and analyze manual annotations
  - Olivia
- Run the model on the bibles
  - Nick
  - Cameron
- Compare model to manual annotations (accuracy)
  - Cameron
  - Gerardo
- Compare sentiment between characters
  - Nick
- Low frequency token analysis
  - Gerardo
  - River
- Lead making the slides
  - River
  - Gerardo
- Lead writing the README
  - Olivia
- Organize the repo/drive for deliverables
  - Olivia

<br>

**File authors**
- [original api pipeline](data/bibles/old_api_pipeline/pipeline.ipynb) - Cameron
- [api_pipeline.ipynb](data/bibles/api_pipeline.ipynb) - Olivia
- [get_random_verses.ipynb](data/random_verses/get_random_verses.ipynb) - Cameron
- [get_random_chunks.ipynb](data/random_chunks/get_random_chunks.ipynb) - Olivia
- [anno_instructions.md](manual_annotation/anno_instructions.md) - Olivia
- [anno.py](manual_annotation/anno.py) - Olivia
- [agreement.ipynb](annotation_analysis/agreement.ipynb) - Olivia
- [bible.ipynb](model_annotation/bible.ipynb)
- [bible_comparison.ipynb](annotation_analysis/bible_comparison.ipynb)
- [positive_negative_analysis.ipynb](annotation_analysis/positive_negative_analysis.ipynb)
- [characters.ipynb](data/character_chunks/characters.ipynb)
- [Low_freq.ipynb](Low_Freq_analysis/Low_freq.ipynb) - Gerardo
