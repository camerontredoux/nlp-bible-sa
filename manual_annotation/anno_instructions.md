## 1. Copy the random verses
 Make a copy of each csv file in `data/random_verses` and put it in a folder named with your name like
```
manual_annotation
└── John
    ├── asv.csv
    ├── fbv.csv
    ├── kjv.csv
    ├── web.csv
    └── wmb.csv

```

## 2. Annotate
### Option A - Use Olivia's script
1. Copy `./anno.py` into your folder with your csv files.
2. Go to the main function and change it to your name
3. Run the file

### Option B - Do it manually
To each file, add a `sentiment` label to each row like this

```
citation,verse,sentiment
JHN.2.2,"And both Jesus was called, and his disciples, to the marriage.", 1
```

CLASSIFICATIONS\
**1**: Neutral\
**2**: Positive\
**3**: Negative
