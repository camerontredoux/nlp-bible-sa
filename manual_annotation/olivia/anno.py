import os
from glob import glob
from pandas import DataFrame, read_csv

def print_progress(name):
    csv_files = glob(f'*/{name}/*.csv')

    for f in csv_files:
        df = read_csv(f)

        # count num annotated rows
        numLabaled = 0
        if 'sentiment' in df.columns:
            numLabaled = len(df[df['sentiment'].isin([1, 2, 3])])

        # print progress
        print(f'{os.path.basename(f).split('.')[0]}: {numLabaled}/{len(df)}')
    

def tater_loop(df) -> DataFrame:
    for i, row in df.iterrows():

        # skip row if already annotated
        if row['sentiment'] in [1, 2, 3]:
            continue

        # print progress and verse text
        print(f'\n{i+1}/{len(df)}')
        print(row['text'])

        # handle input
        while True:
            sentiment = input('\n1: neu, 2: pos, 3: neg, q: save and exit\nsentiment: ')
            match sentiment:
                case 'q':
                    print('\nsaving and quitting')
                    return df
                case '1' | '2' | '3':
                    df.at[i, 'sentiment'] = int(sentiment)
                    print(f'labeled chunk {row['chunk']} as {sentiment}\n')
                    break
                case _:
                    print('Invalid input')

    print('Finished annotating!')
    return df


def annotate(name):

    print_progress(name)

    # input bible name
    bibname = input("\nEnter bible to annotate (e.g. asv, q to quit): ")
    if bibname == 'q':
        print('🫡')
        return

    # read csv
    path = glob(f'*/{name}/{bibname}.csv')[0]
    df = read_csv(path)

    #add empty sentiment column
    if 'sentiment' not in df.columns:
        df['sentiment'] = 0

    # annotate
    df = tater_loop(df)

    # save
    df.to_csv(path, index=False)
    print(f'Saved to {path}\n')

    # continue
    annotate(name)


if __name__ == '__main__':
    annotate('olivia')
