from os.path import dirname, basename, abspath
from glob import glob
from pandas import DataFrame, read_csv

def print_progress(name):
    csv_files = glob(f'{DIR}/*.csv')

    for f in csv_files:
        df = read_csv(f)

        # count num annotated rows
        numLabeled = 0
        if 'sentiment' in df.columns:
            numLabeled = len(df[df['sentiment'].isin([1, 2, 3])])

        # print progress
        print(f'{basename(f).split(".")[0]}: {numLabeled}/{len(df)}')
    

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
                    print(f'labeled chunk {row["chunk"]} as {sentiment}\n')
                    break
                case _:
                    print('Invalid input')

    print('Finished annotating!')
    return df


def annotate(name):

    print_progress(name)

    # input bible name
    bibName = input("\nEnter bible to annotate (e.g. asv, q to quit): ")
    if bibName == 'q':
        print('🫡')
        return

    # read csv
    bibPath = glob(f'{DIR}/{bibName}.csv')[0]
    df = read_csv(bibPath)

    #add empty sentiment column
    if 'sentiment' not in df.columns:
        df['sentiment'] = 0

    # annotate
    df = tater_loop(df)

    # save
    df.to_csv(bibPath, index=False)
    print(f'Saved to {bibPath}\n')

    # continue
    annotate(name)


if __name__ == '__main__':
    DIR = dirname(abspath(__file__))
    
    # add your name in the argument
    annotate('nick')