import pandas as pd

# df = pd.read_csv('./data/ml-20m/ratings.csv', names=['uid', 'mid', 'rating', 'ts'], usecols=[0,1,2], dtype='str')
# mdf = pd.read_csv('./data/ml-20m/movies_i.csv', names=['ind', 'mid', 'title', 'genre'], usecols=[0,1], dtype='str')

# modf = pd.merge(df, mdf, on='mid', how='left')
# modf.drop(['mid'], axis=1, inplace=True)

# modf.to_csv('ratings_mod.csv', index=False)

db = pd.read_csv('ratings_mod.csv', )
db = db.loc[db['user_id']<=1000]
db.to_csv('ratings_mod_1000.csv', index=False)