import os
import pandas as pd
rating_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/ratings.dat'
ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv(rating_file_path, sep='::', names=ratings_cols, engine='python')
orginal_data_size = len(ratings)
ratings.head()

# 3점 이상만 남깁니다.
ratings = ratings[ratings['rating']>=3]
filtered_data_size = len(ratings)

print(f'orginal_data_size: {orginal_data_size}, filtered_data_size: {filtered_data_size}')
print(f'Ratio of Remaining Data is {filtered_data_size / orginal_data_size:.2%}')
# rating 컬럼의 이름을 count로 바꿉니다.
ratings.rename(columns={'rating':'count'}, inplace=True)

# 영화 제목을 보기 위해 메타 데이터를 읽어옵니다.
movie_file_path=os.getenv('HOME') + '/aiffel/recommendata_iu/data/ml-1m/movies.dat'
cols = ['movie_id', 'title', 'genre'] 
movies = pd.read_csv(movie_file_path, sep='::', names=cols, engine='python')
movies.head()

# 유니크한 영화 개수
ratings['movie_id'].nunique()

# 유니크한 사용자 수
ratings['user_id'].nunique()

#영화 ID 별 영화 제목 딕셔너리
movie_dict=movies.set_index('movie_id')['title'].to_dict()

ratings_count = ratings.groupby('movie_id')['user_id'].count()
famous_rating_series=ratings_count.sort_values(ascending=False).head(30)
famous_rating_df = pd.DataFrame(data=famous_rating_series.values,index=famous_rating_series.index,columns=['watch_count'])
famous_rating_df = famous_rating_df.reset_index()

for i in range(len(famous_rating_df)):
    famous_rating_df['title'] = movie_dict[famous_rating_df.loc[i][0]]