from konlpy.tag import Komoran
from konlpy.tag import Okt

set

from gensim.models import Word2Vec
import string

# 형태소 분석기
#tagger = Komoran
tagger = Okt()

txt_line = []
with open('./data/news.txt', 'r') as fr:
    for line in fr:
        txt_line.append(line.strip(string.punctuation))

txt = ' '.join(txt_line)

print(tagger.pos(txt))
tokenized_contents = tagger.nouns(txt)
tokenized_contents = [ele for ele in tokenized_contents if len(ele) > 1]

print(tokenized_contents[:])

# model = Word2Vec(data,         # 리스트 형태의 데이터
#                  sg=1,         # 0: CBOW, 1: Skip-gram
#                  size=100,     # 벡터 크기
#                  window=3,     # 고려할 앞뒤 폭(앞뒤 3단어)
#                  min_count=3,  # 사용할 단어의 최소 빈도(3회 이하 단어 무시)
#                  workers=4)    # 동시에 처리할 작업 수(코어 수와 비슷하게 설정)

model = Word2Vec([tokenized_contents], size=100, window=3, negative=3, min_count=3, sg=0)

result = model.most_similar('게임')
print(result)

cos = model.wv.similarity('게임', '질병')
print(cos)