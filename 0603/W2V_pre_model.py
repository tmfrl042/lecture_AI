#영문 : https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
#한글 : https://drive.google.com/file/d/0B0ZXk88koS2KbDhXdWg1Q2RydlU/view
import gensim
model = gensim.models.Word2Vec.load('d:/temp/ko.bin')

result = model.most_similar("게임")
print(result)