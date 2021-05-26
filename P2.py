import pandas as pd
import numpy as np
import gensim
from gensim.models import KeyedVectors 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer 
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score,recall_score,precision_score
import pickle
import pandas as pd
import numpy as np
import re
from pyvi import ViTokenizer, ViPosTagger
import gensim


class Classfify():
    def __init__(self, tfidfTf_path,key_path,vi_path):
        self.tfidfTf_path = tfidfTf_path
        self.keys = []
        f = open(key_path)
        for x in f:
            self.keys.append(x.replace("\n",""))
        f.close()
        

        self.cv = CountVectorizer(vocabulary=self.keys)
        # self.word_count_vector=self.cv.fit_transform(self.data_train['all'])
        # self.tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
        # self.tfidf_transformer.fit(self.word_count_vector)
        self.tfidf_transformer = pickle.load(open(self.tfidfTf_path, "rb"))



        self.w2v = KeyedVectors.load_word2vec_format(vi_path)
        self.vocab = self.w2v.wv.vocab #Danh sách các từ trong từ điển
    def loadModel(self,model_path):
        with open(model_path, 'rb') as file:
            self.clf = pickle.load(file)

    #tiền xử lý file csv ban dau
    def process(self,source,target):
        data = pd.read_csv(source)
        data = data.fillna("")
        data['all'] = data['category'] +" "+ data['title'] +" "+ data['descriptions'] +" "+ data['content']
        for x in range(data.shape[0]):
            # if x % 1000==0:
            #     print(x)
            data.loc[x,'all'] = ' '.join(gensim.utils.simple_preprocess(data.iloc[x]['all']))
            data.loc[x,'all'] = ViTokenizer.tokenize(data.iloc[x]['all'])
        data.to_csv(target,columns=['all'],index=False)



    def computing_tfidf(self,s):
        input_s =[]
        input_s.append(s)
        count_vector=self.cv.transform(input_s) 
        tf_idf_vector=self.tfidf_transformer.transform(count_vector)
        values=tf_idf_vector.toarray().tolist()[0]
        dictionary = dict(zip(self.keys, values))
        return dictionary

    def convertStringInt(self,str):
        output = ' '.join(gensim.utils.simple_preprocess(str))
        out =  ViTokenizer.tokenize(output)
        X = np.zeros((1,100))
        dictX = self.computing_tfidf(out)
        words = out.split(" ")
        for word in words:
            if word in self.keys and word in self.vocab:
                X[0] += self.w2v[word]*dictX[word]
        return X
    def convertToInt(self,data):
        X = np.zeros((data.shape[0],100))
        for x in range(data.shape[0]):
            dictX = self.computing_tfidf(data.iloc[x]['all'])
            words = data.iloc[x]['all'].split(" ")
            for word in words:
                if word in self.keys and word in self.vocab:
                    X[x] += self.w2v[word]*dictX[word]
        return X
    def predictCsv(self,path): # data da xu ly
        data = pd.read_csv(path)
        X_test_int = self.convertToInt(data)
        y_pred = self.clf.predict(X_test_int)
        return y_pred
    def predictString(self,str):  ## input là string
        Xtest = self.convertStringInt(str)
        # print(Xtest.shape)
        y_pred = self.clf.predict(Xtest)
        # print(y_pred)
        return y_pred


    def train(self,X,Y):
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X, Y['target'])
        return clf
