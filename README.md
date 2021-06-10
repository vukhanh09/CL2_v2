# CL2_v2

# Project dùng để phân loại văn bản cho lĩnh vực Ô tô - xe máy


# File yêu cầu 
Link folder train word2vec : https://drive.google.com/drive/folders/1K8KSXUcbpO93BCj98-l9_knCsHVgIX0w?usp=sharing

# Hướng dẫn sử dụng
- Tạo đối tượng : classify = Classfify(tfidfTf_path,key_path,vi_path)
  + tfidfTf_path : path của file train
  + key_path : path của tập từ điển
  + vi_path : path folder train word2vec

- load model: classify.loadModel(model_path)
  + model_path : path của file model đã train


- predict văn bản : classify.predictString(str) 
  + str : văn bản

- predict tập văn bản dạng csv : 
B1 : tiền xử lý
  classify.process(path_in,path_out)
B2 : predict
  y_pred = classify.predictCsv(path_out)
  + Xtest_path : path file csv
