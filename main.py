from P2 import Classfify


tfidfTf_path = 'tfidftransformer.pkl'


vi_path ="vi/vi.vec"
model_path = "pickle_model.pkl"
key_path = 'dictionary.txt'

classify = Classfify(tfidfTf_path,key_path,vi_path)
classify.loadModel(model_path)

# sử dụng cho string

str = 'Theo dự báo từ BloombergNEF, vào năm 2026, các mẫu sedan và SUV chạy bằng điện sẽ có mức giá rẻ ngang với các mẫu xe chạy bằng xăng hay diesel. Các mẫu xe nhỏ hơn chạy bằng điện sẽ giá ngang hoặc rẻ hơn xe chạy bằng xăng vào năm 2027. ở dĩ vậy vì chi phí cho việc sản xuất pin của xe điện đang ngày càng giảm xuống, kết hợp với việc dây chuyền sản xuất chuyên dụng cho loại xe này đang tăng lên ở các nhà sản xuất ô tô. Số liệu của BloombergBEF cho thấy, giá bán lẻ trung bình trước thuế của 1 chiếc ô tô điện tại Anh là 33.000 euro (khoảng hơn 924 triệu đồng), so với con số 18.600 euro (khoảng hơn 520 triệu đồng) của 1 chiếc chạy bằng xăng. Tuy nhiên, vào năm 2026, dự kiến hai xe sẽ có cùng mức giá 19.000 euro (khoảng hơn 532 triệu đồng). Đến năm 2030, cùng một loại ô tô, xe chạy bằng điện sẽ có giá trung bình 16.300 euro (trước thuế, khoảng hơn 456 triệu đồng), trong khi đó, xe chạy bằng xăng lại có giá cao hơn, với 19.900 euro (khoảng hơn 557 triệu đồng). Một nghiên cứu mới do Transport & Environment (một tổ chức phi lợi nhuận có trụ sở tại Brussels (Bỉ), ủy quyền thực hiện các chiến dịch vận tải sạch hơn ở Châu Âu) cũng dự đoán giá pin mới cho ô tô điện sẽ giảm 58% từ năm 2020 đến năm 2030 xuống chỉ còn khoảng 58 USD cho mỗi kWh, giảm xuống dưới 100 USD mỗi kWh so với hiện nay.'
y_pred = classify.predictString(str)
print(y_pred)

# sử dụng cho file csv
classify.process('out1.csv','out.csv')
y_pred = classify.predictCsv('out.csv')
print(y_pred)


