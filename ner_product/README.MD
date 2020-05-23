<h1><b>PhoBERT-NER-PRODUCT</b></h1>

Named entity recognition (NER) nhận diện tên sản phẩm trong 1 câu chat sử dụng pre-trained model PhoBERT (download pre-trained model và xem hướng dẫn của tác giả tại đây: https://github.com/VinAIResearch/PhoBERT)
<hr>
<h2><b>Bài toán</b></h2>

Comment khách hàng: 
 <i>"Sam sung s8lu có hỗ trợ thu cũ đổi mới không vậy ạ"</i>

 Kết quả:
 <i>"<b>Sam sung s8lu</b> có hỗ trợ thu cũ đổi mới không vậy ạ"</i>
<hr>

<h2><b>Phương pháp</b></h2>

Tạo custom model cho mục đích NER. Nối (concat) 4 trạng thái ẩn (hidden-state) là output của 4 khối encoder cuối cùng của BERT để sử dụng cho:

 Finetune: đầu vào cho lớp Linear

 Feature extractor: cho ra vector ngữ cảnh


