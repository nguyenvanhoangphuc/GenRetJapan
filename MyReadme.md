# create môi trường ảo bằng conda nếu đã có conda sẵn # conda 24.1.2
conda create -n myenv python=3.10
# khởi động môi trường ảo
conda activate myenv
# tắt môi trường ảo
conda deactivate
# cài đặt pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# unzip in window
tar -xf dataset/nq320k.zip

# run genret for nq320k dataset
python run.py --model_name t5-base --code_num 512 --max_length 3 --train_data nq320k/train.json --dev_data nq320k/dev.json --corpus_data nq320k/corpus_lite.json --save_path out/model
# run genret for nq320k dataset split no using bm25
python run.py --model_name t5-base --code_num 512 --max_length 3 --train_data nq320k_split/train.json --dev_data nq320k_split/dev.json --corpus_data nq320k_split/corpus_lite.json --save_path out_split/model
# run genret for nq320k dataset split using bm25
python run.py --model_name t5-base --code_num 512 --max_length 3 --train_data nq320k_bm25_split/train.json --dev_data nq320k_bm25_split/dev.json --corpus_data nq320k_bm25_split/corpus_lite.json --save_path out_bm25/model

Chú thích: 
- file train.json là một list các item, mỗi item là một list gồm 2 phần tử, phần tử đầu tiên sẽ là question (hoặc query), phần tử thứ 2 là số thứ tự (id) của document mà nó liên quan đến trong bộ corpus_lite.json
+ số lượng item: 307373
- file corpus_lite.json là một list các document, mỗi document là một chuỗi string, thứ tự của document cũng chính là id của document đó.
+ số lượng corpus: 109739
- file train.json.qg.json tương tự như train.json nhưng sẽ có những câu đồng nghĩa của từng câu trong train.json làm số lượng được nhân lên gần 4 lần.
+ số lượng item: 1097390
- file dev.json tương tự như train.json nhưng số lượng ít hơn
+ số lượng item: 7830

# Giải thích input + output:
input: 
--train_data nq320k/train.json (file train)
+ là mảng các mảng có 2 phần tử là [a, b] trong đó a là nội dung query (question), b là int là chỉ số doc liên quan đến trong corpus
--dev_data nq320k/dev.json (file validation)
+ tương tự như train
--corpus_data nq320k/corpus_lite.json (file corpus)
+ gồm mảng các string, mỗi string là một doc và id_doc chính bằng số thứ tự bắt đầu từ 0
-- file nq320k/dev.json.seen 
-- file nq320k/dev.json.unseen
+ trong file nq320k/dev.json có các trường hợp dễ (vận dụng thấp, câu tương tự với tập train) và các trường hợp khó hơn (vận dùng cao, câu khác biệt hơn) lần lượt xếp vào file seen(dễ), unseen(khó)
+ cái ni đặc trưng của bộ nq320k thôi còn đối với các tập khác thì nó không có phân biệt ra dễ và khó
output:
--save_path out/model
lần lượt tạo ra các thư mục model-1-pre, model-1, model-2-pre, model-2, model-3-pre, model-3, model-3-fit
số epochs huấn luyện có thể được quy định, mặc định thì model-1-pre là 1 epoch, model-2-pre, model-3-pre là 10 epochs, 
model-1, model-2, model-3 mặc định là 200 mà vì mình lưu step là 9 nên phải là bội của 9 + 1, hiện tại trong code chọn thấp hơn là 100
model-3-fit mặc định là 1000 thoã mãn bội của 9 + 1 mà nó hơi nhiều nên giảm xuống hiện tại trong code là 361


# Chỉnh batch_size:
- Tìm kiếm in_batch_size dòng 635, chỉnh từ 32 lên 128 nếu như có GPU dư