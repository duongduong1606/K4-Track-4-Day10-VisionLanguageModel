Họ tên: Dương Dương
`dataset_version`: `d10-21e4070301`

> Trạng thái báo cáo ngày 15/10/2026: đã hoàn thành E0–E3 và ablation E4 rank 4/rank 64. Tất cả sáu cấu hình đã được chấm trên đủ 311 ảnh test bằng cùng `dataset_version`; số liệu dưới đây được tái tạo trực tiếp từ prediction thô bằng `scripts/score.py`.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | zero-shot | 0 | — | — | 0.1077 | [0.0812, 0.1369] | 0.8360 | 0.1032 | 0.0526 | 0.2231 |
| E1 | LoRA LLM | 33,030,144 | 10.88 | 53.8 | 0.2806 | [0.2470, 0.3195] | 1.2186 | 0.1684 | 0.2235 | 0.5116 |
| E2 | LoRA LLM + projector | 142,135,296 | 13.79 | 54.4 | 0.3419 | [0.3018, 0.3835] | 1.1511 | 0.2081 | 0.2790 | 0.5942 |
| E3 | E2 + LoRA ViT (r=16) | 148,426,752 | 7.47 | 66.1 | 0.4305 | [0.3897, 0.4713] | 0.8103 | 0.2304 | 0.3728 | 0.6689 |
| E4a | E3, rank=4 | 118,935,552 | 8.30 | 65.8 | 0.4335 | [0.3900, 0.4771] | 0.8328 | 0.2791 | 0.3546 | 0.6959 |
| E4b | E3, rank=64 | 266,391,552 | 9.45 | 68.2 | 0.4396 | [0.3989, 0.4843] | 0.7846 | 0.3229 | 0.3492 | 0.6844 |

Nguồn điểm là sáu JSON trong `report/evidence/`; nguồn tài nguyên huấn luyện là các file `work/adapters/*/run.json`. Bản tóm tắt có thể commit nằm ở `report/evidence/training_progress.json`. Adapter/trọng số và prediction thô nằm trong `work/`, được `.gitignore` loại khỏi Git đúng quy định.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Theo cây module và phép đếm dùng trong notebook, Qwen3-VL-4B có tổng khoảng 4.44B tham số: LLM cùng embedding khoảng 4.0B và ViT khoảng 415M. Phần projector được mở ở E2/E3 gồm `visual.merger` và `visual.deepstack_merger_list`; chênh lệch số tham số trainable giữa E2 và E1 là **109,105,152**, phù hợp quy mô khoảng 109M của projector.

- E1 cập nhật LoRA trên 252 linear của LLM; tổng trainable **33,030,144**.
- E2 giữ LoRA LLM như E1 và cập nhật đầy đủ projector float32; tổng trainable **142,135,296**.
- E3 thêm LoRA trên 96 linear của ViT, thành 348 module LoRA, đồng thời vẫn cập nhật projector; tổng trainable **148,426,752**, nhiều hơn E2 **6,291,456** tham số LoRA ViT.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

E0 có `parse_valid_rate = 0.9357`, tức 291/311 output parse được và 20 output hỏng. Hai lỗi định lượng nổi bật là bỏ sót box (**535 FN**, recall 0.0823) và dự đoán dư (**260 FP**, trong đó crack chiếm 182 FP). Ví dụ output thô hợp lệ của ảnh `vlcsnap-00200`:

```json
[{"label":"manhole","bbox_2d":[770,640,845,690]},{"label":"manhole","bbox_2d":[907,786,1000,824]}]
```

Toàn bộ E0 đạt precision 0.1558, recall 0.0823 và F1 0.1077. Sáu ví dụ có box thật/dự đoán được lưu trong `report/evidence/e0_examples.png`.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Nếu model trả tọa độ 0–1000 nhưng scorer hiểu là pixel 640×360 (hoặc ngược lại), box bị co giãn khác nhau theo hai trục. IoU vì thế giảm mạnh và nhiều cặp rơi dưới ngưỡng 0.5, làm TP giảm, FN/FP tăng và F1 có thể về gần 0 dù model chọn đúng vùng theo hệ tọa độ của nó.

Tôi đã kiểm tra round-trip trên một ảnh: đổi nhãn YOLO chuẩn hóa sang JSON 0–1000, đổi ngược về pixel rồi vẽ chồng lên ảnh. Notebook in `Kiểm tra quy đổi: OK`; ví dụ có box manhole `[317,481,616,625]`. Ảnh kiểm tra dữ liệu được lưu tại `report/evidence/sample_boxes.png`.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Mục tiêu của SFT là học chuỗi JSON đáp án có điều kiện theo ảnh và prompt, nên các token trước đáp án được đặt label `-100`. Nếu tính loss trên toàn prompt, phần lớn token dễ dự đoán (system/user template và câu lệnh cố định) sẽ lấn át số token JSON; model tối ưu việc chép prompt thay vì grounding. Token ảnh không phải nhãn ngôn ngữ cần dự đoán; ép loss lên chúng vừa sai mục tiêu vừa tăng bộ nhớ và có thể làm gradient không ổn định. Collator trong notebook chỉ giữ label khác `-100` từ đầu câu trả lời đến `<|im_end|>`.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

Trọng số `Linear4bit` là biểu diễn lượng tử phục vụ tính toán, không phải tham số float đầy đủ để AdamW cập nhật trực tiếp. Lần chạy đầu E2 chỉ tìm được **54,579,200** tham số projector, cho thấy cấu hình skip module chưa bao phủ đúng cây module. Tôi sửa cấu hình thành `llm_int8_skip_modules=["model.visual", "lm_head"]`, thêm kiểm tra không có `Linear4bit` trong projector, rồi mở cả `visual.merger` và `visual.deepstack_merger_list`. Run E2 hoàn chỉnh có thêm **109,105,152** tham số so với E1.

Projector được chuyển sang float32 để lưu trọng số và trạng thái cập nhật với độ chính xác đủ cao. Đây là khối được fine-tune đầy đủ, không phải LoRA; giữ nó ở dạng 4-bit sẽ không cung cấp tham số float hợp lệ cho optimizer, còn fp16 dễ mất độ chính xác khi cộng các cập nhật nhỏ.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Các frame trong cùng phiên quay gần nhau về mặt đường, góc máy, ánh sáng và thậm chí cùng một hư hại. Chia ngẫu nhiên theo ảnh có thể đưa các frame gần trùng vào cả train và test, khiến mô hình gần như nhận diện lại cảnh đã thấy và F1 cao hơn khả năng tổng quát hóa thực. Split hiện tại tách theo `session`; notebook đã kiểm tra không có session giao nhau.

| Split | Ảnh | Pothole | Crack | Manhole | Tổng box |
|---|---:|---:|---:|---:|---:|
| Train | 1,422 | 1,074 | 1,750 | 691 | 3,515 |
| Val | 276 | 66 | 470 | 103 | 639 |
| Test | 311 | 121 | 299 | 163 | 583 |

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

E1 tăng F1 từ **0.1077** lên **0.2806** (+0.1729), vượt yêu cầu +0.10 và hai CI không chồng lấn. Mở projector ở E2 tăng F1 thêm **0.0612** lên 0.3419, nhưng đổi lại +109,105,152 tham số trainable và +2.91 GB VRAM; CI E1/E2 còn chồng nhẹ nên bằng chứng cho phần tăng này chưa mạnh. E3 tăng thêm **0.0886** lên 0.4305, giảm FP/ảnh từ 1.1511 xuống 0.8103; CI E2 kết thúc ở 0.3835 còn CI E3 bắt đầu ở 0.3897, nên LoRA ViT cho cải thiện rõ nhất sau E1.

Về chi phí, E3 có 148,426,752 trainable và mất 66.1 phút. Đỉnh VRAM E3 là 7.47 GB, thấp hơn E1/E2 vì E3 dùng micro-batch 1, accumulation 8, còn E1/E2 dùng batch 2, accumulation 4; vì vậy không thể quy chênh lệch VRAM chỉ cho kiến trúc. Trong ba setup chính, E3 đáng chọn nhất theo chất lượng, nhưng tốn thêm khoảng 11.7 phút so với E2.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

CI95 bootstrap theo ảnh rộng từ **0.0557** ở E0 đến khoảng **0.0871 F1** ở E4 rank 4. Cụ thể: E1 rộng 0.0725, E2 0.0818, E3 0.0816, E4-r4 0.0871 và E4-r64 0.0854. Khoảng này phải bootstrap theo 311 ảnh như `score.py`; không được tính như thể 583 box độc lập vì các box trong cùng ảnh có tương quan. Do CI của hai E4 và E3 chồng lấn gần như toàn bộ, chênh lệch vài phần nghìn giữa chúng không đủ cho kết luận rank cao tốt hơn.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Ở E0, crack kém nhất với F1 0.0526 do 182 FP và 286 FN. Sau fine-tuning, pothole trở thành lớp kém nhất ở E1–E4; trên model tốt nhất E4 rank 64, F1 pothole là **0.3229**, thấp hơn crack 0.3492 và manhole 0.6844. Pothole có ít box train hơn crack (1,074 so với 1,750), biên và hình dạng biến đổi mạnh, dễ hòa vào texture mặt đường. Crack dài/mảnh vẫn khó định vị chính xác, còn manhole có hình học rõ nên cao nhất ở mọi setup. `report/evidence/prediction_examples.png` minh họa ba mẫu có TP và ba mẫu lỗi của E4 rank 64; ground-truth màu xanh, prediction màu đỏ.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

E4 giữ nguyên setup E3 và chỉ đổi LoRA rank: r=4/alpha=8 so với r=64/alpha=128. Rank 4 đạt F1 **0.4335**, CI95 [0.3900, 0.4771]; rank 64 đạt **0.4396**, CI95 [0.3989, 0.4843]. Chênh lệch chỉ **0.0061**, trong khi CI chồng lấn mạnh. Rank 64 cần 266,391,552 trainable, cao hơn rank 4 tới 147,456,000, thêm 1.15 GB VRAM và 2.4 phút. Vì vậy chưa có bằng chứng rank 64 đáng chi phí; rank 4 là lựa chọn hiệu quả hơn trong thí nghiệm này.

## Nhìn lại

Phần đã hoàn thành gồm split không rò rỉ, kiểm tra nhãn/thang tọa độ, E0–E3, E4 rank 4/rank 64, lưu đầy đủ prediction, score và adapter. E3/E4 chạy được khi giảm micro-batch còn 1, tăng gradient accumulation lên 8 và đặt `foreach=False` cho AdamW. Các lỗi thực tế đã gặp là: cấu hình skip module khiến lần đầu chỉ mở 54,579,200 tham số projector; `torchao 0.10.0` không tương thích PEFT; OOM do model inference cũ còn giữ VRAM; Colab hết hạn mức; và môi trường Windows gặp encoding `cp1252`/quyền thư mục tạm của pytest.

Giới hạn còn lại là tập test chỉ có 311 ảnh và các CI E3/E4 còn rộng, nên không thể kết luận chênh lệch nhỏ giữa các rank sẽ lặp lại trên dữ liệu mới. Ngoài ra, VRAM giữa E1/E2 và E3/E4 dùng micro-batch khác nhau nên chỉ là nhật ký thực nghiệm, không phải benchmark kiến trúc được kiểm soát hoàn toàn.
