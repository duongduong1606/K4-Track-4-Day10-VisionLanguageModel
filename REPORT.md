Họ tên: Dương Dương
`dataset_version`: `d10-21e4070301`

> Trạng thái báo cáo ngày 15/10/2026: đã chuẩn bị dữ liệu, kiểm tra trực quan, chạy E0 và huấn luyện xong E1/E2/E3. E0 đã được chấm trên đủ 311 ảnh test; E1–E3 chưa chạy inference test nên chưa có F1/CI95. Các ô chưa chạy được ghi rõ, không nội suy kết quả.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | zero-shot | 0 | — | — | 0.1077 | [0.0812, 0.1369] | 0.8360 | 0.1032 | 0.0526 | 0.2231 |
| E1 | LoRA LLM | 33,030,144 | 10.88 | 53.8 | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm |
| E2 | LoRA LLM + projector | 142,135,296 | 13.79 | 54.4 | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm |
| E3 | E2 + LoRA ViT | 148,426,752 | 7.47 | 66.1 | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm | Chưa chấm |
| E4 | Chưa bắt đầu | — | — | — | — | — | — | — | — | — |

Nguồn điểm E0 là `report/evidence/e0.json`; nguồn số liệu huấn luyện E1/E2/E3 là các file `work/adapters/e*/run.json`. Bản tóm tắt có thể commit nằm ở `report/evidence/training_progress.json`. Adapter/trọng số nằm trong `work/` và được `.gitignore` loại khỏi Git đúng quy định.

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

Hiện chỉ có thể so chi phí huấn luyện, chưa thể kết luận “đáng” về chất lượng vì E1–E3 chưa có prediction test. E2 tăng trainable từ **33,030,144** lên **142,135,296** (+109,105,152), VRAM từ **10.88** lên **13.79 GB** (+2.91 GB), trong khi thời gian tăng từ **53.8** lên **54.4 phút** (+0.6 phút). E3 có **148,426,752** trainable (+6,291,456 so với E2), mất **66.1 phút** và ghi nhận đỉnh **7.47 GB**. Con số VRAM E3 thấp hơn vì run này dùng micro-batch 1 và accumulation 8, còn E1/E2 dùng batch 2 và accumulation 4, nên không phải phép so sánh cùng cấu hình. Chỉ sau khi có F1 và CI95 E1–E3 mới có thể đánh đổi mức cải thiện với chi phí này.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

Với E0, CI95 bootstrap theo ảnh là **[0.0812, 0.1369]**, rộng khoảng **0.0557 F1**. Khoảng này phải bootstrap theo 311 ảnh như `score.py`; không được tính như thể 583 box độc lập vì các box trong cùng ảnh có tương quan. Khi có E1–E3, chỉ nên khẳng định setup tốt hơn nếu mức tăng đủ lớn so với độ rộng và mức chồng lấn CI; chênh lệch F1 nhỏ trong các CI chồng nhau chỉ là tín hiệu, chưa phải bằng chứng chắc chắn.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

Ở E0, **crack kém nhất với F1 0.0526** (13 TP, 182 FP, 286 FN), sau đó là pothole 0.1032 và manhole 0.2231. Crack thường dài, mảnh, biên không rõ và dễ lẫn với vạch/vết mặt đường, phù hợp việc vừa bỏ sót nhiều vừa tạo nhiều false positive. Kết luận này mới áp dụng cho E0; cần chấm E1–E3 để biết fine-tuning có đổi thứ tự lớp hay không. `report/evidence/e0_examples.png` minh họa ba mẫu có TP và ba mẫu lỗi, với ground-truth màu xanh và prediction E0 màu đỏ.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

E4 chưa bắt đầu vì E0–E3 chưa được chấm và chưa xác định setup nền tốt nhất. Không có kết quả ablation để kết luận ở tiến độ hiện tại.

## Nhìn lại

Phần đã hoàn thành gồm chuẩn bị split không rò rỉ, kiểm tra nhãn/thang tọa độ bằng hình, chạy và chấm E0, hoàn thiện pipeline LoRA, lưu adapter E1, và lưu cả LoRA lẫn projector của E2/E3. E3 cuối cùng chạy được khi giảm micro-batch còn 1, tăng gradient accumulation lên 8 và đặt `foreach=False` cho AdamW. Các lỗi thực tế đã gặp là: cấu hình skip module khiến lần đầu chỉ mở 54,579,200 tham số projector; `torchao 0.10.0` không tương thích PEFT; E3 từng OOM trên T4; Colab hết hạn mức; và môi trường Windows gặp encoding `cp1252`/quyền thư mục tạm của pytest.

Giới hạn lớn nhất của báo cáo hiện tại là mới có `e0.json`; E1–E3 đã train nhưng chưa inference/chấm, và E4 chưa bắt đầu. Bước tiếp theo là chạy inference test bằng ba adapter đã lưu, tạo `e1.json`–`e3.json`, rồi chọn setup tốt nhất để chạy E4.
