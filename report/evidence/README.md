# Bằng chứng hiện có

- `sample_boxes.png`: sáu ảnh ground-truth dùng để kiểm tra dữ liệu và cách vẽ box.
- `e0.json`: kết quả E0 do `scripts/score.py` sinh từ đủ 311 prediction test.
- `e0_examples.png`: sáu ảnh test với ground-truth màu xanh và prediction E0 màu đỏ.
- `training_progress.json`: bản ghi trung gian về split, E1/E2/E3 đã huấn luyện và hash của artifact local.

`training_progress.json` **không phải** kết quả từ `scripts/score.py`. Tại thời điểm cập nhật đã có `e0.json`; E1–E3 chưa inference test nên chưa có `e1.json`–`e3.json`. Các adapter nằm trong `work/`, bị loại khỏi Git đúng yêu cầu không commit trọng số.
