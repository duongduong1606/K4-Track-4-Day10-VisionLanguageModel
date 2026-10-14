# Bằng chứng hiện có

- `sample_boxes.png`: sáu ảnh ground-truth dùng để kiểm tra dữ liệu và cách vẽ box.
- `training_progress.json`: bản ghi trung gian về split, E1/E2 đã huấn luyện, E3 bị dừng và hash của artifact local.

`training_progress.json` **không phải** kết quả từ `scripts/score.py`. Tại thời điểm cập nhật, chưa có prediction test nên chưa tạo `e0.json`–`e3.json` và không công bố F1/CI95. Các adapter nằm trong `work/`, bị loại khỏi Git đúng yêu cầu không commit trọng số.
