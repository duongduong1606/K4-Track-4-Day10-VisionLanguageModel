# Bằng chứng hiện có

- `sample_boxes.png`: sáu ảnh ground-truth dùng để kiểm tra dữ liệu và cách vẽ box.
- `e0.json`–`e3.json`: kết quả các setup chính do `scripts/score.py` sinh từ đủ 311 prediction test.
- `e4_rank_4.json`, `e4_rank_64.json`: ablation chỉ thay đổi LoRA rank trên setup E3.
- `e0_examples.png`: sáu ảnh test với ground-truth màu xanh và prediction E0 màu đỏ.
- `prediction_examples.png`: sáu ảnh test đúng/sai của E4 rank 64, ground-truth màu xanh và prediction màu đỏ.
- `training_progress.json`: bản ghi tổng hợp tiến độ, tài nguyên và hash của artifact local.

Các JSON `e*.json` là bằng chứng chấm chính thức. `training_progress.json` chỉ là bản kiểm kê bổ sung. Adapter và prediction thô nằm trong `work/`, bị loại khỏi Git đúng yêu cầu không commit dữ liệu/trọng số.
