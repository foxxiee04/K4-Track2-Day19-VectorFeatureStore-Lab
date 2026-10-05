# Reflection — Lab 19

**Tên:** Nguyễn Văn Sang
**Cohort:** A20-K4
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

**Kết quả trên Golden Set 50 queries:**
1. **Exact queries (BM25 thắng/hòa ~96.7%):** Chứa thuật ngữ kỹ thuật nguyên văn (verbatim tokens). BM25 khớp chính xác từ vựng qua TF-IDF mà không bị nhiễu ngữ nghĩa.
2. **Paraphrase queries (Semantic/Vector ưu thế):** Diễn đạt lại không chứa từ khóa gốc. Vector embeddings ánh xạ ngữ nghĩa tiềm ẩn trong không gian đa chiều, bắt trúng chủ đề mà BM25 bỏ sót.
3. **Mixed queries (Hybrid RRF thắng tuyệt đối 100.0%):** Kết hợp cả từ khóa chính xác và ngữ cảnh mở rộng. RRF ($k=60$) dung hòa tối ưu độ phủ (dense) và độ chuẩn xác (sparse).

**Khi nào KHÔNG dùng Hybrid:**
- **Dùng Pure BM25 khi:** Tìm kiếm mã lỗi, ID đơn hàng, tên hàm/biến code, SKU, log hệ thống có cấu trúc, hoặc hệ thống hạn chế tài nguyên CPU/RAM không thể chạy embedding model.
- **Dùng Pure Vector khi:** Dữ liệu phi văn bản (hình ảnh, âm thanh), tìm kiếm đa ngôn ngữ (cross-lingual), hoặc truy vấn ý tưởng trừu tượng không có từ khóa trùng lặp.

---

## Điều ngạc nhiên nhất khi làm lab này

RRF với công thức đơn giản $1/(k + \text{rank})$ lại giải quyết triệt để vấn đề lệch thang đo điểm số giữa BM25 và Cosine, giúp hybrid đạt 100% precision trên nhóm truy vấn mixed mà không cần học trọng số.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với:
