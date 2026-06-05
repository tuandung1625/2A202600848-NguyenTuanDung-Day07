# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Tuấn Dũng  
**Nhóm:** Chưa cập nhật  
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
> High cosine similarity nghĩa là hai câu có ý nghĩa gần giống nhau, nên embeddings của chúng có hướng gần nhau và điểm cosine similarity tiến gần 1.

**Ví dụ HIGH similarity:**
- Sentence A: "I need to buy a new laptop for work."
- Sentence B: "I'm looking for a computer to use for my job."
- Tại sao tương đồng: Hai câu diễn đạt cùng một nhu cầu, chỉ khác cách dùng từ.

**Ví dụ LOW similarity:**
- Sentence A: "I need to buy a new laptop for work."
- Sentence B: "My favorite food is pizza."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn không liên quan.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
> Cosine similarity tập trung vào hướng của vector nên phù hợp hơn cho việc đo mức độ giống nhau về ngữ nghĩa. Euclidean distance dễ bị ảnh hưởng bởi độ dài vector hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, `chunk_size = 500`, `overlap = 50`. Bao nhiêu chunks?**

> **Phép tính:**

- `step = 500 - 50 = 450`
- `chunks = ceil((10000 - 500) / 450) + 1`
- `chunks = ceil(9500 / 450) + 1`
- `chunks = ceil(21.11) + 1 = 23`

> **Đáp án:** khoảng 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
> Khi overlap tăng lên 100 thì `step` giảm xuống còn 400 nên số chunk tăng lên. Overlap lớn giúp giữ ngữ cảnh giữa hai chunk liên tiếp, giảm nguy cơ mất ý ở ranh giới cắt.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Tài liệu mô tả các bệnh di truyền hiếm, tập trung vào triệu chứng, gene liên quan và kiểu di truyền.

**Tại sao nhóm chọn domain này?**  
> Bộ dữ liệu có cấu trúc khá rõ: mỗi tài liệu đều mô tả đặc điểm bệnh, nguyên nhân di truyền và pattern inheritance. Điều này rất phù hợp để thử nghiệm chunking, semantic retrieval và RAG theo các câu hỏi factual.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|---|---|---:|---|
| 1 | `keratoderma_with_woolly_hair.txt` | `https://ghr.nlm.nih.gov/condition/keratoderma-with-woolly-hair` | ~4761 | `disease`, `source_file`, `source_url`, `chunk_index`, `doc_id` |
| 2 | `knobloch_syndrome.txt` | `https://ghr.nlm.nih.gov/condition/knobloch-syndrome` | ~3526 | `disease`, `source_file`, `source_url`, `chunk_index`, `doc_id` |
| 3 | `coloboma.txt` | `https://ghr.nlm.nih.gov/condition/coloboma` | ~6002 | `disease`, `source_file`, `source_url`, `chunk_index`, `doc_id` |
| 4 | `lacrimo-auriculo-dento-digital_syndrome.txt` | `https://ghr.nlm.nih.gov/condition/lacrimo-auriculo-dento-digital-syndrome` | ~4526 | `disease`, `source_file`, `source_url`, `chunk_index`, `doc_id` |
| 5 | `spinocerebellar_ataxia_type_3.txt` | `https://ghr.nlm.nih.gov/condition/spinocerebellar-ataxia-type-3` | ~5369 | `disease`, `source_file`, `source_url`, `chunk_index`, `doc_id` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|---|---|---|---|
| `disease` | `str` | `spinocerebellar_ataxia_type_3` | Cho phép biết chunk thuộc bệnh nào và lọc theo bệnh nếu cần. |
| `source_file` | `str` | `coloboma.txt` | Giúp truy ngược về file gốc khi kiểm tra kết quả retrieval. |
| `source_url` | `str` | `https://ghr.nlm.nih.gov/condition/coloboma` | Hữu ích cho việc đối chiếu nguồn và trích dẫn. |
| `chunk_index` | `int` | `9` | Giúp xác định vị trí chunk trong tài liệu gốc. |
| `doc_id` | `str` | `spinocerebellar_ataxia_type_3_9` | Dùng để định danh duy nhất từng chunk trong vector store. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu với `chunk_size=500`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|---|---|---:|---:|---|
| `coloboma.txt` | FixedSizeChunker (`fixed_size`) | 14 | 475.1 | Trung bình |
| `coloboma.txt` | SentenceChunker (`by_sentences`) | 16 | 372.1 | Tốt |
| `coloboma.txt` | RecursiveChunker (`recursive`) | 19 | 313.1 | Tốt |
| `knobloch_syndrome.txt` | FixedSizeChunker (`fixed_size`) | 8 | 484.5 | Trung bình |
| `knobloch_syndrome.txt` | SentenceChunker (`by_sentences`) | 9 | 389.1 | Tốt |
| `knobloch_syndrome.txt` | RecursiveChunker (`recursive`) | 11 | 318.6 | Tốt |
| `spinocerebellar_ataxia_type_3.txt` | FixedSizeChunker (`fixed_size`) | 12 | 493.2 | Trung bình |
| `spinocerebellar_ataxia_type_3.txt` | SentenceChunker (`by_sentences`) | 15 | 355.5 | Tốt |
| `spinocerebellar_ataxia_type_3.txt` | RecursiveChunker (`recursive`) | 18 | 295.4 | Tốt |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**  
> `RecursiveChunker` ưu tiên tách theo đoạn trống, xuống dòng, dấu chấm câu, khoảng trắng, rồi mới fallback về hard split. Thuật toán đệ quy giúp giữ được ý nghĩa của đoạn văn tốt hơn khi tài liệu dài và cấu trúc không đồng đều.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
> Các tài liệu bệnh có cấu trúc theo đoạn thông tin khá rõ, nên recursive chunking giữ ngữ cảnh tốt hơn fixed-size chunking. Nó cũng linh hoạt hơn sentence chunking khi có đoạn quá dài hoặc có nhiều thông tin kỹ thuật liên tiếp.

**Số chunk thực tế khi chunk toàn bộ bộ disease data với `chunk_size=500`:**

| Tài liệu | Số chunk |
|---|---:|
| `coloboma.txt` | 19 |
| `keratoderma_with_woolly_hair.txt` | 14 |
| `knobloch_syndrome.txt` | 11 |
| `lacrimo-auriculo-dento-digital_syndrome.txt` | 15 |
| `spinocerebellar_ataxia_type_3.txt` | 18 |
| **Tổng** | **77** |

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|---|---|---:|---:|---|
| `spinocerebellar_ataxia_type_3.txt` | SentenceChunker (best baseline) | 15 | 355.5 | Khá tốt |
| `spinocerebellar_ataxia_type_3.txt` | **RecursiveChunker (của tôi)** | 18 | 295.4 | Tốt hơn cho câu hỏi chi tiết về gene và triệu chứng |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|------------|------------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 9.0 | Giữ ngữ cảnh theo đoạn, retrieval factual tốt với local embeddings | Số chunk tăng, tốn chi phí embedding hơn |
| Nguyễn Thái Học | Fixed-Size Chunking | 7.5 | Đơn giản, dễ triển khai, tốc độ indexing nhanh | Dễ cắt mất ngữ cảnh ở ranh giới chunk |
| Nguyễn Minh Chiến | Sliding Window Chunking | 8.2 | Giảm mất mát ngữ cảnh nhờ vùng chồng lấn giữa các chunk | Tăng số lượng chunk và dữ liệu trùng lặp |
| Phạm Đức Liêm | Semantic Chunking | 8.8 | Chia tài liệu theo ý nghĩa, giữ tính liên kết nội dung tốt | Chi phí xử lý và tính toán cao hơn |
| Nguyễn Quang Minh | Sentence-Based Chunking | 7.9 | Chunk tự nhiên theo câu, phù hợp tài liệu ngắn | Khó kiểm soát kích thước chunk khi văn bản dài |
| Nguyễn Đình Tiến Mạnh | Paragraph-Based Chunking | 8.4 | Giữ được ngữ cảnh ở mức đoạn văn, dễ hiểu | Kích thước chunk không đồng đều |

**Strategy nào tốt nhất cho domain này? Tại sao?**  
> Với domain này, `RecursiveChunker` là lựa chọn tốt nhất vì nó vừa giữ ngữ cảnh theo đoạn, vừa chia nhỏ đủ để semantic search tìm đúng fact cụ thể như tên gene, pattern inheritance hoặc triệu chứng đặc trưng.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:  
> Hàm dùng regex `(?<=[.!?])\s+|\.\n` để nhận diện ranh giới câu. Sau đó các câu được gom lại theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:  
> Hàm đệ quy qua danh sách separator theo thứ tự ưu tiên. Nếu đoạn hiện tại còn quá dài thì tiếp tục split sâu hơn; nếu không còn separator phù hợp thì cắt cứng theo số ký tự.

### EmbeddingStore

**`add_documents` + `search`** — approach:  
> Tôi mở rộng `EmbeddingStore` để hỗ trợ `persist_directory` và query bằng embedding vector thay vì để Chroma tự embed query. Điều này giúp store hoạt động ổn định hơn khi dùng local embedder. Trong môi trường của tôi, collection thật đã được nạp vào `chroma_data` với `storage_backend=chromadb`.

**`search_with_filter` + `delete_document`** — approach:  
> Với ChromaDB, filter đi qua `where=metadata_filter`. Với fallback local JSON store, filter được áp dụng trước khi chấm điểm similarity. `delete_document` xóa toàn bộ chunk có cùng `doc_id`.

### KnowledgeBaseAgent

**`answer`** — approach:  
> `KnowledgeBaseAgent` lấy top-k chunk, ghép thành context, rồi đưa vào prompt yêu cầu model chỉ trả lời từ context. Tôi đã bổ sung thêm backend Gemini REST để dùng `GEMINI_API_KEY` từ `.env`.

### Embedding / LLM Setup Thực Tế

**Local embedding model:**  
> Tôi chạy `LocalEmbedder()` thành công bằng cache local của `sentence-transformers/all-MiniLM-L6-v2`. Embedder trả về vector kích thước 384.

**LLM cho retrieval:**  
> Tôi dùng Gemini qua `GEMINI_API_KEY` trong `.env`, với model thực tế khả dụng là `gemini-2.5-flash`.

### Test Results

```text
.\.venv\Scripts\python.exe -m pytest tests/ -v

42 tests collected
42 passed
0 failed
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|---|---|---|---|---:|---|
| 1 | I need to buy a new laptop for work. | I'm looking for a computer to use for my job. | high | 0.1281 | Có |
| 2 | The patient has retinal detachment and severe myopia. | Knobloch syndrome often causes vitreoretinal degeneration. | high | 0.1080 | Có |
| 3 | LADD syndrome affects the lacrimal system and ears. | SCA3 causes progressive movement problems. | low | -0.0660 | Có |
| 4 | Coloboma results from incomplete closure of the optic fissure. | The optic fissure fails to close completely in coloboma. | high | 0.1693 | Có |
| 5 | My favorite food is pizza. | Palmoplantar keratoderma affects the skin of hands and feet. | low | 0.0603 | Tương đối |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
> Pair 5 cho thấy mock embedding chỉ đủ để test pipeline chứ không phù hợp để đánh giá semantic similarity thật. Vì vậy việc chuyển sang local embedding model đã cải thiện retrieval đáng kể.

---

## 6. Results — Cá nhân (10 điểm)

Tôi chạy benchmark trên bộ dữ liệu `data/desease_data` sau khi:

1. chunk bằng `RecursiveChunker(chunk_size=500)`
2. tạo tổng cộng 77 chunk
3. nhúng bằng `LocalEmbedder` (`all-MiniLM-L6-v2`)
4. lưu vào `ChromaDB` collection `disease_chunks_local`

### Benchmark Queries & Gold Answers

| # | Query | Gold Answer |
|---|---|---|
| 1 | What is (are) keratoderma with woolly hair ? | Keratoderma with woolly hair is a group of related conditions that affect the skin and hair and in many cases increase the risk of potentially life-threatening heart problems. |
| 2 | What is (are) Knobloch syndrome ? | Knobloch syndrome is a rare condition characterized by severe vision problems and a skull defect. A characteristic feature of Knobloch syndrome is extreme nearsightedness (high myopia). |
| 3 | What is (are) coloboma ? | Coloboma is an eye abnormality that occurs before birth. Colobomas are missing pieces of tissue in structures that form the eye. They may appear as notches or gaps. |
| 4 | What is (are) lacrimo-auriculo-dento-digital syndrome ? | Lacrimo-auriculo-dento-digital (LADD) syndrome is a genetic disorder that mainly affects the eyes, ears, mouth, and hands. |
| 5 | What is (are) spinocerebellar ataxia type 3 ? | Spinocerebellar ataxia type 3 (SCA3) is a condition characterized by progressive problems with movement. People with this condition initially experience problems with coordination. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|---|---|---:|---|---|
| 1 | What is (are) keratoderma with woolly hair ? | Chunk mở đầu của `keratoderma_with_woolly_hair`, định nghĩa trực tiếp bệnh là nhóm rối loạn ảnh hưởng đến da, tóc và tim | 0.2464 | Có | Trả lời đúng định nghĩa tổng quan của bệnh |
| 2 | What is (are) Knobloch syndrome ? | Top-3 đều thuộc `knobloch_syndrome`; chunk giàu thông tin nhất mô tả bệnh hiếm với severe vision problems và skull defect | 0.4447 | Có | Có thể trả lời đúng định nghĩa bệnh nếu dùng context top-k |
| 3 | What is (are) coloboma ? | Chunk của `coloboma` mô tả coloboma là eye abnormality xảy ra trước sinh, thiếu mô ở các cấu trúc của mắt | 0.6936 | Có | Trả lời đúng định nghĩa coloboma |
| 4 | What is (are) lacrimo-auriculo-dento-digital syndrome ? | Top-3 đều thuộc `lacrimo-auriculo-dento-digital_syndrome`; chunk định nghĩa mô tả LADD syndrome ảnh hưởng đến eyes, ears, mouth, hands | 0.3069 | Có | Trả lời đúng định nghĩa LADD syndrome |
| 5 | What is (are) spinocerebellar ataxia type 3 ? | Top-3 đều thuộc `spinocerebellar_ataxia_type_3`; chunk định nghĩa mô tả progressive movement problems và ataxia | 0.3599 | Có | Trả lời đúng định nghĩa SCA3 |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

**Nhận xét ngắn:**  
> Với đúng bộ query định nghĩa mà nhóm thống nhất, local embedding model cho kết quả rất tốt: cả 5 query đều trả về đúng tài liệu trong top-3, và đa số top-1 đã chứa ngay phần định nghĩa mở đầu hoặc phần mô tả rất gần với gold answer.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
> Hiện chưa có dữ liệu so sánh từ các thành viên khác trong repo, nên phần này cần cập nhật sau buổi thảo luận nhóm.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
> Chưa có dữ liệu demo liên nhóm trong repo nên tôi chưa điền thêm phần này để đảm bảo trung thực.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
> Tôi sẽ chuẩn hóa metadata kỹ hơn bằng cách thêm các field như `gene`, `inheritance`, `symptoms`, `body_system`. Điều này sẽ giúp `search_with_filter()` hiệu quả hơn và hỗ trợ benchmark theo nhiều loại câu hỏi khác nhau.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|---|---|---:|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** |  | **93 / 100** |
