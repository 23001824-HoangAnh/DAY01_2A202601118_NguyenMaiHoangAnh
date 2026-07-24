# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Qua bốn mức temperature (0.0, 0.7, 1.2, 1.8), quy luật rõ nhất là: temperature càng thấp thì phản hồi càng xác định và lặp lại (temp=0.0 luôn cho cùng một câu trả lời về "36 phố phường"), còn temperature càng cao thì model chọn chủ đề đa dạng hơn (Cầu Long Biên ở 0.7, Phố Đường Tàu ở 1.8). Ở mức 1.2 trở lên, phản hồi bắt đầu có dấu hiệu kém ổn định — đôi khi chọn từ ngữ bất thường hoặc lặp ý, tuy chưa hoàn toàn mất mạch lạc do GPT-4o xử lý khá tốt; nhưng ở mức 1.8, rủi ro sinh ra câu rời rạc hoặc thiếu logic tăng rõ rệt.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Với trợ lý soạn thảo hợp đồng pháp lý, tôi sẽ đặt temperature rất thấp (0.1–0.2) vì loại công việc này đòi hỏi tính chính xác và nhất quán tuyệt đối: sai lệch dù nhỏ trong điều khoản cũng có thể dẫn đến hậu quả pháp lý nghiêm trọng, do đó phải ưu tiên sự an toàn và khả năng dự đoán của mô hình thay vì sáng tạo. Ngược lại, với trợ lý viết slogan quảng cáo, tôi sẽ đặt temperature cao (0.7–1.0) để khuyến khích mô hình đưa ra nhiều phương án đa dạng, độc đáo và giàu hình ảnh; trong lĩnh vực sáng tạo, sự mới mẻ và khả năng gây ấn tượng với khách hàng quan trọng hơn tính chính xác từng từ.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
>Với 20.000 người dùng × 2 lượt/ngày × 500 token/lượt = 20 triệu token/ngày, model lớn (gpt-4o: $5/1M input, $15/1M output) tốn ~100 USD/ngày, trong khi model nhỏ (gpt-4o-mini: $0.25/1M input, $2/1M output) chỉ tốn ~5 USD/ngày — giảm 20 lần.
>
> Trường hợp model lớn xứng đáng: trợ lý chẩn đoán bệnh sử phức tạp đòi hỏi suy luận đa bước, mức độ sai sót thấp có thể ảnh hưởng tính mạng; trường hợp model nhỏ phù hợp: chatbot trả lời FAQ cho website thương mại điện tử, nơi ưu tiên tốc độ và chi phí trên quy mô lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona "nhà thơ", phản hồi ngắn (~80 từ), dùng hình ảnh ví von (hạt giống dữ liệu, khu vườn kỳ diệu, cây thông minh), hoàn toàn không có thuật ngữ kỹ thuật, giọng văn trữ tình và giàu cảm xúc. Với persona "kỹ sư senior", phản hồi dài hơn gấp đôi (~195 từ), có cấu trúc rõ ràng (đánh số, in đậm), dùng thuật ngữ chính xác (supervised learning, mạng nơ-ron, hồi quy tuyến tính) và sẵn sàng đưa ví dụ code. Từ đó rút ra: system prompt điều khiển được **giọng văn** (trữ tình vs kỹ thuật), **mức độ chi tiết và độ dài** (ngắn gọn vs đầy đủ), **mức kỹ thuật** (ví von vs thuật ngữ chuyên ngành), và **định dạng trình bày** (văn xuôi vs danh sách có cấu trúc). Nói cách khác, persona trong system prompt đóng vai trò như "bộ lọc phong cách" toàn diện cho toàn bộ phản hồi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Một đoạn văn tiếng Việt dài ~150 từ có khoảng 195 token theo tiktoken (150/0.75), tức là cao hơn 30% so với ước lượng thô. Nếu dùng ước lượng 0.75 để dự toán ngân sách API, tôi sẽ dự toán thiếu khoảng 30% cho các ứng dụng xử lý nhiều tiếng Việt, dẫn đến nguy cơ hết tiền giữa chừng khi lượng người dùng tăng lên.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> **(a) Chatbot văn bản** hưởng lợi nhiều nhất từ streaming: người dùng nhìn thấy từng từ hiện ra ngay lập tức thay vì chờ 3–5 giây nhìn màn hình trống, tạo cảm giác phản hồi tức thì và tự nhiên như đang trò chuyện thật — giảm đáng kể perceived latency (thời gian chờ cảm nhận). **(b) Trợ lý giọng nói** cũng hưởng lợi đáng kể vì module text-to-speech có thể bắt đầu đọc to ngay khi nhận được câu đầu tiên mà không cần đợi toàn bộ phản hồi hoàn thành, giúp người dùng nghe được phản hồi sớm hơn nhiều. **(c) Pipeline dịch tài liệu chạy ngầm ban đêm** hầu như không cần streaming: không có người dùng nào đang chờ đợi, hệ thống chỉ cần kết quả cuối cùng hoàn chỉnh để lưu file; streaming ở đây còn làm tăng thêm độ phức tạp xử lý (phải ghép chunk, quản lý lỗi giữa chừng) mà không mang lại lợi ích UX nào.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Với delay cố định (ví dụ luôn chờ 1 giây), khi API quá tải, hàng nghìn client sẽ đồng loạt retry vào đúng cùng một thời điểm (sau mỗi 1s), tạo ra các "đợt sóng" request lặp đi lặp lại khiến server không bao giờ kịp phục hồi. **Exponential backoff** giải quyết vấn đề này bằng cách tăng thời gian chờ theo cấp số nhân (0.1s → 0.2s → 0.4s → 0.8s…): client thất bại nhiều lần sẽ chờ lâu hơn, giảm dần áp lực lên server và cho phép server phục hồi tự nhiên. Tuy nhiên, exponential backoff thuần vẫn còn một vấn đề: nếu nhiều client bắt đầu lỗi cùng lúc, chúng sẽ tính ra cùng một delay và vẫn retry đồng thời (hiện tượng "thundering herd"). **Jitter** (thêm độ trễ ngẫu nhiên, ví dụ `delay = base_delay * 2^attempt + random(0, base_delay)`) giải quyết vấn đề còn sót này bằng cách phân tán các lần retry ra các thời điểm khác nhau, phá vỡ sự đồng bộ giữa các client và phân bổ tải đều hơn trên trục thời gian.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt đã dùng: **"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."** — Hai chỗ quan trọng: **(1) "trợ giảng thân thiện của khóa AI"** — nếu xóa cụm này, trợ lý mất vai trò cụ thể và sẽ trả lời như một chatbot đa năng chung chung; không còn ưu tiên giải thích kiến thức AI, không dùng giọng gần gũi kiểu thầy-trò, và có thể trả lời lạc sang các chủ đề không liên quan đến khóa học. **(2) "trả lời ngắn gọn"** — nếu xóa cụm này, trợ lý sẽ có xu hướng trả lời dài dòng, liệt kê chi tiết đầy đủ thay vì tóm tắt cô đọng; trong bối cảnh CLI chatbot, phản hồi dài sẽ khiến trải nghiệm người dùng kém đi vì phải cuộn màn hình nhiều và thời gian chờ streaming lâu hơn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> **Tình huống cụ thể:** Người dùng ở lượt 1 nói "Tên tôi là Minh, tôi đang học năm 3 ngành CNTT", sau đó hỏi tiếp 4 câu về các chủ đề khác nhau (neural network, backpropagation, optimizer, regularization). Đến lượt 6, người dùng hỏi "Dựa trên ngành học của tôi, hãy gợi ý lộ trình học AI phù hợp" — lúc này lượt 1 đã bị cắt khỏi history (chỉ giữ 4 lượt cuối = 8 message), nên trợ lý không biết người dùng tên Minh, đang học CNTT năm 3, và sẽ trả lời chung chung hoặc hỏi lại thông tin đã cung cấp trước đó. **Cách khắc phục:** Áp dụng kỹ thuật "sliding window + summary" — trước khi cắt các lượt cũ, dùng chính LLM để tóm tắt nội dung quan trọng (tên, ngành học, sở thích, yêu cầu đặc biệt) thành một đoạn ngắn ~100 token, rồi chèn đoạn tóm tắt này như một message "system" bổ sung ngay sau system prompt gốc. Như vậy, thông tin quan trọng được giữ lại mà không tốn nhiều token context.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
