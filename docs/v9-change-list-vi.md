# v9 — Danh sách thay đổi (Google Doc v4, proposal, repo)

Ngày 07/10/2026. Nguồn chuẩn: `docs/research-proposal-v9-en.md` (repo) và `research-proposal-v9-en.md` (project).

## 1. Hướng chốt

| Mục | v8 / Doc v4 | v9 |
|---|---|---|
| Vấn đề lõi | 3 RQ: đo, cơ chế, sửa | 1 câu hỏi: SC có giữ được dự đoán và độ phủ conformal khi đổi ống nghe mà không cần nhãn của thiết bị mới không? |
| Giả thuyết | 7 | 3 giả thuyết thành một chuỗi: H3 (gain), rồi H1 (bất biến), rồi H2 (độ phủ). Thêm cổng G và 2 đối chứng dương |
| Nơi quyết định | KAUH + E1 | H1: phổi giả. H2: ICBHI LODO (prior-matched), phổi giả là đồng chính. H3: phổi giả. KAUH chỉ làm đối chứng dương + kiểm tra triển khai |
| Đóng góp | C1–C5 | C1: mô hình AuscultTrust. C2: SC và độ phủ trên thiết bị thật (H2). C3: kiểm chứng vật lý (H3, H1) |
| Mô hình đề xuất | Chọn bằng sàng lọc 10 biến thể | Họ SC cố định theo tên đề tài; CV chỉ chọn giữa P1 và P1P3 |
| Tiêu chí GĐ1 | Vượt P0 | Không kém P0: cận dưới CI của ΔScore > −1,5 điểm, 10 seed. Báo cáo benchmark vẫn dùng 5 seed |
| Sàng lọc | Không claim | Góc nhìn mức bản ghi: bình thường hoặc bất thường, kèm quy tắc chuyển khám |

**Vì sao KAUH bị hạ vai trò.**
- Ba bộ lọc của KAUH là phần mềm của hãng, áp lên cùng một bản ghi, và gần tuyến tính.
- SC ước lượng phổ của từng bộ lọc từ chính các bản ghi đó, nên gần như chắc chắn thành công.
- Nếu dùng KAUH làm bằng chứng chính, giám khảo hỏi một câu là đổ: "kết quả này đúng theo cách dựng thí nghiệm".
- Vì vậy KAUH giữ vai trò đối chứng dương, cộng với kiểm tra triển khai B1 (lấy ngưỡng từ ICBHI, áp sang KAUH).

## 2. Sửa Google Doc v4 theo mục

| Mục | Hiện tại | Sửa thành |
|---|---|---|
| Đoạn mở đầu ("ba câu hỏi") | 3 câu hỏi rời | 1 câu hỏi + chuỗi 3 giả thuyết (bảng §0 của v9) |
| Câu hỏi nghiên cứu (comment "need refinement") | RQ1–RQ3 | Câu hỏi chính + bảng H3/H1/H2 (cột: phép thử chính, đối chứng dương) + cổng G |
| Cách tiếp cận | 2 giai đoạn của anh | Giữ nguyên, bổ sung 5 điểm ở bảng ngay dưới |
| Đóng góp (comment "need refinements") | C1–C5 | C1–C3 |
| 1.2 | "SC cải thiện 56,67 lên 58,29" | Thêm điều kiện: co-tuning ResNet50; phổ tham chiếu lấy từ cả 4 thiết bị, kể cả thiết bị test; không có thiết bị chưa thấy; không đo độ phủ |
| 1.2 | — | Thêm [7] Ang và cs. 2023: 4 ống nghe số trên phổi giả, tương quan đáp ứng tần số giữa các cặp 0,14–0,91. Đây là bằng chứng thiết bị khác nhau thật, và là tiền lệ cho thiết kế phát lại |
| 1.4 | "ID 112, 1558, 218, 226" | 1558 thành 158 |
| 1.4 `<Thêm bảng stats ICBHI>` | Trống | Bảng theo thiết bị: chu kỳ train/test theo [8]; số bệnh nhân 64/23/11/32; AKGC417L giữ 1.543/1.864 crackle [14]; tần số lấy mẫu 4 kHz (90 bản ghi), 10 kHz (6), 44,1 kHz (824) [30]. Ghi chú: [14] đảo nhãn Litt3200 và LittC2SE; số đếm lại từ tên file còn là TODO |
| 1.4, P1–P4 | 4 vấn đề | 3 vấn đề: lỗi im lặng; giả định gain chưa kiểm; benchmark che lỗi theo thiết bị |
| 2. Khoảng trống | 8 hàng | 4 hàng. Hàng conformal sửa thành "Conformal cho âm hô hấp [25], [26]": [25] không nói về thiết bị |
| 3. Giả thuyết | 7 H + F1–F11 | G + H1–H3 + 2 đối chứng dương + F1–F5 |
| 4.1 Tầng dữ liệu | Có Tầng 2 | Bỏ Tầng 2 ("ngoài phạm vi trước 31/01/2027") |
| 4.1 KAUH | "Littmann 3200" không có nguồn | [31]: cùng model với Litt3200 của ICBHI; bộ lọc phần mềm; dải tần từng bộ lọc. "4 kHz" đổi thành [Giả định] |
| 4.2 Biến thể | P0–P9, R1, R2 | P0, P1, P1P3, P4. Biến thể khác đưa vào phụ lục nếu đã chạy |
| 4.2 Quy tắc | Top 3 + P0, G-base | Refit cả 4 arm với seed 0–4; seed 5–9 cho P0 và AuscultTrust (cổng G) |
| 4.2 Phép chia | "Không dùng chia có trùng bệnh nhân" | Ghi rõ: phép chia chính thức công bố có 2 bệnh nhân nằm cả hai phía. GĐ1 giữ nguyên để so sánh được; GĐ2 loại 2 bệnh nhân này |
| 4.3 Bảng E0–E3 | E0, E1, E2 (huấn luyện trên KAUH), E3 | E0 làm điều kiện tham chiếu. Phép thử 1: LODO trên toàn bộ bản ghi, 4 fold. Phép thử 2: KAUH A, B1, B2, C với mô hình đóng băng. Phép thử 3: phổi giả |
| 4.4 | A0–A3, T1, T2; V1–V5 | A0, A1 (SC), A2 (TTA); V1 chính; V5 k-shot làm dự phòng; prior-matched và V4-oracle |
| 4.5 Chỉ số | Decodability, φ | Flip rate, TV, thay đổi tập dự đoán, \|Δ\|, tỷ lệ chuyển khám. Cổng kiểm định bằng mô phỏng null, không dùng nửa độ rộng nhị thức |
| 4.6 Phổi giả | Cảm biến tham chiếu; chia theo clip | Tỷ số Y_B/Y_A, không cần cảm biến tham chiếu; chia và bootstrap theo bệnh nhân nguồn; ≥ 3 đầu thu, cho phép loại rẻ; công thức R² và trần nhiễu (§5.6 v9) |
| 4.7 Tầng 2 | Kế hoạch chi tiết | Xóa |
| 4.8 Cổng | G-base, G0–G9 | G, R1–R7 |
| 5. Đạo đức (trống) | — | Dữ liệu công khai và phổi giả; TT 06/2024 + 24/2025: ≤ 12 tháng đến 31/01/2027, sổ nhật ký, người hướng dẫn không làm phần lõi |
| 6. Mốc (trống ngày) | — | M0 09/10, M1 14/10, M2 17/10, M3 31/10, M4 30/11, M5 31/12, M6 31/01/2027 |
| 8. Phạm vi claims (trống) | — | Danh sách "Không claim" ở §0 của v9 |
| References [7] | Tác giả cần xác minh | Ang, Aw, Koh, Tan (2023), *Medicine in Novel Technology and Devices* 19, 100256 |

**5 điểm bổ sung cho mục "Cách tiếp cận":**

| # | Điểm | Nội dung |
|---|---|---|
| a | "Đóng băng" | KAUH và phổi giả dùng một mô hình cố định. LODO cố định công thức và số epoch, huấn luyện lại mỗi fold |
| b | "Cố định tham số hiệu chỉnh" | s_ref, giới hạn ±20 dB, α, ngưỡng conformal. Chỉ phổ của thiết bị đích được ước lượng, từ bản ghi không nhãn |
| c | LODO | Bệnh nhân có bản ghi trên thiết bị bị giữ phải rời khỏi tập huấn luyện. So sánh với P0 và P4 chạy lại cùng giao thức. [46] có LODO nhưng theo kiểu federated, 3 seed, chỉ báo Score. Thêm giao thức phụ theo đúng [46] để đặt số cạnh Bảng 4 của họ |
| d | KAUH | A = bất biến; B1 = ngưỡng ICBHI (kiểm tra triển khai); B2 = ngưỡng từ bộ lọc nguồn của KAUH (đối chứng dương); C = chạy lại với SC và A2 |
| e | TTA | Hiểu là A2 (chuẩn hóa thống kê đầu vào), không dùng Tent |

## 3. Đính chính và phản biện độc lập

**Đính chính lượt trước.**
- "57,3 không khớp anchors.md" là sai. Con số này khớp `report.md` §17.1 (cache layer 4); `anchors.md` F3 là bản layer 8 cũ.
- Endpoint là âm phổi bất thường mức bản ghi. Ca suy tim trong KAUH được giữ lại.

**Một agent độc lập đã soát v9 bản đầu và tìm ra 2 lỗi chết và nhiều lỗi lớn. Đã sửa trong v9 hiện tại:**

| Lỗi | Cách sửa |
|---|---|
| H1 và H2 đặt quyết định trên KAUH, nơi gần như đúng theo cách dựng thí nghiệm | KAUH thành đối chứng dương; quyết định chuyển sang phổi giả và LODO |
| Cổng G với 5 seed chỉ có khoảng 50% công suất khi SC thật sự trung tính | Dùng 10 seed: khoảng 0,9 với SD 0,9. Lề −1,5 điểm ≈ nửa khoảng cách AST-FT → Patch-Mix đã công bố |
| Arm so sánh chưa rõ | Đặt tên SC pipeline, baseline pipeline, và isolation contrast (P0 có và không có SC) |
| Bootstrap của \|Δ\| không ổn định; nửa độ rộng nhị thức bỏ qua nhiễu hiệu chuẩn và cụm bệnh nhân | Hiệu chuẩn lại trong mỗi lần bootstrap; resample cả seed; cổng null từ 200 lần chia lại hoán đổi được |
| `rotation_partitions` chỉ hiệu chuẩn trên khoảng 22 bệnh nhân | Hiệu chuẩn trên 4 nhóm còn lại |
| Trần nhiễu H3 chưa khớp cách đo | Công thức R² cụ thể; trần lấy từ lần tháo-đặt lại cùng đầu thu; quyết định bằng trung vị qua các cặp |
| Phổi giả chia theo clip | Chia và bootstrap theo bệnh nhân nguồn |
| Hiệu chuẩn trên phổi giả trái quy tắc Tầng 1 | Sửa `CLAUDE.md`: cho phép hiệu chuẩn nội bộ phổi giả |
| Số liệu [7] ghi sai | Sửa theo Bảng 3 của bài: 0,14–0,91 |
| [8] và [14] đảo nhãn hai thiết bị Littmann | Ghi rõ; repo xác nhận theo [8]; đếm lại từ tên file |
| Nhãn bản ghi ICBHI cho góc nhìn sàng lọc chưa định nghĩa | Bất thường nếu có bất kỳ chu kỳ crackle hoặc wheeze nào |

## 4. Repo

**Đã làm (chưa commit; anh quyết định commit):**
- `CLAUDE.md`: viết lại cho v9.
- `docs/research-proposal-v9-en.md`.
- `docs/prereg-v9-amendment-draft.md`: điền ngày, nối vào cuối `prereg-v8.md`, rồi commit.
- `configs/stage1_variants/P1P3_sc_gain.json`.

**Không đụng:** 5 file đang sửa dở của anh.

**Phải sửa trước khi sàng lọc:**

1. **SC chưa giới hạn hệ số.**
   - Tôi đo trên nhiễu tổng hợp: thiết bị băng 4 kHz so với thiết bị toàn dải cho gain +65 đến +70 dB trên 2,2 kHz.
   - ICBHI có 90 bản ghi 4 kHz; KAUH nhiều khả năng toàn bộ là 4 kHz.
   - Chọn một trong hai: giới hạn ±20 dB, hoặc dải chung 50–2.000 Hz. Ghi lựa chọn vào prereg.
2. **s_ref chưa được lưu cùng checkpoint.** Stage 2 cần nó.
3. **`stage1_summary.py screen` tự thêm P8 và chỉ giữ top 3 + P0,** nên có thể loại mất P4 hoặc một arm SC. Cần thêm chế độ v9 (TODO 3).
4. **Trong CV, s_ref được tính cả trên bệnh nhân của fold validation.** Lỗi nhỏ: sửa hoặc ghi chú.

**Thứ tự code:** TODO 1–10 trong `CLAUDE.md`. Trước 17/10 cần TODO 1–3 + bản tối thiểu của TODO 4–5 (suy luận trên KAUH, flip rate).

## 5. Đến 17/10 (vòng trường cần báo cáo kết quả)

| Ngày | Việc | GPU-h |
|---|---|---|
| 07–08/10 | Chọn clip hoặc dải chung, và danh sách seed cho TTA-EQ; sửa SC + lưu s_ref + test; sửa `stage1_summary`; đăng ký prereg | 0 |
| 08–10/10 | Sàng lọc P0, P1, P1P3, P4 × 3 fold | ≈ 12 |
| 10–12/10 | Refit 4 arm × seed 0–4 | ≈ 20 |
| 12–13/10 | Đóng băng, export; E0; KAUH-A (đối chứng dương, P0 so với AuscultTrust) | < 1 |
| 13–14/10 | Bảng GĐ1 (theo thiết bị, 4 và 2 lớp, cột văn liệu); hình hệ số SC | 0 |
| 15–16/10 | Slide 10 phút + poster | 0 |
| Sau 17/10 | Seed 5–9 (cổng G), LODO, KAUH B/C, phổi giả | ≈ 70 |

**Nếu thiếu GPU:** AuscultTrust = P1 (dự phòng đã đăng ký); P1P3 và P4 chạy sau 17/10.

## 6. Thế nào là "thành công"

Không thiết kế trung thực nào bảo đảm mọi giả thuyết đúng. Thiết kế này bảo đảm mỗi nhánh đều cho ra một kết quả báo cáo được, và cách diễn giải khi âm tính đã được đăng ký trước.

| Mức | Nội dung | Khả năng |
|---|---|---|
| Tối thiểu (17/10) | Bảng GĐ1 kèm Sp/Se theo thiết bị; mô hình đóng băng; đối chứng dương KAUH-A | Cao |
| Lõi (30/11) | Cổng G; H2 trên LODO so với P0 và P4 chạy lại; KAUH B1/B2/C; F1–F5 | Trung bình. H2 có thể ra "không phát hiện được" nếu LODO không có thiếu hụt vượt cổng null |
| Đầy đủ (31/12) | H1, H2, H3 trên phổi giả với ống nghe thật | Phụ thuộc phần cứng. Đầu thu rẻ giảm rủi ro |

**Rủi ro lớn nhất còn lại:** bằng chứng trên thiết bị thật trước khi có phổi giả chỉ đến từ LODO, mà LODO bị lẫn tỷ lệ lớp. Giảm rủi ro bằng cách đặt mua hoặc tự làm phổi giả và 3 đầu thu ngay trong tháng 10.

## 7. Cập nhật 07/10 (vẫn là v9): đã đọc [46], thêm nhánh TTA-EQ

### 7.1 Những gì [46] (Koo, Kim, Toikkanen, Kim; arXiv 2605.29862v2) thay đổi

| Điểm | Trước | Sau |
|---|---|---|
| Đã có LODO công bố chưa | "Chưa thấy" ([Giả định]) | [Fact]: có, nhưng là leave-out thiết bị kiểu federated, chỉ báo Score, 3 seed, không SD. Patch-Mix CL OOD: AKGC 51,32; Meditron 58,81; Yunting 52,88; LittC2SE 56,58; Litt3200 35,90 |
| Điểm mới của đề tài | Có LODO | Thu hẹp còn 3 điểm: độ phủ/calibration dưới dịch chuyển thiết bị; SC trên thiết bị chưa thấy; giao thức huấn luyện tập trung |
| Rủi ro SC (R4) | Trung bình | Cao. Trừ trung bình thiết bị (gần với SC) cho OOD tệ hơn FedAvg trên AKGC (32,59 so với 50,26; Sp 11,49) và Yunting (39,04 so với 62,20) |
| P1P3 | Biến thể | Có thêm lý do: nhiễu loạn phổ ngẫu nhiên (gain + GIN) cho lợi ích OOD lớn nhất trong ablation của [46] |
| Rủi ro mới R8 | — | SC thua các nhánh ngẫu nhiên. Nếu xảy ra thì báo như một phát hiện; H2 so SC với baseline, không so với nhánh tốt nhất |
| LODO | 1 giao thức | Thêm giao thức phụ theo [46]: bỏ mọi bệnh nhân dùng nhiều thiết bị; các fold AKGC, Meditron, Littmann gộp; chỉ để tham chiếu. Tốn thêm khoảng 15 GPU-h |
| Số đếm theo thiết bị | — | Bảng 1 của [46] cộng lại được 7.511 chu kỳ, khác 6.898: thêm lý do phải đếm lại từ tên file |

### 7.2 Nhánh TTA-EQ (trong F5, không phải giả thuyết)

- **Định nghĩa:** trung bình softmax qua K = 8 view; mỗi view qua một gain trơn ngẫu nhiên theo bin, lấy từ phân phối của P3 (6 dB SD), danh sách seed cố định.
- **Vai trò:** khởi động lạnh. Không cần thông tin thiết bị, chạy được từ bản ghi đầu tiên. SC thì cần biết ranh giới thiết bị và vài bản ghi không nhãn.
- **Chạy:** đơn lẻ và kết hợp với SC (SC + TTA-EQ), trên P0 và AuscultTrust, ở LODO, KAUH và phổi giả. Chỉ cần suy luận.
- **Conformal:** điểm hiệu chuẩn tính bằng cùng ensemble K view.
- **Đo thêm:** độ bất đồng giữa các view làm điểm báo dịch chuyển không cần nhãn. Tương quan hạng với |Δ| chỉ mang tính thăm dò.
- **Không làm được:** không khử đáp ứng riêng của thiết bị mới. Kỳ vọng: ít đổi nhãn hơn, không phải bảo đảm độ phủ.
- **Đã loại:** consensus qua nhiều tần số lấy mẫu. Resample chỉ là lọc thông thấp, khác biệt trong dải vẫn nguyên, và không có tác dụng trên dữ liệu 4 kHz như KAUH.

### 7.3 Sửa Google Doc v4 thêm

- **Mục 2:** thêm hàng [46] vào bảng khoảng trống. Đoạn "Nhóm gần nhất" thêm [46] (leave-out thiết bị, chỉ có Score; loại bỏ cố định không ổn định).
- **Mục 4.3:** thêm giao thức phụ của LODO theo [46].
- **Mục 4.4:** thêm hàng TTA-EQ và một dòng giải thích vì sao loại phương án nhiều tần số lấy mẫu.
- **Mục 4.8:** R4 nâng lên Cao; thêm R8.
- **References [46]:** điền tác giả, bản v2.
