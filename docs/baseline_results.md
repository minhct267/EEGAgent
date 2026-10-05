# Kết quả baseline Qwen3.8:27B trên TUEV

Ngày chốt số: 5 Oct 2026. Protocol: `tuev_authors_v2`. Planner: `qwen3.8:27b`, GGUF Q4_K_M, họ `qwen35`, 27.3B, Ollama `/api/chat`, `num_ctx` 65536 đã nạp, `think: false`, temperature 0.7, top_p 0.8, top_k 20, seed 0 trừ khi ghi khác. Embedding: `bge-m3:latest`.

Nguồn bài báo: Zhao et al., [arXiv:2511.09947v2](https://arxiv.org/html/2511.09947v2), mục *EEG Event Detection Experiment*. Nguồn tác giả gốc: git `acd2e6a:runs/tuev_agent`, chấm lại bằng `scripts/rescore_tuev.py`. Mọi file `metrics_v2.json` dưới đây có `sanity.coverage_hits_match_stored = true`.

## Cách đọc hai cột điểm

| Cột | Định nghĩa | Dùng để so với |
| --- | --- | --- |
| Coverage | Cùng kênh, phần giao phủ ít nhất 70% độ dài sự kiện chuẩn. Thời gian dự đoán thừa không bị phạt. | Scorer trong code tác giả đã phát hành (`acd2e6a`). |
| IoU > 0.7 | Báo cáo cùng kênh có IoU lớn nhất với sự kiện chuẩn vượt 0.7. | Câu trong bài báo. |

Bài báo viết: gộp sự kiện cách nhau dưới 1 giây, lớp dương là SPSW, GPED và PLED, một dự đoán đúng khi IoU cùng kênh vượt 0.7, hit rate **69.30%**, false rate **44.77%**. Bài không nêu số file, mẫu số của false rate, và cách cắt cửa sổ cho con số đó.

Coverage và IoU trên cùng một đầu ra cách nhau khoảng 39 điểm phần trăm. Một coverage gần 69% không phải là hit rate 69.30% của bài.

Khoảng tin cậy là bootstrap theo file, 1,000 lần, seed 0.

## Quy trình trong `baseline_run.md`

Các thư mục E1–E4 và mọi file rescore đều có. Bộ file và `total_gt` khớp bảng kiểm. Ba mục health không đạt dòng “kỳ vọng” của runbook.

| Bước | Kỳ vọng | Thực tế |
| --- | --- | --- |
| E1 oracle, 4 ngưỡng | `total_gt` 2736 | Đủ 4 thư mục, mỗi thư mục 159 file, 2,736 sự kiện |
| E2, 159 file | 512 cửa sổ, `errors` 0, context 65536, không overflow | 512 cửa sổ, context 65536, overflow 0, truncation 0, **3 ReadTimeout** |
| E2 manifest | protocol, stop `<RETURN>`, seed 0, `ollama_native` | Đủ. Git `6eed8ef`, **dirty** |
| E3 seed 1 và 2 | 36 file, 120 cửa sổ | Đủ. Timeout 4 và 1. Git sạch `9a81722` |
| E4, bốn ablation | 36 file, 447 sự kiện | Đủ, sanity đúng. `think medium` mất 64/120 cửa sổ vì timeout |
| Rescore | 159/2736, 36/447, sanity true | Đủ, kể cả `runs/rescore/e2_on_authors36` và `acd2e6a` |

Ba cửa sổ timeout của E2 nằm trên file có nhãn dương và được ghi với câu trả lời rỗng, nên chúng không cộng hit. Trong ba cửa sổ đó có 23 sự kiện dương (18 ở `gped_052_a_` 202.9–211.2 s, 2 ở `spsw_023_a_1`, 3 ở `pled_006_a_2`). Chạy lại chỉ có thể tăng tử số. Trần là 1,888/2,736 = 69.01% coverage nếu cả 23 sự kiện đều thành hit. Headline 68.17% là cận dưới của seed này, lệch tối đa 0.84 điểm.

110/113 cửa sổ `unparseable` nằm trên file không có sự kiện dương. Chúng không đổi coverage.

## Ba mốc cần đặt cạnh nhau

| Nguồn | Planner | File | Coverage | IoU > 0.7 | Báo cáo không khớp sự kiện dương |
| --- | --- | --- | --- | --- | --- |
| Bài báo | Qwen3-235B, theo lời bài | không nêu | bài không công bố coverage | **69.30%** theo lời bài | false rate **44.77%**, mẫu số không nêu |
| Tác giả phát hành, `acd2e6a` | `qwen3-235b-a22b` | 36 | 330/447 = **73.83%** (60.4–83.9) | 146/447 = **32.66%** (19.4–44.6) | 506/814 = 62.16% |
| Baseline này, cùng 36 file, seed 0 | `qwen3.8:27b` Q4_K_M | 36 | 334/447 = **74.72%** (64.7–84.4) | 137/447 = **30.65%** (19.0–42.3) | 525/841 = 62.43% |
| Baseline này, full eval | cùng planner, seed 0 | 159 | 1,865/2,736 = **68.17%** (59.7–74.5) | 800/2,736 = **29.24%** (22.7–35.5) | 2,173/4,268 = 50.91% |
| Local cũ, protocol khác | `qwen3.8:27b` | 159 | 1,422/2,736 = 51.97% (43.0–59.9) | 692/2,736 = 25.29% | 1,734/3,418 = 50.73% |
| Local cũ, cùng 36 file | cùng run cũ | 36 | 260/447 = 58.17% (45.5–68.9) | 122/447 = 27.29% | 423/694 = 60.95% |

Event F1 (precision theo báo cáo, recall theo coverage): full baseline **0.571**, cặp 36 file của cả 27B seed 0 và 235B đều **0.500**, local cũ full split 0.506.

IoU của tác giả đã phát hành là 32.66%. IoU của baseline này là 29.24% trên 159 file và 30.65% trên 36 file. Cả hai nằm quanh 30%, cách 69.30% của bài khoảng 37–40 điểm. Coverage 68.17% chỉ gần 69.30% về mặt con số, và chỉ trên full split, dưới một phép đo khác.

## 27B và 235B trên cùng 36 file

Seed 0 hơn run 235B bốn sự kiện coverage (334 so với 330) và kém chín sự kiện IoU (137 so với 146). Tỉ lệ báo cáo không khớp gần như cùng một số: 62.43% và 62.16%.

Ba seed của 27B trên đúng 36 file:

| Seed | Coverage | IoU > 0.7 | SPSW /50 | GPED /230 | PLED /167 | Timeout |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 334/447 = 74.72% | 137 = 30.65% | 26 | 191 | 117 | 0 trên subset này |
| 1 | 305/447 = 68.23% | 138 = 30.87% | 24 | 192 | 89 | 4 |
| 2 | 331/447 = 74.05% | 135 = 30.20% | 27 | 195 | 109 | 1 |
| 235B, `acd2e6a` | 330/447 = 73.83% | 146 = 32.66% | 33 | 190 | 107 | — |

Coverage 235B nằm trong dải seed 68.23–74.72%. GPED của 235B (190) sát dải 191–195. SPSW của 235B (33) nằm ngoài dải 24–27 ở cả ba seed. PLED của 235B (107) nằm trong dải 89–117; dải này rộng vì seed 1 rơi 28 hit PLED so với seed 0. Khoảng tin cậy 36 file rộng khoảng 20 điểm, nên việc hai khoảng chồng nhau là điều runbook đã dự kiến.

So sánh này không tách được kích thước model. Phần còn khác, đã ghi trong [baseline.md](baseline.md): sửa chỉ số kênh ở ba tool 1 giây, đổi key `Eyem movement` thành `Eye movement`, harness `authors_v2` thay cho splice XML của tác giả, Q4_K_M local thay cho API 235B, và BGE-M3 thay cho Qwen3-Embedding-8B mà bài nêu. Seed 0 của full split còn nằm trên commit dirty `6eed8ef`. Seed 1 và 2 nằm trên `9a81722` sạch, là con của commit đó.

## Theo lớp, full split

| Lớp | Sự kiện | Baseline 27B | Oracle `seizNormal` 0.5 | Oracle `seizNormal` 0.7 | Local cũ |
| --- | --- | --- | --- | --- | --- |
| SPSW | 216 | 111 (51.39%) | 143 (66.20%) | 117 (54.17%) | 112 (51.85%) |
| GPED | 1,633 | 1,263 (77.34%) | 1,343 (82.24%) | 1,271 (77.83%) | 936 (57.32%) |
| PLED | 887 | 491 (55.36%) | 538 (60.65%) | 482 (54.34%) | 374 (42.16%) |

Ba lớp cộng lại đúng tử số coverage. Phần tăng so với local cũ nằm ở GPED (+327 hit) và PLED (+117). SPSW gần như đứng yên (111 so với 112).

Trên 36 file, SPSW của baseline (26/50) thấp hơn cả 235B (33/50) lẫn local cũ (37/50). Prompt strict đưa SPSW lên 37/50 và kéo GPED xuống 157/230. Xem ablation bên dưới.

## Oracle: trần của detector 1 giây

Oracle gọi một tool trên đúng các cửa sổ đã làm tròn mà agent được hỏi, giữ mỗi giây có `seiz` đạt ngưỡng, rồi chấm bằng cùng scorer. Đây là trần của một luật cố định, không phải trần của mọi cách đọc EEG.

| Luật | Ngưỡng | Coverage | IoU > 0.7 | SPSW | GPED | PLED |
| --- | --- | --- | --- | --- | --- | --- |
| `seizNormal` | 0.5 | 2,024/2,736 = 73.98% | 30.85% | 66.20% | 82.24% | 60.65% |
| `seizNormal` | 0.7 | 1,870/2,736 = 68.35% | 29.75% | 54.17% | 77.83% | 54.34% |
| `seizArtiBckg` | 0.5 | 2,052/2,736 = 75.00% | 32.49% | 73.15% | 81.94% | 62.68% |
| `seizArtiBckg` | 0.7 | 1,921/2,736 = 70.21% | 31.58% | 63.43% | 78.32% | 56.93% |
| Agent, seed 0 | — | 1,865/2,736 = 68.17% | 29.24% | 51.39% | 77.34% | 55.36% |

Bootstrap ghép theo file của (hit agent − hit oracle) / `total_gt`, 1,000 lần, seed 0, 159 file, mẫu số từng file trùng nhau:

| So với | Chênh coverage | Khoảng 95% | File agent cao hơn / thấp hơn / hòa |
| --- | --- | --- | --- |
| `seizNormal` 0.5 | −5.81 điểm | −8.03 đến −3.97 | 5 / 41 / 113 |
| `seizNormal` 0.7 | −0.18 điểm | −2.15 đến +1.77 | 25 / 22 / 112 |
| `seizArtiBckg` 0.5 | −6.83 điểm | −10.58 đến −3.68 | 7 / 41 / 111 |
| `seizArtiBckg` 0.7 | −2.05 điểm | −5.30 đến +0.81 | 21 / 30 / 108 |

Ở ngưỡng 0.5, khoảng không chứa 0: agent thấp hơn detector. Ở ngưỡng 0.7, khoảng chứa 0: số hit coverage không tách được khỏi luật 0.7. Không có khoảng nào nằm hoàn toàn về phía agent.

90 file không có sự kiện dương là hòa cấu trúc, vì cả hai bên đều có 0 hit. Trong 69 file có nhãn, so với `seizNormal` 0.7 còn 22 file hòa và 47 file lệch. Phép này so số hit, không chứng minh cùng một sự kiện được bắt.

IoU của oracle (29.75–32.49%) nằm cùng vùng với IoU của agent (29.24%). Planner không biến detector 1 giây thành một hệ IoU 69%.

## Ablation, một yếu tố, 36 file, seed 0

Dải seed của protocol gốc: coverage 68.23–74.72% (305–334 hit), GPED 191–195, SPSW 24–27, PLED 89–117. PLED chiếm gần hết độ rộng của coverage.

| Run | Coverage | IoU | SPSW | GPED | PLED | Timeout | Đọc được gì |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E2 seed 0 | 74.72% | 30.65% | 26 | 191 | 117 | 0 | mốc |
| `authors_v1` | 11.41% (2.9–18.6) | 6.04% | 11 | 19 | 21 | 14 | ngoài mọi dải |
| `--prompt strict` | 67.11% (54.8–81.9) | 26.85% | 37 | 157 | 106 | 1 | GPED và SPSW ra khỏi dải chặt |
| `--rag off` | 69.57% (58.3–80.7) | 31.99% | 31 | 169 | 111 | 4 | coverage trong dải; GPED thì không |
| `--think medium` | 16.55% (8.7–23.6) | 7.83% | — | — | — | 64 | run hỏng |

`authors_v1` trả 13 cửa sổ có tuple, 93 `unparseable`, 14 rỗng, trên 120 cửa sổ. Splice XML không có stop, không retry câu rỗng, không forced final turn, và không có patch “continue” của run local cũ. 11.41% đo cái harness đó trên model này. Nó không đo lại được 58.17% cũ, và không phải điểm của 235B: 235B chạy splice trên một model khác.

`--prompt strict` đổi câu hỏi 0.50, chín tool và dòng `<NOTE>` cùng lúc. Coverage chỉ thấp hơn dải seed năm hit, trong khi khoảng 36 file vẫn chồng. Phần đứng ngoài dải là cơ cấu lớp: SPSW 37 so với 24–27, GPED 157 so với 191–195. Đó là một điểm vận hành khác, ước lượng bằng một seed và 50 spike.

`--rag off`: coverage 69.57% nằm trong dải seed. GPED 169 thấp hơn dải 191–195 khoảng 22 hit. SPSW 31 chỉ hơn dải 24–27 năm sự kiện. Một seed không tách nội dung retrieval khỏi nhiễu mẫu. Giả thuyết “đoạn văn usually >10 s đang chặn sự kiện ngắn” không được coverage tổng ủng hộ; nếu có tín hiệu thì nó nằm ở GPED, lớp chiếm đa số sự kiện.

`--think medium` đúng cờ `reasoning_effort=medium`, rồi 64/120 cửa sổ `ReadTimeout` ở 300 giây (31 tuple, 25 `unparseable`, 112 lần gọi planner). Điểm 16.55% không phải hiệu ứng của suy luận.

## Nhận xét

Hai người đọc độc lập xem cùng bảng số này, không xem bài của nhau. Một người chấm phương pháp, một người chấm nghĩa lâm sàng của các lớp EEG. Họ gặp nhau ở các câu dưới đây.

Baseline này là một hệ dùng tool, được hỏi đúng các khoảng đã lấy từ annotation, rồi chấm bằng coverage. Trên full split nó đứng ngang luật `seiz >= 0.7` của tool 1 giây, và thấp hơn luật 0.5 khoảng 6 điểm, với khoảng ghép cặp không chứa 0. Trên 36 file dùng chung, coverage và F1 đứng cạnh run 235B đã phát hành; GPED khớp từng sự kiện (191–195 so với 190); SPSW thì thấp hơn ở cả ba seed (24–27 so với 33).

Điểm mới so với local cũ (+16.2 điểm coverage trên full split, 51.97% lên 68.17%) là có thật dưới cùng scorer, và phần lớn đến từ GPED với PLED. Local cũ dùng câu hỏi khác, chín tool, client `/v1` không áp `num_ctx`, và patch continue. Không gán 16 điểm đó cho riêng kích thước model hay riêng harness.

SPSW, GPED và PLED là mã TUEV. Spike thường cỡ 1 giây. GPED và PLED là phóng điện chu kỳ. Câu hỏi lại nói “epileptic seizures”. Một cơn điện não theo nghĩa ACNS thường dài hơn, hoặc có tiến triển. Retrieval còn đưa vào các đoạn định nghĩa “usually >10 s”. Điểm 68% vì vậy là độ phủ các mã đã được chỉ khoảng thời gian, ở ngưỡng bảo thủ của một detector 1 giây. Nó không phải độ nhạy của một báo động cơn trên bản ghi không đánh dấu.

487 báo cáo trên 90 file không có mã dương là bằng chứng specificity sạch nhất trong gói này: file không có gì để một mark bám vào. 50.91% là tỉ lệ khác: 2,173/4,268 báo cáo không phủ sự kiện dương. False rate 44.77% của bài không có mẫu số, nên không xếp hạng với hai tỉ lệ này. Trên 36 file, cả 27B và 235B đều để khoảng 62% báo cáo không khớp.

Giao thức có giờ bắt đầu và giờ kết thúc trong câu hỏi. Nó bỏ qua tìm onset, độ trễ, số báo động trên mỗi giờ EEG âm tính, và những đoạn nằm giữa các annotation. Không có video, thuốc, trạng thái người bệnh, hay một hành động vòng kín. Đây không phải một thí nghiệm BCI.

## Những câu số liệu không nâng được

- 68.17% tái lập hit rate 69.30% của bài.
- 27B bằng 235B, hoặc thí nghiệm này đã cô lập số tham số.
- Agent vượt detector. Ở 0.5 thì detector cao hơn; ở 0.7 thì chưa tách được.
- `--think medium` cho thấy suy luận làm coverage giảm. Run đó chết vì timeout.
- 11.41% của `authors_v1` là điểm của bài báo hoặc của run 235B.

## Nguồn

| Số | File |
| --- | --- |
| Full baseline | `runs/tuev_authors_v2/metrics_v2.json`, `run_health.json`, `manifest.json` |
| 36 file, seed 0 | `runs/rescore/e2_on_authors36/metrics_v2.json` |
| Seed 1, seed 2 | `runs/tuev_authors_v2_36_seed1/`, `runs/tuev_authors_v2_36_seed2/` |
| Bốn ablation | `runs/tuev_ablate_harness_v1/`, `tuev_ablate_strict/`, `tuev_ablate_rag_off/`, `tuev_ablate_think_medium/` |
| Oracle | `runs/tuev_oracle/*/summary.json` |
| 235B phát hành | `runs/rescore/acd2e6a/metrics_v2.json` |
| Local cũ | `runs/tuev_agent_ollama/metrics_v2.json`, `runs/rescore/local_on_authors36/metrics_v2.json` |

Protocol và các khác biệt còn lại so với 235B: [baseline.md](baseline.md). Lệnh và bảng kiểm: [baseline_run.md](baseline_run.md).
