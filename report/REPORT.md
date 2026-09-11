# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**
11/09/2026
**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**
Python 3.13.15 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không 

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
  `class_id=468`, `class_name="cab"`, `rank=1`, `score=0.510915`, `taxonomy_name="ImageNet-1K"`
- Record này mô tả toàn ảnh như thế nào?
Tóm tắt toàn bộ nội dung ảnh đường phố `traffic` thành một nhãn duy nhất là "cab", với độ tin cậy ~51%. Nhãn này không chỉ ra vị trí của bất kỳ xe cụ thể nào trong ảnh, không đếm số lượng xe, và không phân biệt các loại xe khác nhau đang xuất hiện cùng lúc (xe buýt, xe con, xe van...) — chỉ là một phán đoán duy nhất cho tổng thể ảnh.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Class list gồm 1000 lớp đến từ tập dữ liệu ImageNet-1K — tập dữ liệu mà checkpoint `yolo11n-cls.pt` đã được huấn luyện trên đó. Model chỉ có thể chọn ra lớp có xác suất cao nhất trong số 1000 lớp cố định này; nó không tự tạo ra tên lớp mới ngoài danh sách đã học.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
`class_name` (ví dụ "cab") là tên hiển thị, có thể trùng hoặc mang nghĩa khác nhau giữa các
bộ dữ liệu/taxonomy khác nhau. `class_id` (468) mới là định danh chính xác, duy nhất trong đúng phiên bản taxonomy được ghi rõ ở `taxonomy_name` ("ImageNet-1K"). Giữ đủ cả ba giúp
người review truy vết được nhãn về đúng nguồn gốc, tránh nhầm lẫn khi so sánh kết quả giữa các model hoặc phiên bản taxonomy khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
VD ảnh `traffic`: có hàng chục xe khác loại (xe buýt, xe con, xe van) cùng xuất hiện, và điểm số của 3 lớp đứng đầu khá sít sao (cab 0.51, minibus 0.16, police_van 0.09). Guideline cần quy định rõ tiêu chí chọn "chủ thể chính" khi ảnh có nhiều đối tượng (ví dụ: đối tượng chiếm diện tích lớn nhất, nằm giữa khung hình, hoặc là chủ thể theo ý đồ chụp), và quy định cách xử lý khi không xác định được chủ thể chính — nên đánh dấu ảnh là "ambiguous" để chuyển sang review thủ công thay vì mặc định chấp nhận nhãn rank=1 của model.
- Vì sao model score không phải ground truth?
Score 0.510915 chỉ phản ánh mức độ tự tin nội bộ của model, chưa từng được con người xác nhận hay đối chiếu. Khoảng cách sát sao giữa rank 1 (0.51) và rank 2 (0.16) cho thấy bản thân model cũng không chắc chắn tuyệt đối. Ground truth chỉ được xác lập khi có annotator/guideline xác nhận, còn score cao vẫn có thể sai nếu ảnh nằm ngoài phân bố dữ liệu huấn luyện hoặc có nhiều chủ thể gây nhập nhằng như ảnh này.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  `class_name="person"`, `score=0.912624`, `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]`,
  `bbox_width=113.58`, `bbox_height=279.68`.

- Diễn giải vị trí box bằng lời:
Đây là người đứng cạnh bếp trong ảnh `kitchen` (640×427px). Box nằm lệch về phía bên phải của ảnh và kéo dài gần hết chiều cao khung hình, từ gần đỉnh ảnh xuống gần đáy. Vì là người đang đứng nên box có dạng đứng, cao gấp gần 2.5 lần chiều rộng (279.68 với 113.58) 

- So sánh số prediction ở hai threshold:
Hạ ngưỡng xuống 0.2 thì model bắt được tới 17 vật thể: `person×2, bowl×5, oven×2, cup×2,spoon×3, potted plant, dining table, bottle` — kể cả những thứ nhỏ và mờ như thìa hay chậu cây.Đẩy ngưỡng lên 0.6 thì chỉ còn lại 6 vật thể `person×2, bowl×2, oven×2` 
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Ngưỡng thấp giúp không bỏ sót vật thể (recall cao) nhưng đổi lại reviewer phải ngồi xem và xác nhận nhiều box hơn hẳn, trong đó có không ít box có thể là dự đoán sai hoặc chưa chắc chắn. Ngưỡng cao thì ngược lại: nhàn cho reviewer vì ít box phải kiểm tra, nhưng rủi ro là bỏ sót thật những vật nhỏ hoặc bị che khuất một phần — như mấy cái thìa hay chậu cây biến mất hoàn toàn khỏi danh sách khi lên ngưỡng 0.60.

- Đề xuất một quy tắc box chặt: Box nên ôm sát đúng rìa ngoài cùng của vật thể, không để dư khoảng trống xung quanh. Ví dụ với người trong ảnh trên, box phải dừng ngay tại mép vai/tay chứ không lấn thêm vài pixel "cho an toàn" — vì làm vậy sẽ khiến box lấn sang cả phần bếp lò ở ngay bên cạnh.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? 
Cần có quy định rõ ràng cho hai tình huống hay gặp: vật thể bị cắt bởi mép ảnh thì chỉ vẽ box đến đúng biên ảnh, không đoán phần nằm ngoài khung; còn vật thể bị vật khác che một phần thì cần đặt ra ngưỡng "che khuất bao nhiêu % thì vẫn coi là gán nhãn được bình thường" — che nhẹ thì annotator tự quyết, còn che nặng thì nên đánh dấu để escalate cho người review quyết định thay vì để mỗi người tự đoán ranh giới theo cảm tính.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):`instance_id="kitchen-001"`, `class_name="person"`, `score=0.899318`, `bbox_xyxy=[385.45, 66.44,498.02, 348.58]`, `polygon_point_count=348`, vài điểm đầu: `[446.0, 70.0], [445.0, 71.0], [444.0,71.0], [443.0, 72.0]...`. 

- Polygon bổ sung chi tiết gì so với box?
Box chỉ là một khung chữ nhật, bên trong nó lẫn cả người lẫn nền phía sau như bếp lò hay mảng tường. Polygon thì khác hẳn — với tận 348 điểm, nó ôm sát theo đúng dáng người, từng đường cong ở vai, ở tay, ở form dáng đứng. Nhờ vậy mới tách được rạch ròi pixel nào là người, pixel nào là nền, chứ box thì chịu, không làm được việc đó.
- `instance_id` dùng để làm gì và không phải loại ID nào?
Dùng để phân biệt các đối tượng riêng lẻ trong cùng một tấm ảnh — vd "kitchen-001", "kitchen-002"... Giả sử ảnh có hai người thì mỗi người vẫn có một mã riêng dù cùng chung"person". Nó không phải `class_id`, nó chỉ có ý nghĩa trong phạm vi một ảnh tĩnh, sang ảnh khác là đánh số lại từ đầu.
- Đề xuất một quy tắc biên mask: Đường viền nên bám sát rìa thật của vật thể đến từng pixel một, kể cả những chỗ lõm nhỏ như khoảng hở giữa cánh tay và thân người. Không nên làm mượt đường viền cho đẹp vì làm vậy vô tình nuốt luôn phần nền lọt vào các khe đó, khiến mask trông phồng to hơn vật thể thật.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Mask chỉ vẽ cho phần thực sự nhìn thấy được, không đoán mò phần bị che sau vật khác. Ảnh này có mấy cái bowl trên bàn nằm chồng lấn lên nhau (kitchen-002 và kitchen-006 có vùng bbox giao nhau ở khoảng x=31–86, y=344–370), và mask của bàn ăn (kitchen-005) tuy có tới 598 điểm nhưng bị vỡ thành nhiều mảnh rời rạc vì đồ vật đặt trên mặt bàn cắt ngang qua. Với những chỗ này nên đặt ngưỡng tin cậy tối thiểu để chấp nhận, còn nếu vẫn mơ hồ thì escalate cho reviewer quyết định thay vì để mỗi annotator tự vẽ theo cảm tính của riêng mình.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho toàn ảnh (class_name theo ImageNet-1K) | Ảnh `traffic` có cả chục xe khác loại cùng lúc, nên top-5 của model gồm toàn xe gần giống nhau (cab, minibus, police_van...) với điểm số sít sao — khó nói đâu mới là "đúng" nhất | Xem toàn bộ ảnh, chọn ra một nhãn duy nhất mô tả đúng nhất chủ thể chính, theo tiêu chí guideline đã định trước | Đối chiếu nhãn được chọn với ảnh gốc, kiểm tra xem có đúng chủ thể chính không hay annotator bị phân tâm bởi vật phụ |
| Phát hiện vật thể | Một hộp bao (bbox) + nhãn lớp cho mỗi vật thể, theo taxonomy COCO-80 | Box người ở góc trái ảnh `kitchen` gần như bị cắt cụt bởi mép ảnh (x_min = 0.12); nhiều `bowl` nhỏ nằm khuất sau nhau nên dễ vẽ thiếu hoặc vẽ chồng lấn | Vẽ một box riêng cho từng vật thể nhìn thấy được, dừng đúng ở mép ảnh nếu vật thể bị cắt, không đoán phần khuất | Kiểm tra từng box có ôm sát vật thể không, có bỏ sót vật thể nào rõ ràng không, và box có lấn sang vật thể khác không |
| Instance segmentation | Một polygon/mask riêng cho từng đối tượng, có instance_id phân biệt dù cùng lớp | Mask bàn ăn (kitchen-005, 598 điểm) bị vỡ thành nhiều mảnh rời do đồ vật trên mặt bàn cắt ngang; các bowl trên bàn thì chồng lấn bbox lên nhau nên ranh giới khó phân định | Vẽ đường viền sát theo hình dạng thật của từng vật thể, chỉ tô phần nhìn thấy được, không làm mượt ranh giới cho "đẹp" | Kiểm tra đường viền có ôm sát vật thể không, có bị "phình" ra ăn lấn sang nền hoặc vật kế bên không |
## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Notebook chỉ dùng đúng 3 ảnh COCO công khai đã cố định sẵn, mỗi ảnh đều được xác minh bằng SHA-256 trước khi dùng — không được tự ý tải ảnh cá nhân, ảnh khách hàng, ảnh có khuôn mặt hay biển số xe, hoặc bất kỳ dữ liệu nội bộ nào lên Colab hay đẩy lên GitHub công khai.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Mentor / Lab Coach phụ trách lớp — dừng ngay việc chạy tiếp thay vì tự ý xử lý hay xóa dữ liệu.


## 6. Danh sách bằng chứng

- [X] `classification_predictions.json`
- [X] `detection_predictions.json`
- [X] `segmentation_predictions.json`
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS`.
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
