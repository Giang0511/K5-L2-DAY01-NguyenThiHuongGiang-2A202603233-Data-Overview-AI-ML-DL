Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

Ngày chạy: 06/11/2026
Runtime Colab: CPU/GPU
Python / PyTorch / Ultralytics:
Checkpoint: `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`
Ngưỡng detection/segmentation dùng khi tạo JSON: `0.35` 
Thay đổi so với notebook nguồn: Không 

Bài 1. Phân loại ảnh

- Record hạng 1: 
`class_id`= 468 ; 
`class_name`= cab ; 
`rank`=1 ; 
`score`= 0.510915 ; 
`taxonomy_name`= ImageNet-1K
- Record này nó mô tả phân loại toàn bộ bước ảnh. Mô hình phân tích tổng thể hình ảnh và đưa ra nhãn có xác suất dự đoán cao nhất là car.
- Danh sách tên lớp được xác định bởi taxonomy và checkpoint của mô hình. Mô hình sử dụng taxonomy ImageNet-1K với các lớp được định nghĩa sẵn.
- Ghi đầy đủ giúp tránh nhầm lẫn giữa các bộ taxonomy khác nhau.
- Guideline cần quy định rõ tiêu chí lựa chọn nhãn đại diện cho toàn bộ ảnh, ví dụ ưu tiên vật thể chính, vật thể nổi bật nhất hoặc đối tượng chiếm diện tích lớn nhất. Người gán nhãn cần tuân thủ cùng một quy tắc để đảm bảo tính nhất quán.
- Score chỉ thể hiện mức độ tự tin của mô hình đối với dự đoán, không đảm bảo dự đoán đó chính xác. Ground truth phải được xác định dựa trên dữ liệu thực tế và quy trình gán nhãn theo guideline, có thể được kiểm tra bởi người gán nhãn hoặc reviewer.

Bài 2. Phát hiện vật thể

- Record: 
`class_name`= bowl ; 
`score`= 0.701283 ; 
`bbox_xyxy`=[155.02, 168.67, 182.38, 184.18] ; 
`bbox_width`= 27.36 ; 
`bbox_height`= 15.51
Góc trên-trái ở vị trí có tọa độ(155.02, 168.67),
Góc dưới-phải ở vị trí có tọa độ (182.38, 184.18), 
Cao 15.51 pixel,
Rộng 27.36 pixel
- Ngưỡng 0.35  có 11 prediction, ngưỡng 0.6 có 6 prediction
- Khi hạ ngưỡng confidence từ 0.60 xuống 0.35, số lượng prediction tăng từ 6 lên 11. Điều này giúp mô hình phát hiện được nhiều đối tượng hơn nhưng cũng có thể xuất hiện nhiều dự đoán sai hoặc không chính xác. Vì vậy, Reviewer cần: Kiểm tra tính chính xác của từng nhãn, Kiểm tra bounding box có bao quanh đúng vật thể không, Loại bỏ những prediction không phù hợp theo guideline.
- Bounding box phải bao quanh vật thể cần gán nhãn, sát với đường viền bên ngoài của vật thể. Không tự ý cắt bỏ phần vật thể còn nhìn thấy. Các trường hợp vật thể bị che khuất cần tuân thủ guideline thống nhất.
- Trên ảnh `kitchen`, có 2 box "oven" (lò nướng) với score 0.69 và 0.63 đang vẽ chồng lên nhau ở phía bên phải. Đây là trường hợp cần escalation (hỏi Lab Coach). Vì không rõ đây là một lò nướng lớn hay hai lò riêng biệt, và nếu là một vật thể thì annotator phải chọn box nào (box có score cao hơn hay box tight hơn). Việc tự ý quyết định sẽ gây ra sự không nhất quán (inconsistent) giữa các annotator khác nhau.

Bài 3. Phân đoạn theo từng đối tượng

- Record: 
`instance_id`= kitchen-001 ; 
`class_name`= person ; 
`score`= 0.899318 ; 
số điểm polygon=`polygon_point_count`= 348 ; 
4 điểm đầu của `polygon_xy`= [446.0, 70.0], [ 445.0, 71.0], [ 444.0, 71.0], [ 443.0, 72.0]
- Bounding Box cho biết vật thể nằm ở đâu và có kích thước bao nhiêu, còn Polygon cho biết vật thể có hình dạng như thế nào và đường biên thực tế của nó nằm ở đâu. Vì vậy, Polygon cung cấp thông tin chi tiết hơn về hình dạng và vùng pixel của đối tượng so với Bounding Box.
- Instance_id dùng để định danh riêng biệt từng đối tượng cụ thể (instance) trong một ảnh, giúp phân biệt các đối tượng khác nhau ngay cả khi chúng thuộc cùng một lớp
Nó không phải class ID và không phải tracking ID: có nghĩa là instance_id chỉ có nhiệm vụ phân biệt các cá thể đối tượng trong phạm vi dữ liệu được quy định, chứ không dùng để xác định loại đối tượng hoặc theo dõi đối tượng qua nhiều frame.
- Mask phải bám theo đường viền thực tế của đối tượng, không được lấn sang nền hoặc đối tượng khác.
- Trên ảnh 'kitchen', có hai chiếc bát nằm sát nhau ỏ góc dưới bên trái, phần đường viền của chúng khá gần và có thể gây khó khăn khi phân đoạn. Tuy nhiên, nếu vẫn có thể quan sát và phân biệt được hình dạng, đường biên của từng chiếc bát.

Bài 4. Vòng đời và kiểm tra chất lượng

- Phân loại ảnh:
    + Mỗi ảnh có nhãn thể hiện nội dung hoặc loại đối tượng chính. Ví dụ: ảnh giao thông có nhãn traffic.
    + Ảnh có nhiều loại phương tiện nên dễ nhầm lẫn giữa các lớp. Trong ảnh, mô hình dự đoán cab có độ tin cậy cao nhất khoảng 0.51, tiếp theo là minibus, police_van, recreational_vehicle, streetcar.
    + Quan sát toàn bộ ảnh, xác định nội dung chính và gán nhãn đúng theo guideline.
    + Kiểm tra nhãn có phù hợp với nội dung ảnh không, có nhầm lẫn hoặc thiếu nhãn không.

- Phát hiện vật thể:
    + Mỗi đối tượng có class_id, class_name và bounding box gồm 4 tọa độ (x1, y1, x2, y2).
    + Một số bounding box có thể chồng lấn hoặc bao chưa chính xác đối tượng. Trong ảnh kitchen, mô hình phát hiện nhiều đối tượng như person, bowl, oven, cup. Có những chiếc bát nằm sát nhau, dễ bị nhầm lẫn.
    + Xác định từng đối tượng, gán class và vẽ bounding box chính xác theo guideline.
    + Kiểm tra số lượng đối tượng, class và vị trí bounding box có chính xác không, có bỏ sót hoặc nhận diện nhầm vật thể không.

- Instance segmentation:
    + Mỗi đối tượng có class_id, class_name, instance_id, bounding box và polygon mask gồm nhiều điểm tọa độ (x, y).
    + Mask có thể bị lệch, lấn sang nền hoặc đối tượng khác. Trong ảnh kitchen, hai chiếc bát nằm sát nhau nên khó xác định đường biên riêng. JSON có instance_id = kitchen-001, class_name = person, score = 0.899318, polygon_point_count = 348.
    + Vẽ polygon mask riêng cho từng đối tượng, kiểm tra đường biên, không gộp hai đối tượng khác nhau thành một mask.
    + Kiểm tra mask có bám sát hình dạng vật thể không, polygon có hợp lệ không, các instance có bị gộp nhầm hoặc chồng lấn sai không.

Bài 5. An toàn dữ liệu
- Kiểm tra mask có bám sát hình dạng vật thể không, polygon có hợp lệ không, các instance có bị gộp nhầm hoặc chồng lấn sai không.
- Nếu phát hiện ảnh hoặc dữ liệu không đúng phạm vi, có chứa nội dung nhạy cảm, thông tin cá nhân hoặc nội dung bất thường, em sẽ dừng xử lý, không tự ý chỉnh sửa, sao chép hay chia sẻ dữ liệu đó, sau đó báo ngay cho Reviewer, Team Leader hoặc người phụ trách bài lab để được hướng dẫn xử lý.