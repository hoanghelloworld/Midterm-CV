import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import sys
import math

# Đặt encoding cho stdout để tránh lỗi khi in tiếng Việt
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class ObjectCounter:
    def __init__(self):
        self.image = None
        self.template = None
        self.drawing = False
        self.template_points = []
        self.start_x, self.start_y = -1, -1
        self.template_contour = None
        self.min_contour_area = 100  # Diện tích tối thiểu để lọc nhiễu
        
        # Tham số cơ bản
        self.canny_threshold1 = 30
        self.canny_threshold2 = 65
        self.similarity_threshold = 0.5
        self.area_ratio_min = 0.25
        self.area_ratio_max = 4.0
        self.dilation_iterations = 0
        
        # Tham số nâng cao
        self.detection_method = 0  # 0: Canny, 1: Sobel, 2: Laplacian, 3: DoG
        self.preprocess_method = 0  # 0: Standard, 1: CLAHE, 2: Morphological
        self.blur_kernel = 11
        self.morph_kernel = 3
        self.use_adaptive_threshold = False
        self.use_shape_context = False
        
        # Thêm tham số cho phát hiện màu
        self.use_color_matching = False
        self.color_weight = 0.3  # Trọng số cho độ tương đồng màu sắc
        self.template_color_histogram = None
        self.color_hist_size = [8, 8, 8]  # Kích thước bin cho histogram HSV
        self.color_similarity_threshold = 0.5  # Ngưỡng tương đồng màu sắc


    def select_image(self):
        # Sử dụng dialog để chọn file
        root = tk.Tk()
        root.withdraw()
        file_path = "Counting/mouse.jpg"
        
        # Chuyển đổi đường dẫn
        self.image = cv2.imread(file_path.replace("/", "\\"))
        if self.image is None:
            print("Khong the doc anh. Vui long thu lai.")
            return False
        
        print(f"Đã đọc ảnh từ: {file_path}")
        return True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.template_points = [(x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Vẽ đường khi di chuyển chuột
            img_copy = self.image.copy()
            if len(self.template_points) > 0:
                for i in range(len(self.template_points)-1):
                    cv2.line(img_copy, self.template_points[i], self.template_points[i+1], (0, 255, 0), 2)
                cv2.line(img_copy, self.template_points[-1], (x, y), (0, 255, 0), 2)
            cv2.imshow("Ve duong vien template", img_copy)
            self.template_points.append((x, y))
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Kết nối điểm cuối và điểm đầu
            if len(self.template_points) > 2:
                self.template_points.append(self.template_points[0])
                
                # Tạo mask từ các điểm đã vẽ
                mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                points = np.array([self.template_points], dtype=np.int32)
                cv2.fillPoly(mask, points, 255)
                
                # Lưu contour template
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    self.template_contour = contours[0]
                    
                # Hiển thị kết quả
                img_copy = self.image.copy()
                cv2.drawContours(img_copy, [self.template_contour], -1, (0, 255, 0), 2)
                cv2.imshow("Ve duong vien template", img_copy)

    def draw_template(self):
        if self.image is None:
            print("Vui long chon anh truoc.")
            return False
        
        cv2.namedWindow("Ve duong vien template")
        cv2.setMouseCallback("Ve duong vien template", self.mouse_callback)
        
        cv2.imshow("Ve duong vien template", self.image)
        print("Ve duong vien quanh vat the mau bang chuot trai.")
        print("Sau khi ve xong, nhan ENTER de tiep tuc, 'r' de ve lai, 'q' de thoat.")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and self.template_contour is not None:  # 13 là mã ASCII của phím Enter
                # Tính toán histogram màu cho template
                self.calculate_template_color_histogram()
                break
            elif key == ord('r'):
                # Reset template
                self.template_points = []
                self.template_contour = None
                img_copy = self.image.copy()
                cv2.imshow("Ve duong vien template", img_copy)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
        
    def calculate_template_color_histogram(self):
        """Tính toán histogram màu cho vùng template đã chọn"""
        if self.template_contour is None or self.image is None:
            return
            
        # Tạo mask từ contour template
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [self.template_contour], 0, 255, -1)
        
        # Chuyển ảnh sang không gian màu HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Tính toán histogram
        ranges = [0, 180, 0, 256, 0, 256]  # Phạm vi cho từng kênh H, S, V
        self.template_color_histogram = cv2.calcHist(
            [hsv], [0, 1, 2], mask, self.color_hist_size, ranges
        )
        
        # Chuẩn hóa histogram
        cv2.normalize(self.template_color_histogram, self.template_color_histogram, 0, 1, cv2.NORM_MINMAX)

    def create_parameter_window(self):
        # Tạo cửa sổ điều chỉnh tham số
        cv2.namedWindow("Basic Parameters")
        cv2.namedWindow("Advanced Parameters")
        cv2.namedWindow("Color Parameters")  # Thêm cửa sổ mới cho tham số màu sắc
    
        # Tham số cơ bản
        cv2.createTrackbar("Canny Threshold1", "Basic Parameters", self.canny_threshold1, 255, 
                          lambda x: self.update_parameter('canny_threshold1', x))
        cv2.createTrackbar("Canny Threshold2", "Basic Parameters", self.canny_threshold2, 255, 
                          lambda x: self.update_parameter('canny_threshold2', x))
        cv2.createTrackbar("Similarity Threshold (x100)", "Basic Parameters", int(self.similarity_threshold*100), 100, 
                          lambda x: self.update_parameter('similarity_threshold', x/100))
        cv2.createTrackbar("Min Area Ratio (x100)", "Basic Parameters", int(self.area_ratio_min*100), 100, 
                          lambda x: self.update_parameter('area_ratio_min', x/100))
        cv2.createTrackbar("Max Area Ratio (x100)", "Basic Parameters", int(self.area_ratio_max*100), 500, 
                          lambda x: self.update_parameter('area_ratio_max', x/100))
        cv2.createTrackbar("Dilation Iterations", "Basic Parameters", self.dilation_iterations, 10, 
                          lambda x: self.update_parameter('dilation_iterations', x))
                          
        # Tham số nâng cao
        cv2.createTrackbar("Edge Method", "Advanced Parameters", self.detection_method, 3, 
                          lambda x: self.update_parameter('detection_method', x))
        cv2.createTrackbar("Preprocess Method", "Advanced Parameters", self.preprocess_method, 2, 
                          lambda x: self.update_parameter('preprocess_method', x))
        cv2.createTrackbar("Blur Kernel", "Advanced Parameters", self.blur_kernel, 31, 
                          lambda x: self.update_parameter('blur_kernel', x if x % 2 == 1 else x + 1))
        cv2.createTrackbar("Morph Kernel", "Advanced Parameters", self.morph_kernel, 21, 
                          lambda x: self.update_parameter('morph_kernel', x if x % 2 == 1 else x + 1))
        cv2.createTrackbar("Adaptive Thresh", "Advanced Parameters", int(self.use_adaptive_threshold), 1, 
                          lambda x: self.update_parameter('use_adaptive_threshold', bool(x)))
        cv2.createTrackbar("Shape Context", "Advanced Parameters", int(self.use_shape_context), 1, 
                          lambda x: self.update_parameter('use_shape_context', bool(x)))
        
        # Thêm các thanh trượt cho tham số màu sắc
        cv2.createTrackbar("Use Color Matching", "Color Parameters", int(self.use_color_matching), 1,
                          lambda x: self.update_parameter('use_color_matching', bool(x)))
        cv2.createTrackbar("Color Weight (x100)", "Color Parameters", int(self.color_weight*100), 100,
                          lambda x: self.update_parameter('color_weight', x/100))
        cv2.createTrackbar("Color Similarity (x100)", "Color Parameters", int(self.color_similarity_threshold*100), 100,
                          lambda x: self.update_parameter('color_similarity_threshold', x/100))

    def update_parameter(self, param_name, value):
        # Cập nhật giá trị tham số
        setattr(self, param_name, value)
        # Thực hiện phát hiện và hiển thị lại với tham số mới
        self.detect_and_display()

    def preprocess_image(self, img):
        # Tạo bản sao để tránh thay đổi ảnh gốc
        processed = img.copy()
        
        # Bước 1: Chuyển sang ảnh xám
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Bước 2: Xử lý tiền xử lý ảnh theo phương pháp được chọn
        if self.preprocess_method == 0:  # Standard
            # Làm mờ cơ bản
            blur_size = max(3, self.blur_kernel)
            if blur_size % 2 == 0:
                blur_size += 1
            blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        elif self.preprocess_method == 1:  # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Tăng cường độ tương phản
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            blur_size = max(3, self.blur_kernel)
            if blur_size % 2 == 0:
                blur_size += 1
            blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            
        elif self.preprocess_method == 2:  # Morphological
            # Xử lý hình thái học
            kernel_size = max(3, self.morph_kernel)
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            blur_size = max(3, self.blur_kernel)
            if blur_size % 2 == 0:
                blur_size += 1
            blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Bước 3: Phát hiện cạnh theo phương pháp được chọn
        if self.detection_method == 0:  # Canny
            edges = cv2.Canny(blur, self.canny_threshold1, self.canny_threshold2, 3)
            
        elif self.detection_method == 1:  # Sobel
            sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
            
            # Tính toán độ lớn gradient
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            # Chuẩn hóa và chuyển đổi sang uint8
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude = np.uint8(magnitude)
            
            # Áp dụng ngưỡng để tạo ảnh nhị phân
            _, edges = cv2.threshold(magnitude, self.canny_threshold1, 255, cv2.THRESH_BINARY)
            
        elif self.detection_method == 2:  # Laplacian
            laplacian = cv2.Laplacian(blur, cv2.CV_64F)
            # Chuẩn hóa và chuyển đổi sang uint8
            laplacian = np.uint8(np.absolute(laplacian))
            # Áp dụng ngưỡng
            _, edges = cv2.threshold(laplacian, self.canny_threshold1, 255, cv2.THRESH_BINARY)
            
        elif self.detection_method == 3:  # DoG (Difference of Gaussians)
            blur1 = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
            blur2 = cv2.GaussianBlur(gray, (blur_size*2+1, blur_size*2+1), 0)
            dog = blur1 - blur2
            # Chuẩn hóa và chuyển đổi sang uint8
            dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
            dog = np.uint8(dog)
            # Áp dụng ngưỡng
            _, edges = cv2.threshold(dog, self.canny_threshold1, 255, cv2.THRESH_BINARY)
        
        # Bước 4: Áp dụng Adaptive Threshold nếu được chọn
        if self.use_adaptive_threshold:
            # Kết hợp với Adaptive Threshold để cải thiện
            adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)
            # Kết hợp kết quả của hai phương pháp
            edges = cv2.bitwise_or(edges, adaptive_thresh)
        
        # Bước 5: Áp dụng giãn nở
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=self.dilation_iterations)
        
        # Lọc nhiễu nhỏ
        if self.dilation_iterations > 0:
            # Áp dụng xói mòn nhẹ để loại bỏ nhiễu
            eroded = cv2.erode(dilated, kernel, iterations=max(1, self.dilation_iterations//2))
            # Kết nối các thành phần gần nhau
            dilated = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
        
        return dilated
    
    def match_contours(self, contours):
        matches = []
        template_area = cv2.contourArea(self.template_contour)
        
        # Tính toán các đặc trưng của template
        template_moments = cv2.moments(self.template_contour)
        template_hu_moments = cv2.HuMoments(template_moments)
        
        # Tính toán đặc trưng hình dạng bổ sung cho template
        template_perimeter = cv2.arcLength(self.template_contour, True)
        template_circularity = 4 * np.pi * template_area / (template_perimeter * template_perimeter) if template_perimeter > 0 else 0
        
        # Tính toán hình chữ nhật xoay nhỏ nhất cho template
        template_rect = cv2.minAreaRect(self.template_contour)
        template_width, template_height = template_rect[1]
        template_aspect_ratio = max(template_width, template_height) / min(template_width, template_height) if min(template_width, template_height) > 0 else 1
        
        # Chuẩn bị cho phân tích màu sắc nếu được kích hoạt
        hsv = None
        if self.use_color_matching and self.template_color_histogram is not None:
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Lọc contour theo diện tích
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
        
        for contour in filtered_contours:
            area = cv2.contourArea(contour)
            
            # Kiểm tra tỷ lệ diện tích
            area_ratio = area / template_area
            if area_ratio < self.area_ratio_min or area_ratio > self.area_ratio_max:
                continue
            
            # Tính toán Moments
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            # Tính toán Hu Moments
            hu_moments = cv2.HuMoments(M)
            
            # Tính toán tỷ lệ tương đồng hình dạng
            shape_similarity = cv2.matchShapes(self.template_contour, contour, cv2.CONTOURS_MATCH_I3, 0.0)
            
            final_score = shape_similarity  # Mặc định là chỉ dùng tương đồng hình dạng
            
            # Phân tích màu sắc nếu được kích hoạt
            if self.use_color_matching and self.template_color_histogram is not None and hsv is not None:
                # Tạo mask cho contour hiện tại
                mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                
                # Tính toán histogram màu cho contour hiện tại
                ranges = [0, 180, 0, 256, 0, 256]  # Phạm vi cho từng kênh H, S, V
                contour_histogram = cv2.calcHist([hsv], [0, 1, 2], mask, self.color_hist_size, ranges)
                cv2.normalize(contour_histogram, contour_histogram, 0, 1, cv2.NORM_MINMAX)
                
                # So sánh histogram
                color_similarity = cv2.compareHist(self.template_color_histogram, contour_histogram, cv2.HISTCMP_CORREL)
                
                # Nếu tương đồng màu quá thấp thì bỏ qua
                if color_similarity < self.color_similarity_threshold:
                    continue
                
                # Tính điểm tổng hợp (kết hợp hình dạng và màu sắc)
                # Với CONTOURS_MATCH_I3, giá trị thấp hơn = tương đồng cao hơn, 
                # nhưng với color_similarity thì giá trị cao hơn = tương đồng cao hơn
                shape_weight = 1.0 - self.color_weight
                normalized_shape_sim = 1.0 - min(shape_similarity, 1.0)  # Đảo ngược để cả hai đều cao = tốt
                
                final_score = (normalized_shape_sim * shape_weight) + (color_similarity * self.color_weight)
                
                # Chuyển final_score thành thước đo khoảng cách (thấp hơn = tốt hơn)
                # để tương thích với logic hiện tại
                final_score = 1.0 - final_score
            
            # Nếu sử dụng Shape Context (phương pháp cao cấp hơn)
            if self.use_shape_context:
                # Tính toán các đặc trưng bổ sung
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Tính toán hình chữ nhật xoay nhỏ nhất
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1
                
                # Kết hợp nhiều đặc trưng để đánh giá
                circularity_diff = abs(template_circularity - circularity)
                aspect_ratio_diff = abs(template_aspect_ratio - aspect_ratio)
                
                # Tạo trọng số ưu tiên nhiều thành phần
                combined_score = (
                    shape_similarity * 0.5 +
                    circularity_diff * 0.25 +
                    aspect_ratio_diff * 0.25
                )
                
                # Nếu đang sử dụng Color Matching, kết hợp với final_score
                if self.use_color_matching and self.template_color_histogram is not None:
                    combined_score = combined_score * 0.7 + final_score * 0.3
                
                final_score = combined_score
                
                # Đánh giá dựa trên điểm tổng hợp
                if final_score < self.similarity_threshold * 1.5:  # Điều chỉnh ngưỡng
                    matches.append(contour)
            else:
                # Sử dụng phương pháp đơn giản
                if final_score < self.similarity_threshold:
                    matches.append(contour)
        
        return matches


    def detect_and_display(self):
        """Hàm phát hiện vật thể và hiển thị kết quả theo tham số hiện tại"""
        if self.image is None or self.template_contour is None:
            return
        
        # Tiền xử lý ảnh
        processed = self.preprocess_image(self.image)
        
        # Tìm tất cả contour
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tìm các contour khớp với template
        matches = self.match_contours(contours)
        
        # Hiển thị kết quả
        result = self.image.copy()
        cv2.drawContours(result, matches, -1, (0, 255, 0), 2)
        
        # Vẽ hình chữ nhật bao quanh mỗi contour phù hợp
        for contour in matches:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Hiển thị ảnh đã tiền xử lý để dễ điều chỉnh
        processed_display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Vẽ tất cả các contour đã phát hiện lên ảnh tiền xử lý (màu đỏ)
        contours_display = processed_display.copy()
        cv2.drawContours(contours_display, contours, -1, (0, 0, 255), 1)
        
        # Đếm số lượng vật thể được tìm thấy
        count = len(matches)
        cv2.putText(result, f"Count: {count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Hiển thị tên phương pháp phát hiện cạnh
        edge_methods = ["Canny", "Sobel", "Laplacian", "DoG"]
        preprocess_methods = ["Standard", "CLAHE", "Morphological"]
        
        method_info = f"Edge: {edge_methods[self.detection_method]} | Preproc: {preprocess_methods[self.preprocess_method]}"
        if self.use_color_matching:
            method_info += f" | Color: ON (w={self.color_weight:.2f})"
        cv2.putText(result, method_info, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Hiển thị các giá trị tham số hiện tại
        param_info = (
            f"Similarity: {self.similarity_threshold:.2f} | "
            f"Area Ratio: {self.area_ratio_min:.2f}-{self.area_ratio_max:.2f} | "
            f"Dilation: {self.dilation_iterations}"
        )
        cv2.putText(result, param_info, (10, result.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hiển thị cả ảnh phát hiện và ảnh đã tiền xử lý
        cv2.imshow("Ket qua dem vat the", result)
        cv2.imshow("Preprocessed Image", processed_display)
        cv2.imshow("All Contours", contours_display)


    def detect_objects(self):
        if self.image is None or self.template_contour is None:
            print("Vui long chon anh va ve template truoc.")
            return
        
        # Tạo cửa sổ điều chỉnh tham số
        self.create_parameter_window()
        
        # Hiển thị lần đầu
        self.detect_and_display()
        
        print("Dieu chinh cac thong so tren thanh truot de tim ket qua tot nhat.")
        print("Cac phuong phap phat hien canh:")
        print("0: Canny - Phat hien canh co ban")
        print("1: Sobel - Phat hien dao ham bac nhat")
        print("2: Laplacian - Phat hien dao ham bac hai")
        print("3: DoG - Difference of Gaussians")
        print("Cac phuong phap tien xu ly:")
        print("0: Standard - Lam mo co ban")
        print("1: CLAHE - Tang cuong do tuong phan")
        print("2: Morphological - Xu ly hinh thai hoc")
        print("Tuy chon phat hien mau:")
        print("- Kich hoat 'Use Color Matching' de ket hop mau sac va hinh dang khi tim doi tuong")
        print("- 'Color Weight' dieu chinh muc do quan trong cua mau sac so voi hinh dang")
        print("- 'Color Similarity' dat nguong tuong dong mau sac toi thieu")
        print("Nhan 'q' de ket thuc, 's' de luu anh ket qua.")
        
        # Vòng lặp để giữ cửa sổ mở và xử lý sự kiện
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Lưu ảnh kết quả
                result_path = "object_detection_result.jpg"
                # Tạo lại ảnh kết quả
                result = self.image.copy()
                
                # Tiền xử lý ảnh
                processed = self.preprocess_image(self.image)
                
                # Tìm tất cả contour
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Tìm các contour khớp với template
                matches = self.match_contours(contours)
                
                # Vẽ lên ảnh kết quả
                cv2.drawContours(result, matches, -1, (0, 255, 0), 2)
                
                # Đếm số lượng vật thể được tìm thấy
                count = len(matches)
                cv2.putText(result, f"Count: {count}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Lưu ảnh
                cv2.imwrite(result_path, result)
                print(f"Da luu anh ket qua vao: {result_path}")
                
        cv2.destroyAllWindows()

    def run(self):
        if not self.select_image():
            return
        
        if not self.draw_template():
            return
            
        self.detect_objects()

if __name__ == "__main__":
    counter = ObjectCounter()
    counter.run()