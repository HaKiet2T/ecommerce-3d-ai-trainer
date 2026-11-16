## 📝 Mô tả thay đổi (PR Description)
<!-- Tóm tắt ngắn gọn mục tiêu của PR và những thay đổi chính -->

- 

## 🎯 Phân loại PR
- [ ] `feat`: Tính năng mới (kiến trúc model, pipeline huấn luyện, serving)
- [ ] `fix`: Sửa lỗi (shape tensor, camera pose, OOM, logic dữ liệu)
- [ ] `perf`: Tối ưu hiệu năng (tốc độ inference, trích xuất mesh, memory footprint)
- [ ] `refactor`: Tái cấu trúc mã nguồn (modular hóa, tách package)
- [ ] `chore`: Cập nhật cấu hình, dependencies, script phụ trợ
- [ ] `exp`: Thử nghiệm siêu tham số hoặc hàm loss mới

## 🔬 Kiểm tra chất lượng mô hình (Model Verification)
- [ ] Đã chạy kiểm tra cú pháp Python (`py_compile`)
- [ ] Forward pass kiểm thử thành công trên dummy tensor hoặc 1 batch dữ liệu
- [ ] Các thông số hình học (SDF, Chamfer Distance, PSNR/SSIM) nếu có:
  - PSNR: `____ dB`
  - SSIM: `____`
  - Chamfer: `____`

## 📦 Quản lý Artifacts & Git Hygiene
- [ ] KHÔNG commit file binary nặng (`.glb`, `.obj`, weights `.pt`, `.safetensors`) vào repo
- [ ] Đã cập nhật file cấu hình tương ứng trong `configs/` (nếu có thay đổi hyperparams)
- [ ] Tiêu đề commit tuân theo quy chuẩn Conventional Commits

## 📸 Hình ảnh kiểm định trực quan (Visual Sanity Check)
<!-- Đính kèm ảnh render multi-view hoặc ảnh chụp 3D mesh nếu có thay đổi visual -->
