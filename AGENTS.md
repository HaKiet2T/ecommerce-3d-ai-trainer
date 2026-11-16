# Workspace Rules: AI 3D CRM Engineering & Git Workflow

Quy chuẩn vận hành, quản lý mã nguồn và tự động hóa commit/push dành cho Agent chuyên trách `crm-3d-trainer` và các cộng sự AI làm việc trong repository `ecommerce-3d-ai-trainer`.

---

## 1. Quyền hạn và Trách nhiệm của Agent

Agent `crm-3d-trainer` có toàn quyền tự chủ và trách nhiệm quản lý phiên bản mã nguồn trong kho lưu trữ này:
- Tự động rà soát, đánh giá mức độ hoàn thiện của các cập nhật mã nguồn (pipeline, models, notebooks, serving API, configs, docker).
- Sắp xếp phân loại thay đổi (staging theo nhóm logic, tách biệt giữa sửa lỗi cốt lõi, bổ sung tính năng và cấu hình).
- Tự động tạo commit nguyên tử (atomic commit) theo chuẩn Conventional Commits và đẩy (push) lên nhánh Git thích hợp trên remote repository `origin`.

---

## 2. Tiêu chuẩn "Cập nhật Hoàn chỉnh" (Definition of Done - DoD)

Một cập nhật được xác định là "mới và hoàn chỉnh" để tiến hành commit và push khi thỏa mãn đồng thời 4 điều kiện:

1. **Tính đúng đắn kỹ thuật (Correctness & Runtime Verification)**:
   - Mã nguồn không chứa lỗi cú pháp, đủ import, không có hàm giả lập chưa hoàn thiện.
   - Các thuật toán toán học, đồ thị vi phân PyTorch (như Autograd, Grid Sampling, Marching Cubes, SDF computation) đã được kiểm chứng không gây crash hoặc runtime error.
2. **Bảo mật và Vệ sinh Kho lưu trữ (Security & Repository Hygiene)**:
   - Tuyệt đối không commit token nhạy cảm (`HF_TOKEN`, Kaggle API Key, Supabase Service Key, URL private).
   - Tuyệt đối không commit tệp nhị phân kích thước lớn (`*.pt`, `*.pth`, `*.bin`, `*.zip`, thư mục dataset hay checkpoint nặng > 10MB) vào Git tree. Toàn bộ trọng số và dữ liệu phải được loại trừ qua `.gitignore`.
3. **Tính đồng bộ cấu trúc (Architectural Consistency)**:
   - Các sửa đổi trong notebook Kaggle (`kaggle/notebooks/`) phải được đồng bộ tương thích với module tương ứng trong `src/` (như `models/`, `pipeline/`, `serving/`).
4. **Tài liệu & Chỉ số (Documentation & MLOps Metrics)**:
   - Mọi thay đổi về hàm mất mát hoặc kiến trúc mạng phải có giải thích rõ mục đích và thông số định lượng trong commit message hoặc tệp cấu hình.

---

## 3. Chiến lược Nhánh (Branching Strategy) do Agent Quản lý

Agent sắp xếp nhánh theo mô hình **Trunk-Based phát triển thích ứng AI**:

| Nhánh | Mục đích sử dụng | Hành vi của Agent |
| :--- | :--- | :--- |
| `main` | Nhánh gốc ổn định (Source of Truth). Chứa mã nguồn serving, pipeline đã kiểm chứng thực nghiệm, notebook chạy chuẩn. | Tự động commit và push trực tiếp khi mốc cập nhật hoàn chỉnh, sẵn sàng tích hợp với storefront Next.js. |
| `feat/<ten-tinh-nang>` | Nhánh tính năng mới đang trong quá trình phát triển nhiều bước (ví dụ: xuất TensorRT, Gaussian Splatting decoder). | Tạo nhánh, commit các bước hoàn thiện trung gian, sau khi kiểm chứng đạt chuẩn DoD thì merge vào `main` và push. |
| `exp/<ten-thu-nghiem>` | Nhánh thử nghiệm R&D giả thuyết mô hình mới (ví dụ: so sánh FlexiCubes vs Marching Cubes). | Lưu vết thử nghiệm độc lập, không làm ảnh hưởng nhánh `main`. |

---

## 4. Quy chuẩn Thông điệp Commit (Conventional Commits 1.0)

Mọi commit do Agent khởi tạo phải tuân thủ nghiêm ngặt định dạng:
`<type>(<scope>): <mô tả ngắn gọn bằng tiếng Anh hoặc tiếng Việt chuẩn mực>`

- **`feat`**: Tính năng mới (ví dụ: `feat(serving): add glb mesh export in fastapi endpoint`)
- **`fix`**: Sửa lỗi kỹ thuật (ví dụ: `fix(stage2): eliminate double-backward error in cuDNN grid sampler`)
- **`refactor`**: Tái cấu trúc mã nguồn không đổi hành vi (ví dụ: `refactor(core): modularize pipeline into src directory`)
- **`perf`**: Tối ưu hóa hiệu năng, GPU memory, tốc độ suy luận
- **`chore`**: Cấu hình môi trường, dependencies, workflow CI/CD, rule file
- **`docs`**: Cập nhật tài liệu kỹ thuật, hướng dẫn MLOps

---

## 5. Quy trình Thực thi Tự động khi có Cập nhật Mới

Khi Agent hoàn thành một sửa đổi hoặc nhận diện cập nhật hoàn chỉnh:
1. `git status -u`: Kiểm tra toàn bộ tệp thay đổi và tệp mới.
2. Kiểm tra lại `.gitignore` để đảm bảo không lọt tệp checkpoint/dữ liệu nhị phân.
3. `git add <cac-tep-lien-quan>`: Stage các tệp thuộc cùng một khối logic.
4. `git commit -m "<conventional-message>"`: Tạo commit với mô tả chi tiết.
5. `git push origin <nhanh>`: Đẩy trực tiếp lên remote repository để duy trì đồng bộ tức thời.
