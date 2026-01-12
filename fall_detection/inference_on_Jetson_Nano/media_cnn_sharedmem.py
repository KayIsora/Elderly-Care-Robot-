import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import sys
import mediapipe as mp
import posix_ipc # Thư viện cho Shared Memory và Semaphores
import struct # Để đóng gói/giải nén dữ liệu cấu trúc
import mmap

# --- CẤU HÌNH SHARED MEMORY & SEMAPHORES ---
# Tên của vùng nhớ chia sẻ và các semaphores (phải giống với C++)
SHM_NAME = "/robot_shm"
SEM_PRODUCER_NAME = "/robot_sem_producer"
SEM_CONSUMER_NAME = "/robot_sem_consumer"

# Kích thước dữ liệu struct SharedData từ C++
# Tính toán chính xác: 4 int (4*4=16 bytes) + 1 double (8 bytes) + 1 bool (1 byte) = 25 bytes
# Nhưng có thể có padding trong struct. Để an toàn, hãy in sizeof(SharedData) từ C++.
# Ví dụ: printf("Size of SharedData: %zu\n", sizeof(SharedData));
# Giả sử kích thước là 32 bytes (thường là bội số của 8 để căn chỉnh tốt)
SHARED_DATA_SIZE = 32 # Đảm bảo giá trị này khớp với sizeof(SharedData) từ C++

# Format string cho struct.pack/unpack
# '<' là little-endian (phổ biến trên hầu hết các hệ thống)
# 'iiii' cho x, y, width, height (4 int)
# 'd' cho fall_probability (1 double)
# '?' cho data_ready (1 bool)
# '7x' để bù đắp padding nếu tổng là 32 (ví dụ: 16 + 8 + 1 = 25, 32 - 25 = 7)
# Bạn cần điều chỉnh '7x' nếu sizeof(SharedData) khác 32.
SHARED_DATA_FORMAT_STRING = '<iiii d ? 7x'

# --- CẤU HÌNH AI & VIDEO ---
# Đường dẫn video đầu vào. Có thể thay bằng '/dev/video0' cho camera thực.
# video_file = '/home/jetson/Desktop/Quy/ab/test.mp4'
# video_file = '/home/jetson/Desktop/Tung/fall_detection/5. Person Following Robot - Fast Motion - Blur Motion.mp4'
video_file = '/dev/video1' # Sử dụng camera ảo hoặc camera thực

output_video = 'fall_mediapipe_cnn_shm_fps.mp4' # Tên file video đầu ra
model_path = './final_cnn_model.h5' # Đảm bảo đường dẫn này đúng

# Kích thước frame được resize trước khi đưa vào MediaPipe và model
TARGET_AI_WIDTH = 320
TARGET_AI_HEIGHT = 240

# --- KHỞI TẠO TENSORFLOW & MEDIAPIPE ---
tf.compat.v1.disable_eager_execution()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Thiết lập GPU memory growth thành công.")
    except RuntimeError as e:
        print("Lỗi khi thiết lập GPU memory growth:", e)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Cấu hình MediaPipe Pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0, # Sử dụng mô hình đầy đủ để có độ chính xác cao hơn
    smooth_landmarks=True, # Bật làm mượt keypoint giữa các frame
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load model Keras
print("Loading model keras...")
model = keras.models.load_model(model_path)

# Hàm để trích xuất 21 keypoints từ MediaPipe Pose landmarks
# (giữ nguyên logic ánh xạ đã có)
def extract_mediapipe_keypoints(pose_landmarks, width, height):
    if not pose_landmarks:
        return [0] * 63 # Trả về 0 nếu không có landmarks

    lm = pose_landmarks.landmark
    coords = []

    def get_xyc(idx):
        if idx < len(lm):
            return [lm[idx].x * width, lm[idx].y * height, lm[idx].visibility]
        return [0, 0, 0]

    coords += get_xyc(mp_pose.PoseLandmark.NOSE.value)
    neck_x = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2.0 * width
    neck_y = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2.0 * height
    neck_c = (lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility + lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility) / 2.0
    coords += [neck_x, neck_y, neck_c]
    coords += get_xyc(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    coords += get_xyc(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    coords += get_xyc(mp_pose.PoseLandmark.RIGHT_WRIST.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_WRIST.value)
    midhip_x = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].x + lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2.0 * width
    midhip_y = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].y + lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2.0 * height
    midhip_c = (lm[mp_pose.PoseLandmark.LEFT_HIP.value].visibility + lm[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility) / 2.0
    coords += [midhip_x, midhip_y, midhip_c]
    coords += get_xyc(mp_pose.PoseLandmark.RIGHT_HIP.value)
    coords += get_xyc(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    coords += get_xyc(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_HIP.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_KNEE.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_EAR.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value)
    coords += get_xyc(mp_pose.PoseLandmark.LEFT_HEEL.value)
    coords += get_xyc(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value)
    coords += get_xyc(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value)
    return coords

# --- HÀM CHÍNH: Xử lý video và Shared Memory ---
def main_processor():
    shm = None
    sem_producer = None
    sem_consumer = None
    cap = None
    fd = -1 # Khởi tạo file descriptor cho shared memory

    try:
        # 1. Kết nối Shared Memory và Semaphores
        shm = posix_ipc.SharedMemory(SHM_NAME)
        fd = os.dup(shm.fd)
        data_buffer = mmap.mmap(fd, SHARED_DATA_SIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE)

        print(f"Connected to shared memory '{SHM_NAME}'")

        sem_producer = posix_ipc.Semaphore(SEM_PRODUCER_NAME)
        sem_consumer = posix_ipc.Semaphore(SEM_CONSUMER_NAME)
        print(f"Connected to semaphores '{SEM_PRODUCER_NAME}' and '{SEM_CONSUMER_NAME}'")

        # 2. Mở camera/video
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_file}")
            sys.exit(1)
        
        frame_width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps_video, (frame_width_orig, frame_height_orig))

        print(f"Processing video: {frame_width_orig}x{frame_height_orig} @ {fps_video:.2f} FPS")
        print(f"AI input resolution: {TARGET_AI_WIDTH}x{TARGET_AI_HEIGHT}")

        # Biến để tính toán tổng FPS của toàn bộ pipeline
        prev_total_frame_time = 0 
        frame_index = 0

        while True:
            # Ghi lại thời gian bắt đầu xử lý frame hiện tại (để tính tổng FPS)
            start_total_frame_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("End of video stream or camera disconnected.")
                break

            frame_index += 1
            
            # --- ĐỌC DỮ LIỆU TỪ C++ ---
            sem_producer.acquire() # Đợi C++ báo hiệu có dữ liệu mới

            data_buffer.seek(0)
            raw_data = data_buffer.read(SHARED_DATA_SIZE)
            
            try:
                current_x, current_y, current_width, current_height, \
                fall_probability_from_c, current_data_ready = struct.unpack(SHARED_DATA_FORMAT_STRING, raw_data)

                sem_consumer.release() # Báo hiệu cho C++ rằng Python đã đọc xong
            except struct.error as e:
                print(f"Error unpacking shared data: {e}. Raw data: {raw_data}")
                time.sleep(0.01)
                sem_consumer.release()
                continue

            # --- TIỀN XỬ LÝ VÀ XỬ LÝ AI TRÊN TOÀN BỘ FRAME ---
            # Ghi lại thời gian bắt đầu xử lý AI (bao gồm resize, mediapipe, cnn)
            start_ai_process_time = time.time()

            processed_frame = cv2.resize(frame, (TARGET_AI_WIDTH, TARGET_AI_HEIGHT), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
        
            keypoints_proc = [0] * 63
            prob = 0.0

            if results.pose_landmarks:
                keypoints_proc = extract_mediapipe_keypoints(results.pose_landmarks, TARGET_AI_WIDTH, TARGET_AI_HEIGHT)
                
                for j in range(0, 63, 3):
                    keypoints_proc[j] /= TARGET_AI_WIDTH
                    keypoints_proc[j + 1] /= TARGET_AI_HEIGHT

                x_input = np.array(keypoints_proc).reshape((1, 3, 21, 1))
                
                pred = model.predict(x_input, verbose=0) 
                prob = float(pred[0][0])
            else:
                prob = 0.0

            # Ghi lại thời gian kết thúc xử lý AI
            end_ai_process_time = time.time()
            ai_process_duration = end_ai_process_time - start_ai_process_time
            ai_process_fps = 1 / ai_process_duration if ai_process_duration > 0 else 0

            # --- GHI DỮ LIỆU XÁC SUẤT VÀO SHARED MEMORY ---
            updated_data_for_shm = struct.pack(SHARED_DATA_FORMAT_STRING, 
                                            current_x, current_y, current_width, current_height, 
                                            prob, True) 

            data_buffer.seek(0)
            data_buffer.write(updated_data_for_shm)
            data_buffer.flush()

            # --- HIỂN THỊ KẾT QUẢ TRÊN FRAME GỐC ---
            # if current_width > 0 and current_height > 0:
            #     cv2.rectangle(frame, (current_x, current_y, current_width, current_height), (255, 0, 0), 2) # Blue for C++ bbox

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

            label = f"Fall Prob: {prob:.2f}"
            text_color = (0, 0, 255) if prob > 0.8 else (0, 255, 0)
            cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

            if prob > 0.8:
                cv2.putText(frame, "⚠️ FALL DETECTED", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Tính toán và hiển thị tổng FPS (của toàn bộ pipeline)
            end_total_frame_time = time.time()
            total_frame_duration = end_total_frame_time - start_total_frame_time
            total_fps = 1 / total_frame_duration if total_frame_duration > 0 else 0

            # Hiển thị FPS trên frame
            total_fps_label = f"Total FPS: {total_fps:.2f}"
            ai_fps_label = f"AI Process FPS: {ai_process_fps:.2f}"
            cv2.putText(frame, total_fps_label, (30, frame_height_orig - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, ai_fps_label, (30, frame_height_orig - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            out.write(frame)
            cv2.imshow('Fall Detection (Python) - Full Frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(f"Processed frame {frame_index}, C++ BBox: ({current_x},{current_y},{current_width},{current_height}), Fall Prob (Python): {prob:.2f}, AI FPS: {ai_process_fps:.2f}, Total FPS: {total_fps:.2f}")

    except posix_ipc.ExistentialError:
        print(f"Error: Shared memory '{SHM_NAME}' or semaphores not created by C++ program. Ensure the C++ program is running first.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # --- DỌN DẸP TÀI NGUYÊN ---
        if cap:
            cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        if 'data_buffer' in locals() and data_buffer:
            try:
                data_buffer.close()
            except Exception as e:
                print(f"Error closing mmap buffer: {e}")

        if fd != -1:
            try:
                os.close(fd)
            except Exception as e:
                print(f"Error closing duplicated fd: {e}")

        if shm:
            try:
                shm.close_fd()
            except Exception as e:
                print(f"Error closing posix_ipc SharedMemory fd: {e}")
                
        if sem_producer:
            try:
                sem_producer.close()
            except Exception as e:
                print(f"Error closing producer semaphore: {e}")
        if sem_consumer:
            try:
                sem_consumer.close()
            except Exception as e:
                print(f"Error closing consumer semaphore: {e}")
        
        print("Cleaned up IPC resources.")
        print(f"[✅] Xử lý hoàn tất. Video đã lưu tại: {output_video}")

if __name__ == "__main__":
    main_processor()