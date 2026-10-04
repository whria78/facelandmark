import os
import json
import time
import shutil  
import cv2            
import numpy as np    
from datetime import datetime
from LesionMapper_v1 import FaceRegionMapper, LesionMapper 
import mediapipe as mp

REQUIRED_MEDIAPIPE_VERSION = "0.10.21"

if mp.__version__ != REQUIRED_MEDIAPIPE_VERSION:
    error_message = (
        f"\n[Version Error] Mediapipe version mismatch detected!\n"
        f"  - Current installed version : {mp.__version__}\n"
        f"  - Required version          : {REQUIRED_MEDIAPIPE_VERSION}\n\n"
        f"This program is optimized for Mediapipe version {REQUIRED_MEDIAPIPE_VERSION}.\n"
        f"To prevent malfunctions caused by version mismatch, please run the following command to match the version:\n\n"
        f"  [Installation Command]\n"
        f"  pip3 install mediapipe=={REQUIRED_MEDIAPIPE_VERSION}\n"
    )
    raise ImportError(error_message)


region_mapper = FaceRegionMapper()
local_mapper = LesionMapper()

# ==========================================
# Configuration Variables
# ==========================================
TARGET_DIR = "./result"  

SAVE_DEBUG_IMAGES = True   # Whether to render and save debug images
DRAW_DEBUG_TEXT = False    # Whether to display text overlay on top of debug images
DRAW_TILT_RATIO = False    # Whether to display tilt and area ratio text in STEP1

# ==========================================
# Utility Function 1: Draw dashed rectangle
# ==========================================
def draw_dashed_rectangle(img, pt1, pt2, color, thickness=2, dash_length=8):
    x1, y1 = pt1
    x2, y2 = pt2
    
    def draw_line(pA, pB):
        dist = np.linalg.norm(np.array(pA) - np.array(pB))
        if dist == 0: return
        dashes = max(1, int(dist / dash_length))
        for i in range(dashes):
            if i % 2 == 0:  
                start_pos = (int(pA[0] + (pB[0] - pA[0]) * i / dashes), int(pA[1] + (pB[1] - pA[1]) * i / dashes))
                end_pos = (int(pA[0] + (pB[0] - pA[0]) * (i + 1) / dashes), int(pA[1] + (pB[1] - pA[1]) * (i + 1) / dashes))
                cv2.line(img, start_pos, end_pos, color, thickness)
                
    draw_line((x1, y1), (x2, y1))
    draw_line((x2, y1), (x2, y2))
    draw_line((x2, y2), (x1, y2))
    draw_line((x1, y2), (x1, y1))

# ==========================================
# Utility Function 2: Dynamic resizing and multi-line text rendering
# ==========================================
def draw_smart_text(img, text, pos, max_width=None, base_scale=1.0, color=(0, 255, 0), thickness=2):
    h, w = img.shape[:2]
    font_scale = max(0.4, (w / 1000.0) * base_scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if max_width is None:
        max_width = w - pos[0] - 20 
        
    words = text.split(' ')
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        test_line = current_line + ' ' + word
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        
        if text_size[0] <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)
    
    x, y = pos
    line_height = cv2.getTextSize("Tj", font, font_scale, thickness)[0][1]
    line_spacing = int(line_height * 1.5)
    
    current_y = y + line_height
    
    for line in lines:
        cv2.putText(img, line, (x, current_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, current_y), font, font_scale, color, thickness, cv2.LINE_AA)
        current_y += line_spacing
        
    return current_y

# ==========================================
# Utility Function 3: Calculate 2D triangle area (Shoelace formula)
# ==========================================
def calculate_2d_triangle_area(p1, p2, p3):
    return 0.5 * abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))

# ==========================================
# Utility Function 4: Calculate tilt and area ratio based on explicit triangle_id
# ==========================================
def get_triangle_info(t_id, tgt_landmarks, ref_landmarks, mapper):
    if t_id in ["UNKNOWN", "OUT_OF_BOUNDS"]:
        return None, None
        
    try:
        c_idx = mapper.triangle_ids.index(t_id)
        c_tri = mapper.triangles[c_idx]
        
        tgt_p1 = tgt_landmarks[c_tri[0]]
        tgt_p2 = tgt_landmarks[c_tri[1]]
        tgt_p3 = tgt_landmarks[c_tri[2]]
        
        tilt = None
        if len(tgt_p1) >= 3 and len(tgt_p2) >= 3 and len(tgt_p3) >= 3:
            v1 = np.array(tgt_p2) - np.array(tgt_p1)
            v2 = np.array(tgt_p3) - np.array(tgt_p1)
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len != 0:
                cos_theta = abs(normal[2]) / norm_len
                tilt = int(round(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))))
                
        area_ratio = None
        if ref_landmarks is not None:
            ref_p1 = ref_landmarks[c_tri[0]]
            ref_p2 = ref_landmarks[c_tri[1]]
            ref_p3 = ref_landmarks[c_tri[2]]
            
            tgt_area = calculate_2d_triangle_area(tgt_p1, tgt_p2, tgt_p3)
            ref_area = calculate_2d_triangle_area(ref_p1, ref_p2, ref_p3)
            
            tgt_y_coords = [lm[1] for lm in tgt_landmarks]
            tgt_face_h = max(tgt_y_coords) - min(tgt_y_coords)
            
            ref_y_coords = [lm[1] for lm in ref_landmarks]
            ref_face_h = max(ref_y_coords) - min(ref_y_coords)
            
            if ref_area > 0 and ref_face_h > 0 and tgt_face_h > 0:
                scale_factor = (tgt_face_h / ref_face_h) ** 2
                normalized_tgt_area = tgt_area / scale_factor
                area_ratio = round(normalized_tgt_area / ref_area, 2)
                
        return tilt, area_ratio
    except Exception:
        pass
        
    return None, None

# ==========================================
# Utility Function 5: Calculate Average
# ==========================================
def calculate_mean(values):
    valid_vals = [v for v in values if v > 0]
    if not valid_vals:
        return 0.0
    return sum(valid_vals) / len(valid_vals)

# ==========================================
# Utility Function 6: Calculate Harmonic Mean
# ==========================================
def calculate_harmonic_mean(values):
    return calculate_mean(values)

    valid_vals = [v for v in values if v > 0]
    if not valid_vals:
        return 0.0
    return len(valid_vals) / sum(1.0 / v for v in valid_vals)

# ==========================================
# Utility Function 7: Draw perfect 2D boundary (Silhouette)
# ==========================================
def draw_2d_mesh_boundary(img, landmarks, triangles, color=(0, 255, 0), thickness=2):
    if not landmarks or not triangles:
        return
        
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for tri_indices in triangles:
        try:
            pt1 = [int(landmarks[tri_indices[0]][0]), int(landmarks[tri_indices[0]][1])]
            pt2 = [int(landmarks[tri_indices[1]][0]), int(landmarks[tri_indices[1]][1])]
            pt3 = [int(landmarks[tri_indices[2]][0]), int(landmarks[tri_indices[2]][1])]
            pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        except Exception:
            pass
            
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)


# ==========================================
# Main Process
# ==========================================
def process_patients():
    print("\n[*] Scanning entire folder to determine the number of patients to analyze...")
    patient_folders = []
    total_images_in_target = 0
    
    for root, dirs, files in os.walk(TARGET_DIR):
        jpg_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg' , '.png'))]
        if jpg_files:
            patient_id = os.path.basename(root)
            patient_folders.append((patient_id, root, jpg_files))
            total_images_in_target += len(jpg_files)
            
    total_patients = len(patient_folders)
    print(f"[*] Scan complete: Found a total of {total_patients} patients and {total_images_in_target} images.")
    print("==================================================\n")
    
    processed_patients_count = 0
    
    for patient_id, root, jpg_files in patient_folders:
        result_save_path = os.path.join(root, "result.json")
        
        patient_error = False
        pre_results_cache = {}
        
        # Always rebuild POST data from scratch
        post_updates_list = []  
        
        # ==========================================
        # [Step 0] Load existing data (Result caching)
        # ==========================================
        if os.path.exists(result_save_path):
            print(f"\n==================================================")
            print(f"[*] Patient {patient_id} already has '{result_save_path}', loading data.")
            try:
                with open(result_save_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                loaded_pre = loaded_data.get('pre', {})
                
                # [Modified] Extract and load into PRE only the data whose actual photos exist in the folder (root)
                for uuid_key, file_info in loaded_pre.items():
                    filename = file_info.get('filename', 'Unknown')
                    img_path = os.path.join(root, filename)
                    
                    if os.path.exists(img_path):
                        pre_results_cache[uuid_key] = file_info
                    else:
                        print(f"  -> [Warning] Image file does not exist, excluded from PRE data: {filename}")
                        import sys
                        sys.exit()
                
                print(f"  -> Loaded a total of {len(pre_results_cache)} valid image analysis results (images exist).")
                
                for file_uuid, file_info in pre_results_cache.items():
                    filename = file_info.get('filename', 'Unknown')
                    result = file_info.get('result', {})
                    landmarks = result.get('face_landmarks', [])
                    
                    if landmarks and len(landmarks[0]) < 3:
                        img_path = os.path.join(root, filename)
                        img = cv2.imread(img_path)
                        if img is not None:
                            landmarks_3d = local_mapper._get_landmarks(img)
                            if landmarks_3d is not None:
                                result['face_landmarks'] = landmarks_3d.tolist()
                                print(f"      [Update] Recalculated and replaced landmarks of '{filename}' in 3D.")

                for file_uuid, file_info in pre_results_cache.items():
                    filename = file_info.get('filename', 'Unknown')
                    result = file_info.get('result', {})
                    
                    is_face = "face_box_loc" in result or result.get("loc_string") == "face"
                    face_str = "Face Detected" if is_face else "Face Not Detected"
                    
                    blob_list = result.get('blob_list', [])
                    lesion_list1 = result.get('lesion_list1', [])
                    face_landmarks = result.get('face_landmarks', [])

                    triangle_malig_map = {}
                    all_boxes_info = []  
                    dx_counts = {}

                    for lesion in lesion_list1:
                        dx = lesion.get('top1_dx', 'Unknown')
                        dx_counts[dx] = dx_counts.get(dx, 0) + 1

                        lesion_box = [
                            lesion.get('x1', 0), lesion.get('y1', 0), 
                            lesion.get('x2', 0), lesion.get('y2', 0)
                        ]
                        malig = lesion.get('malignancy_output', 0)
                        
                        t_id = region_mapper.get_lesion_triangle_id(lesion_box, face_landmarks)
                        tilt, _ = get_triangle_info(t_id, face_landmarks, face_landmarks, region_mapper)
                        area_ratio = 1.0 
                        
                        all_boxes_info.append((lesion_box, malig, tilt, area_ratio))

                        if malig > 0.095:
                            overlapping_ids = region_mapper.get_all_overlapping_triangle_ids(lesion_box, face_landmarks)
                            for t in overlapping_ids:
                                triangle_malig_map[t] = max(triangle_malig_map.get(t, 0), malig)
                            
                            print(f"      ㄴ [Warning] Suspected malignant lesion found (Malignancy: {malig:.4f})")
                            print(f"         - Overlapping Triangle ID list: {overlapping_ids}")

                    if SAVE_DEBUG_IMAGES:
                        img_path = os.path.join(root, filename)
                        if face_landmarks and os.path.exists(img_path):
                            image = cv2.imread(img_path)
                            if image is not None:
                                img_h, img_w = image.shape[:2]
                                
                                dyn_scale = img_w / 1000.0
                                font_scale = max(0.4, 0.5 * dyn_scale)
                                thick_1 = max(1, int(1 * dyn_scale))
                                thick_2 = max(1, int(2 * dyn_scale))
                                pad = max(2, int(5 * dyn_scale))
                                
                                overlay = image.copy()
                                for idx, tri_indices in enumerate(region_mapper.triangles):
                                    t_id = region_mapper.triangle_ids[idx]
                                    pt1 = [int(face_landmarks[tri_indices[0]][0]), int(face_landmarks[tri_indices[0]][1])]
                                    pt2 = [int(face_landmarks[tri_indices[1]][0]), int(face_landmarks[tri_indices[1]][1])]
                                    pt3 = [int(face_landmarks[tri_indices[2]][0]), int(face_landmarks[tri_indices[2]][1])]
                                    pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
                                    
                                    if t_id in triangle_malig_map:
                                        t_malig = triangle_malig_map[t_id]
                                        if t_malig > 0.2:
                                            color_fill = (120, 120, 255) # Light Red
                                        else:
                                            color_fill = (100, 200, 255) # Light Orange
                                        cv2.fillPoly(overlay, [pts], color_fill)
                                
                                alpha = 0.4
                                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                                
                                for box, malig, tilt, ratio in all_boxes_info:
                                    if malig <= 0.095:
                                        color = (128, 128, 128)  
                                    elif malig <= 0.2:
                                        color = (0, 165, 255)    
                                    else:
                                        color = (0, 0, 255)      
                                        
                                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thick_2)
                                    
                                    score_int = int(round(malig * 100))
                                    text_str = f"{score_int}"
                                    if DRAW_TILT_RATIO and tilt is not None:
                                        text_str += f" [{tilt}]"
                                        
                                    text_size = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick_1)[0]
                                    text_y = max(int(box[1]) - pad, text_size[1] + pad)
                                    cv2.putText(image, text_str, (int(box[0]), text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick_1, cv2.LINE_AA)

                                if DRAW_DEBUG_TEXT:
                                    title_text = "Purpose: Single Image Lesion Mapping (Malig > 0.09)"
                                    pos_x = max(10, int(20 * dyn_scale))
                                    pos_y = max(10, int(20 * dyn_scale))
                                    draw_smart_text(image, title_text, pos=(pos_x, pos_y), base_scale=0.8, color=(0, 255, 0), thickness=thick_2)

                                # Modify filename and convert to 1000x1000 png (STEP0)
                                base_name = os.path.splitext(filename)[0]
                                debug_save_path = os.path.join(root, f"STEP0_{base_name}.png")
                                image_resized = cv2.resize(image, (1000, 1000), interpolation=cv2.INTER_AREA)
                                cv2.imwrite(debug_save_path, image_resized)
                                print(f"      [*] Debug image saved: {debug_save_path}")

                    if dx_counts:
                        sorted_dx = sorted(dx_counts.items(), key=lambda x: x[1], reverse=True)
                        dx_summary = ", ".join([f"{name}({cnt})" for name, cnt in sorted_dx[:5]])
                        if len(sorted_dx) > 5:
                            dx_summary += " etc..."
                    else:
                        dx_summary = "No detailed diagnosis info"

                    print(f"    - Filename: {filename} ({face_str})")
                    print(f"      ㄴ Total lesions: {len(blob_list)} / Detailed diagnosis lesions: {len(lesion_list1)}")
                    print(f"      ㄴ Summary diagnosis: {dx_summary}")
                    
            except Exception as e:
                print(f"  [Error] An error occurred while loading the existing json file: {e}")
                patient_error = True

        if patient_error:
            print(f"\n❌ [Error] An error occurred while processing patient {patient_id}, skipping.")
            continue

        # ==========================================
        # [Step 2.5] Pre-filtering lesions (Exclude out of bounds and inside nostrils)
        # ==========================================
        print(f"\n  [Step 2.5] Filtering out lesions out of bounds or inside nostrils before cross-checking...")
        
        for file_uuid, file_info in pre_results_cache.items():
            result = file_info.get('result', {})
            face_landmarks = result.get('face_landmarks', [])
            original_lesions = result.get('lesion_list1', [])
            
            if not face_landmarks:
                continue
                
            filtered_lesions = []
            for lesion in original_lesions:
                box = [
                    lesion.get('x1', 0), lesion.get('y1', 0), 
                    lesion.get('x2', 0), lesion.get('y2', 0)
                ]
                
                # 1. Exclude lesions located inside nostrils
                if region_mapper.is_box_center_in_nostril(box, face_landmarks):
                    continue
                    
                # 2. Exclude lesions outside the face boundary
                t_id = region_mapper.get_lesion_triangle_id(box, face_landmarks)
                if t_id in ["UNKNOWN", "OUT_OF_BOUNDS"]:
                    continue
                    
                filtered_lesions.append(lesion)
                
            # Permanently replace the lesion list in cached data with only valid lesions
            result['lesion_list1'] = filtered_lesions
            
            if len(original_lesions) != len(filtered_lesions):
                removed_count = len(original_lesions) - len(filtered_lesions)
                print(f"    - '{file_info.get('filename')}': Excluded {removed_count} out of {len(original_lesions)} (Remaining lesions: {len(filtered_lesions)})")

        # ==========================================
        # [Step 3] Lesion cross-validation (Cross-check) and rendering
        # Always regenerate POST data (post_updates_list).
        # ==========================================
        print(f"\n  [Step 3] Starting lesion cross-validation (Cross-check) with other photos...")
        
        post_updates_list = [] # Unconditionally update even if it already exists
        
        for ref_uuid, ref_info in pre_results_cache.items():
            ref_filename = ref_info.get('filename')
            ref_result = ref_info.get('result', {})
            ref_landmarks = ref_result.get('face_landmarks', [])
            ref_lesions = ref_result.get('lesion_list1', [])
            
            if not ref_landmarks:
                continue
                
            for ref_idx, lesion in enumerate(ref_lesions):
                malignancy = lesion.get('malignancy_output', 0)
                corrected_val = round(malignancy, 4)
                
                if malignancy > 0.09:
                    ref_box = [
                        lesion.get('x1', 0), lesion.get('y1', 0), 
                        lesion.get('x2', 0), lesion.get('y2', 0)
                    ]
                    
                    center_t_id = region_mapper.get_lesion_triangle_id(ref_box, ref_landmarks)
                    target_t_ids = region_mapper.get_visible_overlapping_triangle_ids(ref_box, ref_landmarks)
                    
                    if target_t_ids:
                        print(f"\n    [Analysis] Lesion found in photo '{ref_filename}' (Malignancy: {malignancy:.4f})")
                        print(f"      - Target triangle ID list: {target_t_ids}")
                        print(f"      - Reference center triangle ID: {center_t_id}")
                        
                        collected_max_maligs = []
                        
                        for tgt_uuid, tgt_info in pre_results_cache.items():
                            if ref_uuid == tgt_uuid:
                                continue 
                                
                            tgt_filename = tgt_info.get('filename')
                            tgt_result = tgt_info.get('result', {})
                            tgt_landmarks = tgt_result.get('face_landmarks', [])
                            tgt_lesions = tgt_result.get('lesion_list1', [])
                            
                            if not tgt_landmarks:
                                continue
                            
                            tilt_angle, area_ratio = get_triangle_info(center_t_id, tgt_landmarks, ref_landmarks, region_mapper)
                            
                            is_ratio_skipped = False
                            if area_ratio is not None and area_ratio <= 0.5:
                                is_ratio_skipped = True
                                
                            visible_t_ids = []
                            invisible_t_ids = []
                            
                            for t_id in target_t_ids:
                                if region_mapper.is_triangle_visible(t_id, tgt_landmarks):
                                    visible_t_ids.append(t_id)
                                else:
                                    invisible_t_ids.append(t_id)
                                    
                            is_center_visible = False
                            center_cx, center_cy = 0, 0
                            
                            if center_t_id not in ["UNKNOWN", "OUT_OF_BOUNDS"]:
                                is_center_visible = region_mapper.is_triangle_visible(center_t_id, tgt_landmarks)
                                try:
                                    c_idx = region_mapper.triangle_ids.index(center_t_id)
                                    c_tri = region_mapper.triangles[c_idx]
                                    p1 = tgt_landmarks[c_tri[0]]
                                    p2 = tgt_landmarks[c_tri[1]]
                                    p3 = tgt_landmarks[c_tri[2]]
                                    
                                    center_cx = int((p1[0] + p2[0] + p3[0]) / 3)
                                    center_cy = int((p1[1] + p2[1] + p3[1]) / 3)
                                except Exception:
                                    pass

                            matched_boxes = []
                            for tgt_lesion in tgt_lesions:
                                tgt_box = [
                                    tgt_lesion.get('x1', 0), tgt_lesion.get('y1', 0), 
                                    tgt_lesion.get('x2', 0), tgt_lesion.get('y2', 0)
                                ]
                                tgt_malig = tgt_lesion.get('malignancy_output', 0)
                                
                                if visible_t_ids and region_mapper.is_box_overlapping_target_ids(tgt_box, tgt_landmarks, visible_t_ids):
                                    matched_boxes.append((tgt_box, tgt_malig))

                            if not is_ratio_skipped:
                                print(f"      ㄴ Target photo '{tgt_filename}' inspection result:")
                            else:
                                print(f"      ㄴ Target photo '{tgt_filename}' inspection result: [Skip] Area ratio {area_ratio:.2f} <= 0.5")
                                
                            print(f"         - Visible triangles: {visible_t_ids if visible_t_ids else 'None'}")
                            print(f"         - Invisible triangles: {invisible_t_ids if invisible_t_ids else 'None'}")
                            
                            if tilt_angle is not None:
                                ratio_str = f"{area_ratio:.2f}" if area_ratio is not None else "N/A"
                                print(f"         - Center tilt: {tilt_angle} deg, Area ratio: {ratio_str}")
                            
                            if not is_center_visible:
                                print(f"         - [Notice] The center of the reference lesion ({center_t_id}) is invisible.")
                            
                            if SAVE_DEBUG_IMAGES and (visible_t_ids or not is_center_visible):
                                tgt_img_path = os.path.join(root, tgt_filename)
                                if os.path.exists(tgt_img_path):
                                    tgt_image = cv2.imread(tgt_img_path)
                                    if tgt_image is not None:
                                        img_h, img_w = tgt_image.shape[:2]
                                        
                                        dyn_scale = img_w / 1000.0
                                        font_scale = max(0.4, 0.5 * dyn_scale)
                                        thick_1 = max(1, int(1 * dyn_scale))
                                        thick_2 = max(1, int(2 * dyn_scale))
                                        thick_3 = max(2, int(3 * dyn_scale))
                                        thick_4 = max(2, int(4 * dyn_scale))
                                        pad = max(2, int(5 * dyn_scale))
                                        dash_len = max(4, int(8 * dyn_scale))
                                        
                                        overlay = tgt_image.copy()
                                        
                                        for idx, tri_indices in enumerate(region_mapper.triangles):
                                            t_id = region_mapper.triangle_ids[idx]
                                            pt1 = [int(tgt_landmarks[tri_indices[0]][0]), int(tgt_landmarks[tri_indices[0]][1])]
                                            pt2 = [int(tgt_landmarks[tri_indices[1]][0]), int(tgt_landmarks[tri_indices[1]][1])]
                                            pt3 = [int(tgt_landmarks[tri_indices[2]][0]), int(tgt_landmarks[tri_indices[2]][1])]
                                            pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
                                            
                                            if t_id in visible_t_ids:
                                                if t_id == center_t_id:
                                                    cv2.fillPoly(overlay, [pts], (128, 0, 0))
                                                else:
                                                    cv2.fillPoly(overlay, [pts], (255, 190, 190))
                                            else:
                                                cv2.polylines(overlay, [pts], isClosed=True, color=(200, 200, 200), thickness=thick_1)
                                                
                                        alpha = 0.5
                                        cv2.addWeighted(overlay, alpha, tgt_image, 1 - alpha, 0, tgt_image)
                                        
                                        for tgt_lesion in tgt_lesions:
                                            tgt_box_draw = [
                                                tgt_lesion.get('x1', 0), tgt_lesion.get('y1', 0), 
                                                tgt_lesion.get('x2', 0), tgt_lesion.get('y2', 0)
                                            ]
                                            tgt_malig_draw = tgt_lesion.get('malignancy_output', 0)
                                            
                                            tgt_t_id_draw = region_mapper.get_lesion_triangle_id(tgt_box_draw, tgt_landmarks)
                                            tgt_tilt_draw, tgt_ratio_draw = get_triangle_info(tgt_t_id_draw, tgt_landmarks, ref_landmarks, region_mapper)
                                            
                                            score_int = int(round(tgt_malig_draw * 100))
                                            text_str = f"{score_int}"
                                            if DRAW_TILT_RATIO and tgt_tilt_draw is not None:
                                                if tgt_ratio_draw is not None:
                                                    text_str += f" [{tgt_tilt_draw}, {tgt_ratio_draw:.2f}]"
                                                else:
                                                    text_str += f" [{tgt_tilt_draw}]"
                                            
                                            if tgt_malig_draw <= 0.09:
                                                color = (128, 128, 128)  
                                            elif tgt_malig_draw <= 0.2:
                                                color = (0, 165, 255)    
                                            else:
                                                color = (0, 0, 255)      
                                                
                                            cv2.rectangle(tgt_image, (int(tgt_box_draw[0]), int(tgt_box_draw[1])), (int(tgt_box_draw[2]), int(tgt_box_draw[3])), color, thick_2)
                                            
                                            text_size = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick_1)[0]
                                            text_y = max(int(tgt_box_draw[1]) - pad, text_size[1] + pad)
                                            cv2.putText(tgt_image, text_str, (int(tgt_box_draw[0]), text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick_1, cv2.LINE_AA)

                                        for m_box, _ in matched_boxes:
                                            draw_dashed_rectangle(tgt_image, (int(m_box[0]), int(m_box[1])), (int(m_box[2]), int(m_box[3])), (255, 0, 0), thickness=thick_2, dash_length=dash_len)

                                        if DRAW_DEBUG_TEXT:
                                            title_text = f"Purpose: Cross-check Ref [{ref_filename}] -> Tgt [{tgt_filename}]"
                                            pos_x = max(10, int(20 * dyn_scale))
                                            pos_y = max(10, int(20 * dyn_scale))
                                            next_y = draw_smart_text(tgt_image, title_text, pos=(pos_x, pos_y), base_scale=0.8, color=(0, 255, 255), thickness=thick_2)

                                            if not is_center_visible:
                                                next_y = draw_smart_text(tgt_image, "INVISIBLE", pos=(pos_x, next_y + 10), base_scale=1.5, color=(0, 0, 255), thickness=thick_3)
                                                
                                            if is_ratio_skipped:
                                                skip_text = f"SKIPPED (Ratio: {area_ratio:.2f} <= 0.5)"
                                                next_y = draw_smart_text(tgt_image, skip_text, pos=(pos_x, next_y + 10), base_scale=1.2, color=(0, 165, 255), thickness=thick_3)

                                        if DRAW_TILT_RATIO and tilt_angle is not None and is_center_visible:
                                            text = f"{tilt_angle} deg"
                                            if area_ratio is not None:
                                                text += f" (Ratio: {area_ratio:.2f})"
                                                
                                            tilt_font_scale = max(0.4, 0.8 * dyn_scale)
                                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, tilt_font_scale, thick_2)[0]
                                            text_x = max(center_cx - (text_size[0] // 2), 10)
                                            text_y = max(center_cy - 20, 30) 
                                            
                                            cv2.putText(tgt_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, tilt_font_scale, (0, 0, 0), thick_4, cv2.LINE_AA)
                                            cv2.putText(tgt_image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, tilt_font_scale, (0, 255, 255), thick_2, cv2.LINE_AA)

                                        # Draw accurate 2D outer boundary in green (STEP1)
                                        draw_2d_mesh_boundary(tgt_image, tgt_landmarks, region_mapper.triangles, color=(0, 255, 0), thickness=thick_2)

                                        # Modify filename and convert to 1000x1000 png (STEP1)
                                        base_ref = os.path.splitext(ref_filename)[0]
                                        base_tgt = os.path.splitext(tgt_filename)[0]
                                        debug_cross_path = os.path.join(root, f"STEP1_idx{ref_idx}_{base_ref}_to_{base_tgt}.png")
                                        tgt_image_resized = cv2.resize(tgt_image, (1000, 1000), interpolation=cv2.INTER_AREA)
                                        cv2.imwrite(debug_cross_path, tgt_image_resized)
                                        print(f"         [*] Cross overlay saved: {debug_cross_path}")
                                        
                            if matched_boxes:
                                print(f"         [Found] Checked lesions in target photo overlapping with the triangle:")
                                for m_box, m_malig in matched_boxes:
                                    print(f"            - Coordinates: {m_box}, Malignancy: {m_malig:.4f}")
                                    
                                if is_ratio_skipped:
                                    print(f"         [Skip] Excluded from collection due to unmet area ratio condition")
                                else:
                                    max_tgt_malig = max(m_malig for _, m_malig in matched_boxes)
                                    collected_max_maligs.append(max_tgt_malig)
                                    print(f"         [Collected] Max Malignancy of overlapping lesions in target photo: {max_tgt_malig:.4f}")
                            else:
                                if visible_t_ids:
                                    print(f"         [Not Found] No overlapping lesions with this triangle in this photo.")

                        if collected_max_maligs:
                            all_maligs = [malignancy] + collected_max_maligs
                            h_mean = calculate_mean(all_maligs)
                            
                            orig_malig = round(malignancy, 4)
                            corrected_malig = round(h_mean, 4)
                            lesion['corrected_malignancy'] = corrected_malig
                            
                            # [Core] Extract only updated lesions (values different from original) and save to post_updates
                            if orig_malig != corrected_malig:
                                post_updates_list.append({
                                    "uuid": ref_uuid,
                                    "lesion_id": ref_idx + 1,  # 1-based index
                                    "original_score": orig_malig,
                                    "final_score": corrected_malig
                                })
                            
                            print(f"\n      => [Correction Calculation] Original: {orig_malig:.4f}, Collected target max values: {collected_max_maligs}")
                            print(f"      => [Correction Result] Mean update (corrected_malignancy): {corrected_malig:.4f}")
                        else:
                            # Maintain original value if no overlapping lesions or skipped by area ratio filtering
                            lesion['corrected_malignancy'] = round(malignancy, 4)
                else:
                    # Maintain original value if malignancy is 0.09 or less (not inspected)
                    lesion['corrected_malignancy'] = round(malignancy, 4)

        # ==========================================
        # [Step 4] Render debug images based on final corrected results (post_updates only)
        # ==========================================
        if SAVE_DEBUG_IMAGES and not patient_error and post_updates_list:
            print(f"\n  [Step 4] Rendering final debug images reflecting only corrected results (post_updates)...")
            
            updates_by_uuid = {}
            for update in post_updates_list:
                u = update['uuid']
                if u not in updates_by_uuid:
                    updates_by_uuid[u] = []
                updates_by_uuid[u].append(update)
                
            for file_uuid, updates in updates_by_uuid.items():
                if file_uuid not in pre_results_cache:
                    continue
                    
                file_info = pre_results_cache[file_uuid]
                filename = file_info.get('filename')
                result = file_info.get('result', {})
                lesion_list = result.get('lesion_list1', [])
                face_landmarks = result.get('face_landmarks', [])
                
                img_path = os.path.join(root, filename)
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    if image is not None:
                        img_h, img_w = image.shape[:2]
                        
                        dyn_scale = img_w / 1000.0
                        font_scale = max(0.4, 0.5 * dyn_scale)
                        thick_1 = max(1, int(1 * dyn_scale))
                        thick_2 = max(1, int(2 * dyn_scale))
                        pad = max(2, int(5 * dyn_scale))
                        
                        for update in updates:
                            lesion_idx = update['lesion_id'] - 1 
                            
                            if lesion_idx < 0 or lesion_idx >= len(lesion_list):
                                continue
                                
                            lesion = lesion_list[lesion_idx]
                            box = [
                                lesion.get('x1', 0), lesion.get('y1', 0), 
                                lesion.get('x2', 0), lesion.get('y2', 0)
                            ]
                            
                            orig_malig = update['original_score']
                            corrected_malig = update['final_score']
                            
                            if corrected_malig <= 0.1:
                                box_color = (128, 128, 128)  
                            elif corrected_malig <= 0.2:
                                box_color = (0, 165, 255)    
                            else:
                                box_color = (0, 0, 255)      
                                
                            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), box_color, thick_2)
                            
                            orig_int = int(round(orig_malig * 100))
                            corr_int = int(round(corrected_malig * 100))
                            
                            text_str = f"U: {corr_int} (was {orig_int})"
                            text_color = (255, 0, 255) 
                                
                            text_size = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick_1)[0]
                            text_y = max(int(box[1]) - pad, text_size[1] + pad)
                            cv2.putText(image, text_str, (int(box[0]), text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thick_1, cv2.LINE_AA)
                            
                        if DRAW_DEBUG_TEXT:
                            title_text = "Purpose: Updated Lesions Only (post_updates)"
                            pos_x = max(10, int(20 * dyn_scale))
                            pos_y = max(10, int(20 * dyn_scale))
                            draw_smart_text(image, title_text, pos=(pos_x, pos_y), base_scale=0.8, color=(255, 255, 0), thickness=thick_2)

                        # Draw accurate 2D outer boundary in green (STEP2)
                        draw_2d_mesh_boundary(image, face_landmarks, region_mapper.triangles, color=(0, 255, 0), thickness=thick_2)

                        # Modify filename and convert to 1000x1000 png (STEP2)
                        base_name = os.path.splitext(filename)[0]
                        final_debug_save_path = os.path.join(root, f"STEP2_{base_name}.png")
                        image_resized = cv2.resize(image, (1000, 1000), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(final_debug_save_path, image_resized)
                        print(f"      [*] Update details found! Debug image saved: {final_debug_save_path}")

        # ==========================================
        # [Step 5] Render all final results for STEP3
        # ==========================================
        if SAVE_DEBUG_IMAGES and not patient_error:
            print(f"\n  [Step 5] Rendering debug images of all lesions reflecting final corrected values (STEP3)...")
            for file_uuid, file_info in pre_results_cache.items():
                filename = file_info.get('filename')
                result = file_info.get('result', {})
                lesion_list = result.get('lesion_list1', [])
                face_landmarks = result.get('face_landmarks', [])
                
                img_path = os.path.join(root, filename)
                if os.path.exists(img_path) and lesion_list:
                    image = cv2.imread(img_path)
                    if image is not None:
                        img_h, img_w = image.shape[:2]
                        
                        dyn_scale = img_w / 1000.0
                        font_scale = max(0.4, 0.5 * dyn_scale)
                        thick_1 = max(1, int(1 * dyn_scale))
                        thick_2 = max(1, int(2 * dyn_scale))
                        pad = max(2, int(5 * dyn_scale))
                        
                        for lesion in lesion_list:
                            box = [
                                lesion.get('x1', 0), lesion.get('y1', 0), 
                                lesion.get('x2', 0), lesion.get('y2', 0)
                            ]
                            
                            t_id = region_mapper.get_lesion_triangle_id(box, face_landmarks)
                            if t_id in ["UNKNOWN", "OUT_OF_BOUNDS"]:
                                continue
                                
                            malig = lesion.get('corrected_malignancy', lesion.get('malignancy_output', 0))
                            
                            if malig <= 0.1:
                                color = (128, 128, 128)  
                            elif malig <= 0.2:
                                color = (0, 165, 255)    
                            else:
                                color = (0, 0, 255)      
                                
                            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thick_2)
                            
                            score_int = int(round(malig * 100))
                            text_str = f"{score_int}"
                            
                            text_size = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick_1)[0]
                            text_y = max(int(box[1]) - pad, text_size[1] + pad)
                            cv2.putText(image, text_str, (int(box[0]), text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick_1, cv2.LINE_AA)

                        
                        # [Added] Paint nostril triangle regions in black (0, 0, 0)
                        for t_id in region_mapper.nostril_triangle_ids:
                            try:
                                if t_id in region_mapper.triangle_ids:
                                    idx = region_mapper.triangle_ids.index(t_id)
                                    tri_indices = region_mapper.triangles[idx]
                                    pt1 = [int(face_landmarks[tri_indices[0]][0]), int(face_landmarks[tri_indices[0]][1])]
                                    pt2 = [int(face_landmarks[tri_indices[1]][0]), int(face_landmarks[tri_indices[1]][1])]
                                    pt3 = [int(face_landmarks[tri_indices[2]][0]), int(face_landmarks[tri_indices[2]][1])]
                                    pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
                                    cv2.fillPoly(image, [pts], (0, 0, 0))
                            except Exception:
                                pass

                        # Draw accurate 2D outer boundary in green (STEP3)
                        draw_2d_mesh_boundary(image, face_landmarks, region_mapper.triangles, color=(0, 255, 0), thickness=thick_2)
                        
                        if DRAW_DEBUG_TEXT:
                            title_text = "Purpose: Final Updated All Lesions (STEP3)"
                            pos_x = max(10, int(20 * dyn_scale))
                            pos_y = max(10, int(20 * dyn_scale))
                            draw_smart_text(image, title_text, pos=(pos_x, pos_y), base_scale=0.8, color=(0, 255, 0), thickness=thick_2)

                        # Modify filename and convert to 1000x1000 png (STEP3)
                        base_name = os.path.splitext(filename)[0]
                        final_debug_save_path = os.path.join(root, f"STEP3_{base_name}.png")
                        image_resized = cv2.resize(image, (1000, 1000), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(final_debug_save_path, image_resized)
                        print(f"      [*] STEP3 image saved: {final_debug_save_path}")

        # ==========================================
        # ==== Final Data Saving Logic ====
        # ==========================================
        if not patient_error:
            # Cache saved with PRE info of deleted photos excluded
            patient_final_data = {
                "pre": pre_results_cache,
                "post_updates": post_updates_list  
            }

            with open(result_save_path, 'w', encoding='utf-8') as f:
                json.dump(patient_final_data, f, indent=4, ensure_ascii=False)
                
            processed_patients_count += 1
            print(f"\n✅ [Progress] Patient: {processed_patients_count} / {total_patients} completed (result.json updated)")
        

    print("\n==================================================")
    print("All processing tasks completed!")
    print("==================================================")

if __name__ == "__main__":
    process_patients()