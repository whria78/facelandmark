import cv2
import numpy as np
import mediapipe as mp
import itertools
import string
from collections import defaultdict
import urllib.request

# ==========================================
# 468 Facial Landmark Symmetry Index Map
# ==========================================
SYMMETRY_MAP = {
    0:   0,   1:   1,   2:   2,   3: 248,   4:   4,   5:   5,   6:   6,   7: 249,   8:   8,   9:   9, 
    10:  10,  11:  11,  12:  12,  13:  13,  14:  14,  15:  15,  16:  16,  17:  17,  18:  18,  19:  19, 
    20: 250,  21: 251,  22: 252,  23: 253,  24: 254,  25: 255,  26: 256,  27: 257,  28: 258,  29: 259, 
    30: 260,  31: 261,  32: 262,  33: 263,  34: 264,  35: 265,  36: 266,  37: 267,  38: 268,  39: 269, 
    40: 270,  41: 271,  42: 272,  43: 273,  44: 274,  45: 275,  46: 276,  47: 277,  48: 278,  49: 279, 
    50: 280,  51: 281,  52: 282,  53: 283,  54: 284,  55: 285,  56: 286,  57: 287,  58: 288,  59: 289, 
    60: 290,  61: 291,  62: 292,  63: 293,  64: 294,  65: 295,  66: 296,  67: 297,  68: 298,  69: 299, 
    70: 300,  71: 301,  72: 302,  73: 303,  74: 304,  75: 305,  76: 306,  77: 307,  78: 308,  79: 309, 
    80: 310,  81: 311,  82: 312,  83: 313,  84: 314,  85: 315,  86: 316,  87: 317,  88: 318,  89: 319, 
    90: 320,  91: 321,  92: 322,  93: 323,  94:  94,  95: 324,  96: 325,  97: 326,  98: 327,  99: 328, 
    100: 329, 101: 330, 102: 331, 103: 332, 104: 333, 105: 334, 106: 335, 107: 336, 108: 337, 109: 338, 
    110: 339, 111: 340, 112: 341, 113: 342, 114: 343, 115: 344, 116: 345, 117: 346, 118: 347, 119: 348, 
    120: 349, 121: 350, 122: 351, 123: 352, 124: 353, 125: 354, 126: 355, 127: 356, 128: 357, 129: 358, 
    130: 359, 131: 360, 132: 361, 133: 362, 134: 363, 135: 364, 136: 365, 137: 366, 138: 367, 139: 368, 
    140: 369, 141: 370, 142: 371, 143: 372, 144: 373, 145: 374, 146: 375, 147: 376, 148: 377, 149: 378, 
    150: 379, 151: 151, 152: 152, 153: 380, 154: 381, 155: 382, 156: 383, 157: 384, 158: 385, 159: 386, 
    160: 387, 161: 388, 162: 389, 163: 390, 164: 164, 165: 391, 166: 392, 167: 393, 168: 168, 169: 394, 
    170: 395, 171: 396, 172: 397, 173: 398, 174: 399, 175: 175, 176: 400, 177: 401, 178: 402, 179: 403, 
    180: 404, 181: 405, 182: 406, 183: 407, 184: 408, 185: 409, 186: 410, 187: 411, 188: 412, 189: 413, 
    190: 414, 191: 415, 192: 416, 193: 417, 194: 418, 195: 195, 196: 419, 197: 197, 198: 420, 199: 199, 
    200: 200, 201: 421, 202: 422, 203: 423, 204: 424, 205: 425, 206: 426, 207: 427, 208: 428, 209: 429, 
    210: 430, 211: 431, 212: 432, 213: 433, 214: 434, 215: 435, 216: 436, 217: 437, 218: 438, 219: 439, 
    220: 440, 221: 441, 222: 442, 223: 443, 224: 444, 225: 445, 226: 446, 227: 447, 228: 448, 229: 449, 
    230: 450, 231: 451, 232: 452, 233: 453, 234: 454, 235: 455, 236: 456, 237: 457, 238: 458, 239: 459, 
    240: 460, 241: 461, 242: 462, 243: 463, 244: 464, 245: 465, 246: 466, 247: 467, 248:   3, 249:   7, 
    250:  20, 251:  21, 252:  22, 253:  23, 254:  24, 255:  25, 256:  26, 257:  27, 258:  28, 259:  29, 
    260:  30, 261:  31, 262:  32, 263:  33, 264:  34, 265:  35, 266:  36, 267:  37, 268:  38, 269:  39, 
    270:  40, 271:  41, 272:  42, 273:  43, 274:  44, 275:  45, 276:  46, 277:  47, 278:  48, 279:  49, 
    280:  50, 281:  51, 282:  52, 283:  53, 284:  54, 285:  55, 286:  56, 287:  57, 288:  58, 289:  59, 
    290:  60, 291:  61, 292:  62, 293:  63, 294:  64, 295:  65, 296:  66, 297:  67, 298:  68, 299:  69, 
    300:  70, 301:  71, 302:  72, 303:  73, 304:  74, 305:  75, 306:  76, 307:  77, 308:  78, 309:  79, 
    310:  80, 311:  81, 312:  82, 313:  83, 314:  84, 315:  85, 316:  86, 317:  87, 318:  88, 319:  89, 
    320:  90, 321:  91, 322:  92, 323:  93, 324:  95, 325:  96, 326:  97, 327:  98, 328:  99, 329: 100, 
    330: 101, 331: 102, 332: 103, 333: 104, 334: 105, 335: 106, 336: 107, 337: 108, 338: 109, 339: 110, 
    340: 111, 341: 112, 342: 113, 343: 114, 344: 115, 345: 116, 346: 117, 347: 118, 348: 119, 349: 120, 
    350: 121, 351: 122, 352: 123, 353: 124, 354: 125, 355: 126, 356: 127, 357: 128, 358: 129, 359: 130, 
    360: 131, 361: 132, 362: 133, 363: 134, 364: 135, 365: 136, 366: 137, 367: 138, 368: 139, 369: 140, 
    370: 141, 371: 142, 372: 143, 373: 144, 374: 145, 375: 146, 376: 147, 377: 148, 378: 149, 379: 150, 
    380: 153, 381: 154, 382: 155, 383: 156, 384: 157, 385: 158, 386: 159, 387: 160, 388: 161, 389: 162, 
    390: 163, 391: 165, 392: 166, 393: 167, 394: 169, 395: 170, 396: 171, 397: 172, 398: 173, 399: 174, 
    400: 176, 401: 177, 402: 178, 403: 179, 404: 180, 405: 181, 406: 182, 407: 183, 408: 184, 409: 185, 
    410: 186, 411: 187, 412: 188, 413: 189, 414: 190, 415: 191, 416: 192, 417: 193, 418: 194, 419: 196, 
    420: 198, 421: 201, 422: 202, 423: 203, 424: 204, 425: 205, 426: 206, 427: 207, 428: 208, 429: 209, 
    430: 210, 431: 211, 432: 212, 433: 213, 434: 214, 435: 215, 436: 216, 437: 217, 438: 218, 439: 219, 
    440: 220, 441: 221, 442: 222, 443: 223, 444: 224, 445: 225, 446: 226, 447: 227, 448: 228, 449: 229, 
    450: 230, 451: 231, 452: 232, 453: 233, 454: 234, 455: 235, 456: 236, 457: 237, 458: 238, 459: 239, 
    460: 240, 461: 241, 462: 242, 463: 243, 464: 244, 465: 245, 466: 246, 467: 247, 
}

# ==========================================
# FaceRegionMapper Class
# ==========================================
class FaceRegionMapper:
    """
    Class responsible for assigning IDs to facial landmark triangles and mapping left-right symmetry
    """
    def __init__(self):
        # --- [Added] Define landmark indices for the nostril region ---
        self.RIGHT_NOSTRIL_IDX = [48, 115, 220, 44, 1, 19, 94, 2, 97, 98, 64]
        # Automatically map left nostril landmark indices using SYMMETRY_MAP
        self.LEFT_NOSTRIL_IDX = [SYMMETRY_MAP[idx] for idx in self.RIGHT_NOSTRIL_IDX]
        self.nostril_triangle_ids = []  # For storing Triangle ID list belonging to the nostrils
        # ----------------------------------------------

        # 1. Extract triangles and adjacency list (edges) from Mediapipe's Tesselation data
        self.triangles, self.adj = self._build_triangles()
        print(f"Total triangles: {len(self.triangles)}")
        
        # 2. Generate unique IDs (AA, AB, AC ... ZZZ) corresponding to the number of triangles
        self.triangle_ids = self._generate_ids(len(self.triangles))
        
        # 3. Create a symmetric ID map by directly referencing the global variable SYMMETRY_MAP
        self.sym_id_map = self._build_symmetric_id_map(SYMMETRY_MAP)
        
        # 4. Classify left/right/center by directly referencing the global variable SYMMETRY_MAP
        self.id_to_side = self._build_side_map(SYMMETRY_MAP)

        # 5. [Modified] Generate 5000x5000 debug map (map.png) and detect nostril region
        self._generate_debug_map()

    def _generate_debug_map(self):
        """
        [Modified] Load the Canonical Face Model, visualize it on a 5000x5000 canvas, 
        paint the triangles corresponding to the inside of the nostrils black, and output them as a list.
        """
        image_size = 2000  # Increased resolution to 5000x5000
        margin = 100
        
        # 1. Get standard face model data
        url = "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj"
        try:
            response = urllib.request.urlopen(url)
            obj_data = response.read().decode('utf-8')
        except Exception as e:
            print(f"Skipping map generation due to standard face model download failure: {e}")
            return

        # 2. Extract 3D coordinates and scale to 5000x5000
        vertices = []
        for line in obj_data.split('\n'):
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2])])
                
        vertices = np.array(vertices[:468], dtype=np.float32)
        vertices[:, 1] = -vertices[:, 1] # Y-axis inversion
        
        min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
        min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
        
        width = max_x - min_x
        height = max_y - min_y
        
        scale = min((image_size - 2 * margin) / width, (image_size - 2 * margin) / height)
        
        center_x = (max_x + min_x) / 2.0
        center_y = (max_y + min_y) / 2.0
        
        scaled_landmarks = np.zeros_like(vertices)
        scaled_landmarks[:, 0] = (vertices[:, 0] - center_x) * scale + (image_size / 2.0)
        scaled_landmarks[:, 1] = (vertices[:, 1] - center_y) * scale + (image_size / 2.0)
        scaled_landmarks = scaled_landmarks.astype(np.int32)
        
        # 3. Create 5000x5000 canvas (white background)
        canvas = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        # --- [Added] Create nostril contour polygon ---
        right_contour = np.array([scaled_landmarks[i] for i in self.RIGHT_NOSTRIL_IDX], dtype=np.int32)
        left_contour = np.array([scaled_landmarks[i] for i in self.LEFT_NOSTRIL_IDX], dtype=np.int32)
        
        nostril_triangles = []

        # 4. Render triangles and add Triangle ID text
        for idx, tri_indices in enumerate(self.triangles):
            tri_id = self.triangle_ids[idx]
            
            pt1 = scaled_landmarks[tri_indices[0]]
            pt2 = scaled_landmarks[tri_indices[1]]
            pt3 = scaled_landmarks[tri_indices[2]]
            pts = np.array([pt1, pt2, pt3], dtype=np.int32)
            
            # Center of gravity coordinates of the triangle
            cx = float((pt1[0] + pt2[0] + pt3[0]) / 3.0)
            cy = float((pt1[1] + pt2[1] + pt3[1]) / 3.0)
            
            # Determine if the center point is inside the nostril polygon
            in_right = cv2.pointPolygonTest(right_contour, (cx, cy), False) >= 0
            in_left = cv2.pointPolygonTest(left_contour, (cx, cy), False) >= 0
            
            if in_right or in_left:
                nostril_triangles.append(tri_id)
                # Paint the triangles inside the nostrils black
                cv2.fillConvexPoly(canvas, pts, (0, 0, 0))
            else:
                # Regular triangle outline
                cv2.polylines(canvas, [pts], isClosed=True, color=(200, 200, 200), thickness=1)

            
            # Insert ID text (white text for black background nostrils, blue for the rest)
            text_color = (255, 255, 255) if (in_right or in_left) else (255, 0, 0)
            cv2.putText(canvas, tri_id, (int(cx) - 15, int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

        # Draw thick green outlines for the nostrils to make them clearly visible
        cv2.polylines(canvas, [right_contour], isClosed=True, color=(0, 255, 0), thickness=4)
        cv2.polylines(canvas, [left_contour], isClosed=True, color=(0, 255, 0), thickness=4)
        
        # 5. Add landmark points and number text
        for i, pt in enumerate(scaled_landmarks):
            cv2.circle(canvas, tuple(pt), 4, (0, 0, 255), -1)
            # Apply a white border around the text so it doesn't blend into the black background
            cv2.putText(canvas, str(i), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 6. Save image and output list
        cv2.imwrite("map.png", canvas)
        print(" -> Facial landmark and Triangle ID visualization map (map.png) saved.")

        # --- [Added] 9-grid division saving logic ---
        h, w, _ = canvas.shape
        h_step = h // 3
        w_step = w // 3
        
        part_num = 1
        for row in range(3):
            for col in range(3):
                y_start = row * h_step
                y_end = (row + 1) * h_step if row < 2 else h
                x_start = col * w_step
                x_end = (col + 1) * w_step if col < 2 else w
                
                # Image splitting via Numpy array slicing
                part_img = canvas[y_start:y_end, x_start:x_end]
                cv2.imwrite(f"map_part{part_num}.png", part_img)
                part_num += 1
                
        print(" -> Map saved in 9 divisions (map_part1.png ~ map_part9.png).")
        # ----------------------------------
        
        self.nostril_triangle_ids = nostril_triangles
        print(f" -> Triangle IDs belonging to the inside of the nostrils ({len(self.nostril_triangle_ids)}): {self.nostril_triangle_ids}")


    # --- [Added] Interface function to check if the box center is in the nostril region ---
    def is_box_center_in_nostril(self, box, landmarks):
        """
        Returns whether the exact center coordinates of the given box are located inside the nostril (left/right) region.
        """
        if box is None or landmarks is None:
            return False
            
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        
        right_contour = np.array([[landmarks[i][0], landmarks[i][1]] for i in self.RIGHT_NOSTRIL_IDX], dtype=np.float32)
        left_contour = np.array([[landmarks[i][0], landmarks[i][1]] for i in self.LEFT_NOSTRIL_IDX], dtype=np.float32)
        
        in_right = cv2.pointPolygonTest(right_contour, (cx, cy), False) >= 0
        in_left = cv2.pointPolygonTest(left_contour, (cx, cy), False) >= 0
        
        return in_right or in_left

    # (The following existing methods like _build_triangles, _generate_ids, etc., are kept as is)

    def _build_triangles(self):
        mp_face_mesh = mp.solutions.face_mesh
        tesselation = mp_face_mesh.FACEMESH_TESSELATION
        
        adj = defaultdict(set)
        for u, v in tesselation:
            adj[u].add(v)
            adj[v].add(u)
            
        triangles = set()
        for u in adj:
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            triangles.add((u, v, w))
                            
        return sorted(list(triangles)), adj

    def _generate_ids(self, count):
        ids = []
        length = 2
        while len(ids) < count:
            for combo in itertools.product(string.ascii_uppercase, repeat=length):
                ids.append("".join(combo))
                if len(ids) >= count:
                    break
            length += 1
        return ids

    def _build_symmetric_id_map(self, symmetry_map):
        triangle_to_id = {tri: self.triangle_ids[i] for i, tri in enumerate(self.triangles)}
        sym_id_map = {}
        for i, tri in enumerate(self.triangles):
            curr_id = self.triangle_ids[i]
            sym_tri = tuple(sorted([
                symmetry_map[tri[0]], 
                symmetry_map[tri[1]], 
                symmetry_map[tri[2]]
            ]))
            
            if sym_tri in triangle_to_id:
                sym_id_map[curr_id] = triangle_to_id[sym_tri]
            else:
                sym_id_map[curr_id] = "UNKNOWN"
                
        return sym_id_map

    def _build_side_map(self, symmetry_map):
        center_nodes = {i for i, sym in symmetry_map.items() if i == sym}
        
        left_nodes = {33}
        queue = [33]
        while queue:
            curr = queue.pop(0)
            for neighbor in self.adj[curr]:
                if neighbor not in left_nodes and neighbor not in center_nodes:
                    left_nodes.add(neighbor)
                    queue.append(neighbor)
                    
        right_nodes = set(symmetry_map.keys()) - left_nodes - center_nodes
        
        id_to_side = {}
        for i, tri in enumerate(self.triangles):
            t_id = self.triangle_ids[i]
            if any(n in left_nodes for n in tri):
                id_to_side[t_id] = 'L'
            elif any(n in right_nodes for n in tri):
                id_to_side[t_id] = 'R'
            else:
                id_to_side[t_id] = 'C'
                
        return id_to_side

    # ------------------------------------------
    # Shape collision detection algorithms
    # ------------------------------------------
    def _is_point_in_triangle(self, pt, v1, v2, v3):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def _is_box_intersect_triangle(self, box, v1, v2, v3):
        """
        Accurately determines collision between AABB (Axis-Aligned Bounding Box) and a 2D triangle 
        using the Separating Axis Theorem (SAT).
        """
        x1, y1, x2, y2 = box

        tx_min = min(v1[0], v2[0], v3[0])
        tx_max = max(v1[0], v2[0], v3[0])
        ty_min = min(v1[1], v2[1], v3[1])
        ty_max = max(v1[1], v2[1], v3[1])

        if tx_max < x1 or tx_min > x2 or ty_max < y1 or ty_min > y2:
            return False

        box_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        tri_vertices = [v1, v2, v3]

        for i in range(3):
            p1 = tri_vertices[i]
            p2 = tri_vertices[(i + 1) % 3]

            nx = p2[1] - p1[1]
            ny = p1[0] - p2[0]

            tri_proj = [nx * v[0] + ny * v[1] for v in tri_vertices]
            tri_min, tri_max = min(tri_proj), max(tri_proj)

            box_proj = [nx * bc[0] + ny * bc[1] for bc in box_corners]
            box_min, box_max = min(box_proj), max(box_proj)

            if box_max < tri_min or tri_max < box_min:
                return False

        return True

    # ------------------------------------------
    # Mapping interface
    # ------------------------------------------
    def get_lesion_triangle_id(self, box, landmarks):
        """
        [Modified] Returns only 1 ID of the Triangle closest to (in front of) the camera 
        among the Triangles containing the 'Center point' of the box.
        """
        if box is None or landmarks is None:
            return "UNKNOWN"
            
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        pt = (cx, cy)
        
        candidate_triangles = []
        
        for idx, tri_indices in enumerate(self.triangles):
            idx1, idx2, idx3 = tri_indices
            
            v1 = landmarks[idx1]
            v2 = landmarks[idx2]
            v3 = landmarks[idx3]
            
            # 1. Check if the center point is inside the triangle in 2D space
            if self._is_point_in_triangle(pt, v1, v2, v3):
                # 2. Check Z (depth) coordinates and calculate average depth (MediaPipe: smaller negative values are closer)
                if len(v1) >= 3:
                    avg_z = (v1[2] + v2[2] + v3[2]) / 3.0
                else:
                    avg_z = 0.0
                    
                candidate_triangles.append((self.triangle_ids[idx], avg_z))
                
        if not candidate_triangles:
            return "OUT_OF_BOUNDS"
            
        # 3. Return the triangle with the smallest Z value (closest to the front)
        closest_triangle = min(candidate_triangles, key=lambda x: x[1])
        return closest_triangle[0]

    def get_all_overlapping_triangle_ids(self, box, landmarks):
        """
        Returns a list of 'all' Triangle IDs that slightly overlap (collide) with the box area.
        """
        if box is None or landmarks is None:
            return []

        overlapping_ids = []
        for idx, tri_indices in enumerate(self.triangles):
            idx1, idx2, idx3 = tri_indices
            v1 = landmarks[idx1]
            v2 = landmarks[idx2]
            v3 = landmarks[idx3]

            if self._is_box_intersect_triangle(box, v1, v2, v3):
                overlapping_ids.append(self.triangle_ids[idx])

        return overlapping_ids

    def get_visible_overlapping_triangle_ids(self, box, landmarks, z_threshold=5.0):
        """
        [Modified] Finds all candidate triangles overlapping with the box in 2D, 
        individually judges if each is obscured by another triangle based on the '2D center point', and returns them.
        """
        if box is None or landmarks is None:
            return []

        # 1. Extract all candidate triangles intersecting with the 2D Box
        overlapping_candidates = self.get_all_overlapping_triangle_ids(box, landmarks)

        visible_ids = []
        # 2. Determine if each triangle in the candidate group is fully visible on screen
        for t_id in overlapping_candidates:
            if self.is_triangle_visible(t_id, landmarks, z_threshold):
                visible_ids.append(t_id)

        return visible_ids

    def is_triangle_visible(self, target_id, landmarks, z_threshold=5.0):
        """
        [Modified] Checks if the 'center point' of the target triangle is included (overlapped) inside another triangle based on 2D, 
        and if overlapped, judges it as obscured if the average Z value of the other triangle is closer to the camera (smaller value) than the target.
        """
        if landmarks is None or target_id not in self.sym_id_map:
            return False
            
        target_idx = self.triangle_ids.index(target_id)
        tri_indices = self.triangles[target_idx]
        
        t_v1 = landmarks[tri_indices[0]]
        t_v2 = landmarks[tri_indices[1]]
        t_v3 = landmarks[tri_indices[2]]
        
        if len(t_v1) < 3:
            return True 
            
        # 1. Calculate the 2D exact center point of the target triangle
        target_cx = (t_v1[0] + t_v2[0] + t_v3[0]) / 3.0
        target_cy = (t_v1[1] + t_v2[1] + t_v3[1]) / 3.0
        target_pt = (target_cx, target_cy)
        
        # 2. Average depth (Z) of the target triangle
        target_z = (t_v1[2] + t_v2[2] + t_v3[2]) / 3.0
        
        # 3. Check for 2D center point overlap with all other triangles
        for idx, other_indices in enumerate(self.triangles):
            current_id = self.triangle_ids[idx]
            
            # Exclude itself from the comparison
            if current_id == target_id:
                continue
                
            v1 = landmarks[other_indices[0]]
            v2 = landmarks[other_indices[1]]
            v3 = landmarks[other_indices[2]]
            
            # Whether the 'center point' of the target triangle falls within the area of another triangle (full overlap in 2D)
            if self._is_point_in_triangle(target_pt, v1, v2, v3):
                current_z = (v1[2] + v2[2] + v3[2]) / 3.0
                
                # If the overlapped triangle (current) protrudes forward (smaller value) more than the z_threshold compared to the target, it is considered obscured
                if current_z < (target_z - z_threshold):
                    return False  
                    
        return True

    def is_box_overlapping_target_ids(self, box, landmarks, target_ids):
        """
        Determines if the box overlaps (collides) with at least one of the triangles 
        included in the given target_ids list, returning True/False.
        """
        if box is None or landmarks is None or not target_ids:
            return False

        target_ids_set = set(target_ids)

        for idx, tri_indices in enumerate(self.triangles):
            t_id = self.triangle_ids[idx]
            
            if t_id in target_ids_set:
                idx1, idx2, idx3 = tri_indices
                v1 = landmarks[idx1]
                v2 = landmarks[idx2]
                v3 = landmarks[idx3]

                if self._is_box_intersect_triangle(box, v1, v2, v3):
                    return True  

        return False

    def get_left_right_ids(self, triangle_id):
        """
        Takes the ID of a specific triangle and returns a tuple of (Left ID, Right ID).
        """
        if triangle_id not in self.sym_id_map:
            return ("UNKNOWN", "UNKNOWN")
            
        side = self.id_to_side.get(triangle_id, 'C')
        sym_id = self.sym_id_map[triangle_id]
        
        if side == 'L':
            return (triangle_id, sym_id)
        elif side == 'R':
            return (sym_id, triangle_id)
        else:
            return (triangle_id, triangle_id)


# ==========================================
# LesionMapper Class
# ==========================================
class LesionMapper:
    """
    Integrated library for facial landmark-based lesion mapping, TTA cropping, face angle extraction, and identity verification.
    """
    def __init__(self, min_detection_confidence=0.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )

    # ------------------------------------------
    # Facial landmark extraction (Modified to include Z values)
    # ------------------------------------------
    def _get_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        h, w, _ = image.shape
        # Extract Z depth information at pixel scale
        landmarks = [[lm.x * w, lm.y * h, lm.z * w] for lm in results.multi_face_landmarks[0].landmark]
        return np.array(landmarks, dtype=np.float32)

    # ------------------------------------------
    # Extract full face Box and angle (Pitch/Yaw)
    # ------------------------------------------
    def get_face_info(self, image):
        landmarks = self._get_landmarks(image)
        if landmarks is None:
            return None, 0.0, 0.0, None

        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        x_min, y_min = int(np.min(x_coords)), int(np.min(y_coords))
        x_max, y_max = int(np.max(x_coords)), int(np.max(y_coords))
        face_box = (x_min, y_min, x_max, y_max)

        pitch, yaw = self._calculate_angles(landmarks, image.shape)
        return face_box, pitch, yaw, landmarks

    def _calculate_angles(self, landmarks, img_shape):
        h, w, _ = img_shape
        landmark_indices = [1, 152, 33, 263, 61, 291] 
        
        model_points = np.array([
            (0.0, 0.0, 0.0),             
            (0.0, -330.0, -65.0),        
            (-225.0, 170.0, -135.0),     
            (225.0, 170.0, -135.0),      
            (-150.0, -150.0, -125.0),    
            (150.0, -150.0, -125.0)      
        ], dtype=np.float64)

        # Extract only x, y 2D coordinates from 3D landmarks and pass to PnP solver
        image_points = np.array([landmarks[idx][:2] for idx in landmark_indices], dtype=np.float64)

        focal_length = 1.0 * w
        cam_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_matrix)
        if not success:
            return 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch = round(-angles[0], 2)
        yaw = round(-angles[1], 2)
        return pitch, yaw

    # ------------------------------------------
    # Internal calculations related to lesion mapping
    # ------------------------------------------
    def _get_closest_landmarks(self, landmarks, center_x, center_y, num_points=3):
        # Apply slicing to use only x, y coordinates when calculating distance
        distances = np.linalg.norm(landmarks[:, :2] - np.array([center_x, center_y]), axis=1)
        return np.argsort(distances)[:num_points]

    def _transform_box(self, box, matrix):
        x1, y1, x2, y2 = box
        pts = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.float32)
        mapped_pts = cv2.transform(pts, matrix)
        return (
            int(mapped_pts[0][0][0]), int(mapped_pts[0][0][1]),
            int(mapped_pts[1][0][0]), int(mapped_pts[1][0][1])
        )

    def _expand_box(self, box, margin=15):
        x1, y1, x2, y2 = box
        return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)

    def _overlap_area(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        return max(0, xB - xA) * max(0, yB - yA)

    def _get_surrounding_boxes(self, box):
        x1, y1, x2, y2 = box
        w = int((x2 - x1) / 2)
        h = int((y2 - y1) / 2)
        return [
            (x1 - w, y1 - h, x2 - w, y2 - h),
            (x1 + w, y1 + h, x2 + w, y2 + h),
            (x1 - w, y1 + h, x2 - w, y2 + h),
            (x1 + w, y1 - h, x2 + w, y2 - h)
        ]

    def _make_square(self, box):
        x1, y1, x2, y2 = box
        w = int((x2 - x1) / 2)
        h = int((y2 - y1) / 2)
        gap = abs(w - h)
        
        if w > h:
            y1 -= gap
            y2 += gap
        else:
            x1 -= gap
            x2 += gap
            
        if (y2 - y1) < (x2 - x1): y2 += 1
        if (y2 - y1) > (x2 - x1): x2 += 1
        return (x1, y1, x2, y2)

    # ------------------------------------------
    # TTA crop and final mapping caller
    # ------------------------------------------
    def get_tta_crops(self, image, box):
        img_h, img_w, _ = image.shape
        crops = []
        all_boxes = self._get_surrounding_boxes(box) + [box]
        
        for b in all_boxes:
            sq_box = self._make_square(b)
            x1, y1, x2, y2 = sq_box
            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                continue 
            crop_img = image[y1:y2, x1:x2]
            h_, w_, _ = crop_img.shape
            if h_ != w_ or h_ == 0:
                continue
            crops.append(crop_img)
        return crops

    def map_lesion(self, image_A, image_B, box_A, draw_debug=False):
        debug_image = image_B.copy() if draw_debug else None
        
        landmarks_A = self._get_landmarks(image_A)
        landmarks_B = self._get_landmarks(image_B)
        if landmarks_A is None or landmarks_B is None:
            return False, None, debug_image

        # cv2.estimateAffinePartial2D only supports 2D(x,y) coordinates, so apply [:2] slicing
        global_matrix, _ = cv2.estimateAffinePartial2D(landmarks_A[:, :2], landmarks_B[:, :2])
        if global_matrix is None:
            return False, None, debug_image
        global_box_B = self._transform_box(box_A, global_matrix)

        center_x = (box_A[0] + box_A[2]) / 2.0
        center_y = (box_A[1] + box_A[3]) / 2.0
        local_indices = self._get_closest_landmarks(landmarks_A, center_x, center_y, num_points=3)
        
        # Slicing local landmark array used for cv2.estimateAffinePartial2D
        local_matrix, _ = cv2.estimateAffinePartial2D(landmarks_A[local_indices, :2], landmarks_B[local_indices, :2])
        if local_matrix is None:
            return False, None, debug_image
        local_box_B = self._transform_box(box_A, local_matrix)

        # 1. Existing overlap check (Applied 15px margin expansion)
        expanded_global_box = self._expand_box(global_box_B, margin=15)
        expanded_local_box = self._expand_box(local_box_B, margin=15)
        overlap_success = self._overlap_area(expanded_global_box, expanded_local_box) > 0

        # -------------------------------------------------------------------------
        # 2. Verify if left/right sides are the same based on the nose (Nose tip, index 1)
        # -------------------------------------------------------------------------
        nose_x_A = landmarks_A[1][0]
        box_A_cx = (box_A[0] + box_A[2]) / 2.0
        side_A = "left" if box_A_cx < nose_x_A else "right"

        nose_x_B = landmarks_B[1][0]
        box_B_cx = (global_box_B[0] + global_box_B[2]) / 2.0
        side_B = "left" if box_B_cx < nose_x_B else "right"

        same_side_success = (side_A == side_B)

        # 3. Final success condition: Overlap AND located on the anatomically same side is recognized as a success
        is_success = overlap_success and same_side_success

        if draw_debug:
            if is_success:
                cv2.rectangle(debug_image, (global_box_B[0], global_box_B[1]), 
                            (global_box_B[2], global_box_B[3]), (255, 0, 255), 4)
                cv2.rectangle(debug_image, (local_box_B[0], local_box_B[1]), 
                            (local_box_B[2], local_box_B[3]), (0, 255, 0), 4)
            else:
                cv2.rectangle(debug_image, (global_box_B[0], global_box_B[1]), 
                            (global_box_B[2], global_box_B[3]), (255, 255, 0), 4)
                cv2.rectangle(debug_image, (local_box_B[0], local_box_B[1]), 
                            (local_box_B[2], local_box_B[3]), (0, 0, 255), 4)

        final_box = local_box_B if is_success else None
        return is_success, final_box, debug_image