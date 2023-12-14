import numpy as np
import pymysql
import tracker
from detector import Detector
import cv2
import time


# MariaDB 연결 설정
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='root',
    database='pythondb'
)

cursor = conn.cursor()

# 테이블 생성 SQL 문
create_table_query = '''
CREATE TABLE IF NOT EXISTS 1F_R_P (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    in_count1 INT,
    out_count1 INT,
    in_count2 INT,
    out_count2 INT,
    in_count3 INT,
    out_count3 INT,
    in_count4 INT,
    out_count4 INT
)
'''

in_out_counts = {"IN": 0, "OUT": 0}

# 테이블 생성
cursor.execute(create_table_query)
conn.commit()

# IN/OUT 카운트를 저장하는 함수
def save_counts_to_db():
    global down_count_RB, up_count_RB, down_count_GY, up_count_GY, down_count_OP, up_count_OP, down_count_BW, up_count_BW
    insert_query = f"INSERT INTO 1F_R_P (in_count1, out_count1, in_count2, out_count2, in_count3, out_count3, in_count4, out_count4) VALUES ({down_count_RB}, {up_count_RB}, {down_count_GY}, {up_count_GY}, {down_count_OP}, {up_count_OP}, {down_count_BW}, {up_count_BW})"
    cursor.execute(insert_query)
    conn.commit()
    print(f"Counts saved to database: IN={down_count_RB}, OUT={up_count_RB}, IN2={down_count_GY}, OUT2={up_count_GY}, IN3={down_count_OP}, OUT3={up_count_OP}, IN4={down_count_BW}, OUT4={up_count_BW}")

# 10분마다 IN/OUT 카운트를 데이터베이스에 저장
last_save_time = time.time()
save_interval = 1  # 10분 = 600초
current_time = time.time()


if __name__ == '__main__':
    # --------------------------영상 입력란-------------------------------------------------
    #capture = cv2.VideoCapture('./video/test_1.mp4')
    #capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture('http://192.168.0.40:65534/')
    #--------------------------------------------------------------------------------------

    # 동영상 크기에 따라，폴리곤 채우기，충돌 계산을 위해

    # red polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_red = [[1000, 500], [1200, 500], [1200, 530], [1000, 530]]
    ndarray_pts_red = np.array(list_pts_red, np.int32)
    polygon_red_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_red], color=1)
    polygon_red_value = polygon_red_value[:, :, np.newaxis]

    # blue polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_blue = [[1000, 600], [1300, 600], [1300, 630], [1000, 630]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=2)
    polygon_blue_value = polygon_blue_value[:, :, np.newaxis]



    # green polygon------------
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_green = [[700, 200], [900, 200], [900, 230], [700, 230]]
    ndarray_pts_green = np.array(list_pts_green, np.int32)
    polygon_green_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_green], color=1)
    polygon_green_value = polygon_green_value[:, :, np.newaxis]

    # yellow polygon-----------
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_yellow = [[700, 100], [900, 100], [900, 130], [700, 130]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value = polygon_yellow_value[:, :, np.newaxis]



    # orange polygon-----------
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_orange = [[1670, 500], [1700, 500], [1700, 1080], [1670, 1080]]
    ndarray_pts_orange = np.array(list_pts_orange, np.int32)
    polygon_orange_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_orange], color=2)
    polygon_orange_value = polygon_orange_value[:, :, np.newaxis]

    # purple polygon-----------
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_purple = [[1770, 500], [1800, 500], [1800, 1080], [1770, 1080]]
    ndarray_pts_purple = np.array(list_pts_purple, np.int32)
    polygon_purple_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_purple], color=2)
    polygon_purple_value = polygon_purple_value[:, :, np.newaxis]



    # black polygon-----------
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_black = [[220, 500], [250, 500], [250, 1080], [220, 1080]]
    ndarray_pts_black = np.array(list_pts_black, np.int32)
    polygon_black_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_black], color=2)
    polygon_black_value = polygon_black_value[:, :, np.newaxis]

    # white polygon-----------
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_white = [[120, 500], [150, 500], [150, 1080], [120, 1080]]
    ndarray_pts_white = np.array(list_pts_white, np.int32)
    polygon_white_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_white], color=2)
    polygon_white_value = polygon_white_value[:, :, np.newaxis]




    # 라인충돌 감지용mask，polygon 2개 포함，（값의 범위 0、1、2），충돌계산 위해
    polygon_mask_red_and_blue = polygon_red_value + polygon_blue_value
    polygon_mask_green_and_yellow = polygon_green_value + polygon_yellow_value
    polygon_mask_orange_and_purple = polygon_orange_value + polygon_purple_value
    polygon_mask_black_and_white = polygon_black_value + polygon_white_value
    
    # 축소된 크기，1920x1080->960x540
    polygon_mask_red_and_blue = cv2.resize(polygon_mask_red_and_blue, (960, 540))
    polygon_mask_green_and_yellow = cv2.resize(polygon_mask_green_and_yellow, (960, 540))#---
    polygon_mask_orange_and_purple = cv2.resize(polygon_mask_orange_and_purple, (960, 540))
    polygon_mask_black_and_white = cv2.resize(polygon_mask_black_and_white, (960, 540))



    # 빨간 다각형 polygon-----------
    red_color_plate = [0, 0, 255] # b,g,r
    red_image = np.array(polygon_red_value * red_color_plate, np.uint8)

    # 파란 다각형 polygon-----------
    blue_color_plate = [255, 0, 0]
    blue_image = np.array(polygon_blue_value * blue_color_plate, np.uint8)

    # 초록 다각형 polygon-----------
    green_color_plate = [0, 255, 0]
    green_image = np.array(polygon_green_value * green_color_plate, np.uint8)

    # 노란 다각형 polygon------------
    yellow_color_plate = [0, 255, 255]
    yellow_image = np.array(polygon_yellow_value * yellow_color_plate, np.uint8)

    # 오랜지 다각형 polygon-----------
    orange_color_plate = [0, 160, 255]
    orange_image = np.array(polygon_orange_value * orange_color_plate, np.uint8)

    # 보라 다각형 polygon-----------
    purple_color_plate = [204, 0, 204]
    purple_image = np.array(polygon_purple_value * purple_color_plate, np.uint8)

    # 검정 다각형 polygon-----------
    black_color_plate = [30, 30, 30]
    black_image = np.array(polygon_black_value * black_color_plate, np.uint8)

    # 흰색 다각형 polygon-----------
    white_color_plate = [245, 245, 245]
    white_image = np.array(polygon_white_value * white_color_plate, np.uint8)



    # 컬러사진（값의 범위 0-255）
    color_polygons_image = red_image + blue_image + green_image + yellow_image + orange_image + purple_image + black_image + white_image#-------
    # 크기축소，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # list 빨간 polygon과 겹침
    list_overlapping_red_polygon = []
    # list 파란 polygon과 겹침
    list_overlapping_blue_polygon = []
    # list 초록 polygon과 겹침-----------
    list_overlapping_green_polygon = []
    # list 노란 polygon과 겹침-----------
    list_overlapping_yellow_polygon = []
    # list 오랜지 polygon과 겹침-----------
    list_overlapping_orange_polygon = []
    # list 보라 polygon과 겹침-----------
    list_overlapping_purple_polygon = []
    # list 검정 polygon과 겹침-----------
    list_overlapping_black_polygon = []
    # list 흰색 polygon과 겹침-----------
    list_overlapping_white_polygon = []

    #카운트
    down_count_RB = 0
    up_count_RB = 0

    down_count_GY = 0
    up_count_GY = 0

    down_count_OP = 0
    up_count_OP = 0

    down_count_BW = 0
    up_count_BW = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # 초기화 yolov5
    detector = Detector()

    while True:
        #ret,img_color = capture.read()

        # 각 프레임 읽기
        _, im = capture.read()

        if im is None:
            break

        # 화면 축소，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)

        # 화면에 bbox가 있는 경우
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 사진 프레임
            # 직선 충돌 감지 지점, (x1, y1), y 방향 오프셋 비율 0.0~1.0
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
            pass
        else:
            # 화면에 bbox가 없는 경우
            output_image_frame = im
        pass

        # 출력 이미지
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
# ----------------------라인 감지 판정----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id_RB = item_bbox
                a1, b1, a2, b2, label, track_id_GY = item_bbox
                c1, d1, c2, d2, label, track_id_OP = item_bbox
                e1, f1, e2, f2, label, track_id_BW = item_bbox

                # 직선 충돌 감지 지점, (x1, y1), y 방향 오프셋 비율 0.0~1.0 RB
                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                # 충돌 지점
                y = y1_offset
                x = x1

                b1_offset = int(b1 + ((b2 - b1) * 0.6))
                b = b1_offset
                a = a1

                d1_offset = int(d1 + ((d2 - d1) * 0.6))
                d = d1_offset
                c = c1

                f1_offset = int(f1 + ((f2 - f1) * 0.6))
                f = f1_offset
                e = e1

 #------------------------------------------------
                if polygon_mask_red_and_blue[y, x] == 1:
                    # 빨간색 polygon에 맞으면
                    if track_id_RB not in list_overlapping_red_polygon:
                        list_overlapping_red_polygon.append(track_id_RB)
                    pass

                    # 파란색polygon list에 track_id 있는지 확인
                    # track_id_RB 확인시 아래로 이동 간주
                    if track_id_RB in list_overlapping_blue_polygon:
                        # OUT+1
                        up_count_RB += 1
                        save_counts_to_db()
                        last_save_time = current_time
                        print(f'범주: {label} | id: {track_id_RB} | Upline | 윗선 충돌 수: {up_count_RB} | 업링크id목록: {list_overlapping_blue_polygon}')

                        list_overlapping_blue_polygon.remove(track_id_RB)

                        pass
                    else:
                        pass
#----------------------------------
                elif polygon_mask_red_and_blue[y, x] == 2:
                    # 파란색 polygon에 맞으면
                    if track_id_RB not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id_RB)
                    pass

                    if track_id_RB in list_overlapping_red_polygon:

                        down_count_RB += 1
                        save_counts_to_db()
                        last_save_time = current_time
                        print(f'범주: {label} | id: {track_id_RB} | Dwline | 아선 충돌 수: {down_count_RB} | 다링크id목록: {list_overlapping_red_polygon}')

                        list_overlapping_red_polygon.remove(track_id_RB)

                        pass
                    else:

                        pass
#------------------------------------------------
                elif polygon_mask_green_and_yellow[b, a] == 1:
                    # 초록 polygon에 맞으면
                    if track_id_GY not in list_overlapping_green_polygon:
                        list_overlapping_green_polygon.append(track_id_GY)
                    pass

                    # 노란 polygon list에 track_id_GY 있는지 확인
                    # track_id_GY 확인시 아래로 이동 간주
                    if track_id_GY in list_overlapping_yellow_polygon:
                        # OUT+1
                        up_count_GY += 1
                        save_counts_to_db()
                        last_save_time = current_time
                        print(f'범주: {label} | id: {track_id_GY} | Upline | 윗선 충돌 수: {up_count_GY} | 업링크id목록: {list_overlapping_yellow_polygon}')


                        list_overlapping_yellow_polygon.remove(track_id_GY)

                        pass
                    else:

                        pass
#------------------------------------------------
                elif polygon_mask_green_and_yellow[b, a] == 2:

                    if track_id_GY not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id_GY)
                    pass

                    if track_id_GY in list_overlapping_green_polygon:
                        down_count_GY += 1
                        save_counts_to_db()
                        last_save_time = current_time
                        print(f'범주: {label} | id: {track_id_GY} | Dwline | 아선 충돌 수: {down_count_GY} | 다링크id목록: {list_overlapping_green_polygon}')

                        list_overlapping_green_polygon.remove(track_id_GY)

                        pass
                    else:
                        pass
#------------------------------------------------
                if polygon_mask_orange_and_purple[d, c] == 1:
                    # 오랜지 polygon에 맞으면
                    if track_id_OP not in list_overlapping_orange_polygon:
                        list_overlapping_orange_polygon.append(track_id_OP)
                    pass

                    # 보라 polygon list에 track_id 있는지 확인
                    # track_id_OP 확인시 아래로 이동 간주
                    if track_id_OP in list_overlapping_purple_polygon:
                        # OUT+1
                        up_count_OP += 1
                        save_counts_to_db()
                        last_save_time = current_time
                        print(f'범주: {label} | id: {track_id_OP} | Upline | 윗선 충돌 수: {up_count_OP} | 업링크id목록: {list_overlapping_purple_polygon}')

                        list_overlapping_purple_polygon.remove(track_id_OP)

                        pass
                    else:
                        pass
#----------------------------------
                elif polygon_mask_orange_and_purple[d, c] == 2:
                    # 보라 polygon에 맞으면
                    if track_id_OP not in list_overlapping_purple_polygon:
                        list_overlapping_purple_polygon.append(track_id_OP)
                    pass

                    if track_id_OP in list_overlapping_orange_polygon:

                        down_count_OP += 1
                        save_counts_to_db()
                        last_save_time = current_time
                        print(f'범주: {label} | id: {track_id_OP} | Dwline | 아선 충돌 수: {down_count_OP} | 다링크id목록: {list_overlapping_orange_polygon}')

                        list_overlapping_orange_polygon.remove(track_id_OP)

                        pass
                    else:
                        pass
#------------------------------------------------
                if polygon_mask_black_and_white[f, e] == 1:
                    # 검정 polygon에 맞으면
                    if track_id_BW not in list_overlapping_black_polygon:
                        list_overlapping_black_polygon.append(track_id_BW)
                    pass

                    # 하얀 polygon list에 track_id 있는지 확인
                    # track_id_BW 확인시 아래로 이동 간주
                    if track_id_BW in list_overlapping_white_polygon:
                        # OUT+1
                        up_count_BW += 1
                        save_counts_to_db()
                        last_save_time = current_time
                        print(f'범주: {label} | id: {track_id_BW} | Upline | 윗선 충돌 수: {up_count_BW} | 업링크id목록: {list_overlapping_white_polygon}')

                        list_overlapping_white_polygon.remove(track_id_BW)

                        pass
                    else:

                        pass
#----------------------------------
                elif polygon_mask_black_and_white[f, e] == 2:
                    # 하얀 polygon에 맞으면
                    if track_id_BW not in list_overlapping_white_polygon:
                        list_overlapping_white_polygon.append(track_id_BW)
                    pass

                    if track_id_BW in list_overlapping_black_polygon:
                        down_count_BW += 1
                        save_counts_to_db()
                        last_save_time = current_time
                        print(f'범주: {label} | id: {track_id_BW} | Dwline | 아선 충돌 수: {down_count_BW} | 다링크id목록: {list_overlapping_black_polygon}')

                        list_overlapping_black_polygon.remove(track_id_BW)

                        pass
                    else:
                        pass
#------------------------------------------------

                        pass
                    pass
                else:
                    pass

                pass

            pass

# ----------------------쓸데없는 id 지우기----------------------
            list_overlapping_all = list_overlapping_blue_polygon + list_overlapping_red_polygon + list_overlapping_yellow_polygon + list_overlapping_green_polygon + list_overlapping_orange_polygon + list_overlapping_purple_polygon + list_overlapping_black_polygon + list_overlapping_white_polygon

            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 발견되지 않은 경우 id 삭제
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_red_polygon:
                        list_overlapping_red_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_green_polygon:
                        list_overlapping_green_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_orange_polygon:
                        list_overlapping_orange_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_purple_polygon:
                        list_overlapping_purple_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_black_polygon:
                        list_overlapping_black_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_white_polygon:
                        list_overlapping_white_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 빈 list
            list_bboxs.clear()

            pass
        else:
            # 이미지에 bbox가 없는 경우
            list_overlapping_red_polygon.clear()
            list_overlapping_blue_polygon.clear()
            list_overlapping_green_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            list_overlapping_orange_polygon.clear()
            list_overlapping_purple_polygon.clear()
            list_overlapping_black_polygon.clear()
            list_overlapping_white_polygon.clear()
            pass
        pass

        text_draw = 'IN: ' + str(down_count_RB) + \
                    ' OUT: ' + str(up_count_RB) + \
                    '  IN: ' + str(down_count_GY) + \
                    ' OUT: ' + str(up_count_GY) + \
                    '  IN: ' + str(down_count_OP) + \
                    ' OUT: ' + str(up_count_OP) + \
                    '  IN: ' + str(down_count_BW) + \
                    ' OUT: ' + str(up_count_BW)



        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=0.5, color=(255, 0, 0), thickness=2
                                         )

        cv2.imshow('demo', output_image_frame,)
        cv2.waitKey(1)

        pass

    pass

    capture.release()
    cv2.destroyAllWindows()
cursor.close()
conn.close()








