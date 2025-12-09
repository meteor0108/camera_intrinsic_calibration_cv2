import cv2
import numpy as np
from glob import glob
import os
import argparse # argparse 라이브러리 추가

def calibrate_camera(image_path, pattern_width, pattern_height, square_size):
    """
    카메라 내부 파라미터를 캘리브레이션하는 함수

    Args:
        image_path (str): 캘리브레이션에 사용할 이미지들이 있는 폴더 경로
        pattern_width (int): 체커보드의 가로 방향 내부 코너 개수
        pattern_height (int): 체커보드의 세로 방향 내부 코너 개수
        square_size (float): 체커보드 한 칸의 실제 크기 (mm 단위)
    """

    # --- NumPy 출력 옵션 설정 ---
    np.set_printoptions(precision=6, suppress=True)

    # --- 체커보드 파라미터 설정 ---
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    # 3D 월드 좌표계 포인트 생성
    objp = np.zeros((pattern_width * pattern_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_width, 0:pattern_height].T.reshape(-1, 2)
    objp = objp * square_size # 실제 격자 크기 적용

    objpoints = []  # 3D 포인트 (월드 좌표계)
    imgpoints = []  # 2D 포인트 (이미지 평면)
    
    # --- 경로 설정 ---
    output_path = os.path.join(image_path, "result")
    os.makedirs(output_path, exist_ok=True)

    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif') 
    images = []
    
    # 각 확장자에 대해 파일을 찾아서 리스트에 추가
    for fmt in supported_formats:
        images.extend(glob(os.path.join(image_path, fmt)))

    if not images:
        print(f"오류: '{image_path}' 경로에 지원하는 형식의 이미지가 없습니다. 경로를 확인해주세요.")
        return
    else:
        # 찾은 이미지 타입 출력
        found_extensions = set()
        for image_file in images:
            extension = os.path.splitext(image_file)[1]
            found_extensions.add(extension) # 집합에 확장자 추가
        print(f"경로 속 이미지 타입: {found_extensions}")

    total_images = len(images)
    found_corners_count = 0

    print(f"총 {total_images}개의 이미지에서 체스보드 코너({pattern_width}x{pattern_height})를 찾습니다.")
    print("-" * 50)

    # --- 이미지 처리 루프 ---
    gray = None 
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"[{i+1}/{total_images}] {os.path.basename(fname)} - 이미지 로드 실패")
            continue

        print(f"[{i+1}/{total_images}] 처리 중: {os.path.basename(fname)}...", end='', flush=True)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, (pattern_width, pattern_height), None)

        if ret == True:
            found_corners_count += 1
            print(" ✓ 찾음")
            objpoints.append(objp)
            
            # 코너 위치 정밀도 향상
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # 코너가 검출된 이미지 저장
            cv2.drawChessboardCorners(img, (pattern_width, pattern_height), corners2, ret)
            save_filename = os.path.join(output_path, os.path.basename(fname))
            cv2.imwrite(save_filename, img)
        else:
            print(" ✗ 실패")

    cv2.destroyAllWindows()

    print("-" * 50)
    print("캘리브레이션을 시작합니다...")

    # --- 캘리브레이션 및 결과 출력 ---
    if found_corners_count > 0:
        # 카메라 캘리브레이션 수행
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("\n" + "="*25 + " 캘리브레이션 결과 " + "="*25)
        print(f"\n[ 요약 ]")
        print(f" ✓ 총 이미지: {total_images}개")
        print(f" ✓ 코너 검출 성공: {found_corners_count}개")
        print(f" ✓ Reprojection Error: {ret:.4f} pixel")

        print(f"\n[ Camera Matrix (mtx) ]")
        print(mtx)

        print(f"\n[ Distortion Coefficients (dist) ]")
        print(dist.ravel())

        # 결과 파일 저장
        result_txt_path = os.path.join(output_path, "intrinsic.txt")
        print(f"\n[ 결과 저장 (txt) ]")
        try:
            with open(result_txt_path, 'w') as f:
                f.write(f"## Reprojection Error\n")
                f.write(f"{ret}\n\n")

                f.write(f"## Intrinsic Matrix (mtx)\n")
                # np.savetxt를 사용해 행렬을 텍스트 파일에 저장 (소수점 6자리)
                np.savetxt(f, mtx, fmt='%.6f', delimiter=', ')
                f.write("\n")

                f.write(f"## Distortion Coefficients (dist)\n")
                np.savetxt(f, dist, fmt='%.6f', delimiter=', ')

            print(f" ✓ 캘리브레이션 결과가 '{result_txt_path}' 파일로 저장되었습니다.")
        except Exception as e:
            print(f" ✗ txt 파일 저장 중 오류 발생: {e}")

        print("\n" + "="*68)

    else:
        print("\n캘리브레이션 실패: 체스보드 코너를 찾은 이미지가 없습니다.")


if __name__ == '__main__':
    # --- 대화형으로 사용자에게 정보 입력받기 ---
    print("="*50)
    print("카메라 내부 캘리브레이션을 시작합니다.")

    # 1. 이미지 폴더 경로 입력받기
    input_dir = input("1. 데이터 폴더 경로를 입력 (ex:/home/yusungcuk/antlab/0_cam_intrinsic_cal_ws/data): ")

    # 2. 체커보드 가로 코너 개수 입력받기 (숫자가 아니면 다시 입력)
    while True:
        try:
            width = int(input("2. 체커보드의 가로 방향 내부 코너 개수를 입력 (예: 9): "))
            break # 숫자가 올바르게 입력되면 루프 탈출
        except ValueError:
            print("   [오류] 숫자로 입력해야 합니다. 다시 시도해주세요.")

    # 3. 체커보드 세로 코너 개수 입력받기 (숫자가 아니면 다시 입력)
    while True:
        try:
            height = int(input("3. 체커보드의 세로 방향 내부 코너 개수를 입력 (예: 6): "))
            break
        except ValueError:
            print("   [오류] 숫자로 입력해야 합니다. 다시 시도해주세요.")

    # 4. 격자 크기 입력받기 (숫자가 아니면 다시 입력)
    while True:
        try:
            square_size = float(input("4. 체커보드 한 칸의 실제 크기를 입력 (mm 단위, 예: 25.0): "))
            break
        except ValueError:
            print("   [오류] 숫자나 소수로 입력해야 합니다. 다시 시도해주세요.")

    print("\n" + "-"*50)
    print("입력이 완료되었습니다. 캘리브레이션을 진행합니다.")

    # 입력받은 정보들을 사용하여 캘리브레이션 함수 호출
    calibrate_camera(input_dir, width, height, square_size)