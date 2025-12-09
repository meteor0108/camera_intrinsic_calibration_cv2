import cv2
import numpy as np
from glob import glob
import os

def calibrate_and_undistort(image_path, pattern_width, pattern_height, square_size):
    """
    카메라 캘리브레이션을 수행하고, 그 결과로 이미지 왜곡을 보정하는 함수

    Args:
        image_path (str): 캘리브레이션에 사용할 이미지들이 있는 폴더 경로
        pattern_width (int): 체커보드의 가로 방향 내부 코너 개수
        pattern_height (int): 체커보드의 세로 방향 내부 코너 개수
        square_size (float): 체커보드 한 칸의 실제 크기 (mm 단위)
    """

    # --- 1. 캘리브레이션 준비 ---

    # NumPy 출력 옵션 설정
    np.set_printoptions(precision=6, suppress=True)

    # 체커보드 코너 찾기 위한 기준 설정
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    # 3D 월드 좌표계 포인트 생성 (z=0)
    objp = np.zeros((pattern_width * pattern_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_width, 0:pattern_height].T.reshape(-1, 2)
    objp = objp * square_size  # 실제 격자 크기 적용

    objpoints = []  # 3D 포인트 (월드 좌표계)
    imgpoints = []  # 2D 포인트 (이미지 평면)
    
    # 결과 저장 경로 설정
    cal_output_path = os.path.join(image_path, "calibration_result")
    undistort_output_path = os.path.join(image_path, "undistorted")
    os.makedirs(cal_output_path, exist_ok=True)
    os.makedirs(undistort_output_path, exist_ok=True)

    # 지원하는 이미지 형식으로 파일 검색
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif') 
    images = []
    for fmt in supported_formats:
        images.extend(glob(os.path.join(image_path, fmt)))

    if not images:
        print(f"오류: '{image_path}' 경로에 지원하는 형식의 이미지가 없습니다. 경로를 확인해주세요.")
        return

    # --- 2. 체커보드 코너 검출 ---
    
    gray = None
    total_images = len(images)
    found_corners_count = 0
    print(f"총 {total_images}개의 이미지에서 체스보드 코너({pattern_width}x{pattern_height})를 찾습니다.")
    print("-" * 50)

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"[{i+1}/{total_images}] {os.path.basename(fname)} - 이미지 로드 실패")
            continue

        print(f"[{i+1}/{total_images}] 처리 중: {os.path.basename(fname)}...", end='', flush=True)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 체스보드 코너 찾기
        ret, corners = cv2.findChessboardCorners(gray, (pattern_width, pattern_height), None)

        if ret:
            found_corners_count += 1
            print(" ✓ 찾음")
            objpoints.append(objp)
            
            # 코너 위치 정밀도 향상
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # 코너가 검출된 이미지 저장
            cv2.drawChessboardCorners(img, (pattern_width, pattern_height), corners2, ret)
            save_filename = os.path.join(cal_output_path, "corners_" + os.path.basename(fname))
            cv2.imwrite(save_filename, img)
        else:
            print(" ✗ 실패")

    if found_corners_count == 0:
        print("\n캘리브레이션 실패: 체스보드 코너를 찾은 이미지가 없습니다.")
        return
        
    # --- 3. 카메라 캘리브레이션 수행 ---

    print("-" * 50)
    print("캘리브레이션을 시작합니다...")
    
    # 카메라 캘리브레이션
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n" + "="*25 + " 캘리브레이션 결과 " + "="*25)
    print(f"\n[ 요약 ]")
    print(f" ✓ 총 이미지: {total_images}개")
    print(f" ✓ 코너 검출 성공: {found_corners_count}개")
    print(f" ✓ Reprojection Error: {ret:.4f} pixel")
    print(f"\n[ Camera Matrix (mtx) ]\n{mtx}")
    print(f"\n[ Distortion Coefficients (dist) ]\n{dist.ravel()}")

    # 결과 파일 저장
    result_txt_path = os.path.join(cal_output_path, "intrinsic.txt")
    print(f"\n[ 결과 저장 (txt) ]")
    try:
        with open(result_txt_path, 'w') as f:
            f.write(f"## Reprojection Error\n{ret}\n\n")
            f.write(f"## Intrinsic Matrix (mtx)\n")
            np.savetxt(f, mtx, fmt='%.6f', delimiter=', ')
            f.write("\n## Distortion Coefficients (dist)\n")
            np.savetxt(f, dist, fmt='%.6f', delimiter=', ')
        print(f" ✓ 캘리브레이션 결과가 '{result_txt_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f" ✗ txt 파일 저장 중 오류 발생: {e}")
    print("\n" + "="*68)


    # --- 4. 왜곡 보정 수행 ---
    
    print("\n" + "="*25 + " 왜곡 보정 시작 " + "="*25)
    print(f"캘리브레이션 결과를 사용하여 원본 이미지들의 왜곡을 보정합니다.")

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        # 최적의 새 카메라 행렬 계산
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # 왜곡 보정
        undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

        # 결과 이미지 저장
        save_filename = os.path.join(undistort_output_path, "undistorted_" + os.path.basename(fname))
        cv2.imwrite(save_filename, undistorted_img)
        print(f" ✓ '{os.path.basename(save_filename)}' 저장 완료")
        
    print(f"\n왜곡 보정이 완료되었습니다. 결과는 '{undistort_output_path}' 폴더에 저장되었습니다.")
    print("="*68 + "\n")


if __name__ == '__main__':
    print("="*50)
    print("카메라 캘리브레이션 및 왜곡 보정을 시작합니다.")
    print("="*50)

    # 1. 이미지 폴더 경로 입력받기
    input_dir = input("1. 캘리브레이션 이미지 폴더 경로를 입력하세요: ")

    # 2. 체커보드 가로 코너 개수 입력받기
    while True:
        try:
            width = int(input("2. 체커보드 가로 방향 내부 코너 개수를 입력하세요 (예: 9): "))
            break
        except ValueError:
            print("   [오류] 숫자로 입력해야 합니다. 다시 시도해주세요.")

    # 3. 체커보드 세로 코너 개수 입력받기
    while True:
        try:
            height = int(input("3. 체커보드 세로 방향 내부 코너 개수를 입력하세요 (예: 6): "))
            break
        except ValueError:
            print("   [오류] 숫자로 입력해야 합니다. 다시 시도해주세요.")

    # 4. 격자 크기 입력받기
    while True:
        try:
            square_size = float(input("4. 체커보드 한 칸의 실제 크기를 입력하세요 (mm 단위, 예: 25.0): "))
            break
        except ValueError:
            print("   [오류] 숫자나 소수로 입력해야 합니다. 다시 시도해주세요.")

    print("\n" + "-"*50)
    print("입력이 완료되었습니다. 작업을 진행합니다.")
    print("-"*50)

    # 캘리브레이션 및 왜곡 보정 함수 호출
    calibrate_and_undistort(input_dir, width, height, square_size)
