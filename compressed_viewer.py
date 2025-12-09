import rclpy  # ROS 2 Python 클라이언트 라이브러리
from rclpy.node import Node  # ROS 2 노드 클래스
from sensor_msgs.msg import CompressedImage  # 압축된 이미지 메시지 타입
import numpy as np  # 이미지 디코딩을 위한 NumPy
import cv2  # OpenCV 라이브러리 (이미지 처리 및 표시)

class CompressedImageSubscriber(Node):
    def __init__(self):
        """ROS 2 압축 이미지 구독 노드 초기화"""
        super().__init__('compressed_image_subscriber')  # 노드 이름 설정

        #  압축된 이미지 메시지 구독 (토픽명은 필요에 따라 변경)
        self.subscription = self.create_subscription(
            CompressedImage,
            # '/pylon_camera_node/image_raw/compressed',  # 다른 카메라 노드 사용 가능
            '/pylon_camera_node2/image_rect/compressed',  # 현재 사용 중인 카메라의 토픽
            self.image_callback,  # 메시지 수신 시 실행될 콜백 함수
            10)  # 큐 크기 (버퍼 크기)

        self.subscription  # 변수 미사용 경고 방지

        # OpenCV 창 설정
        self.window_width = 640   # 창 너비
        self.window_height = 480  # 창 높이
        self.window_name = "Compressed Image"  # 창 이름

        # 창 크기 및 위치 지정
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # 창 크기 조절 가능하도록 설정
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)  # 창 크기 설정
        cv2.moveWindow(self.window_name, 100, 100)  # 창 위치 조절 (화면 좌측 상단 기준 X=100, Y=100)

    def image_callback(self, msg):
        """압축된 이미지를 수신하여 디코딩하고 출력하는 콜백 함수"""
        try:
            # 메시지 데이터를 NumPy 배열로 변환 (압축된 JPEG 형식)
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # JPEG 이미지를 OpenCV 형식(BGR)으로 디코딩

            if image is not None:
                #  창 크기에 맞게 이미지 리사이징
                resized_image = cv2.resize(image, (self.window_width, self.window_height))

                #  OpenCV 창에 이미지 표시
                cv2.imshow(self.window_name, resized_image)
                cv2.waitKey(1)  # GUI 창이 응답할 수 있도록 설정
            else:
                self.get_logger().warn("Failed to decode image")  # 디코딩 실패 시 경고 메시지 출력

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")  # 오류 발생 시 로그 출력

def main(args=None):
    """메인 실행 함수"""
    rclpy.init(args=args)  # ROS 2 노드 초기화
    node = CompressedImageSubscriber()  # 노드 인스턴스 생성
    rclpy.spin(node)  # 노드 실행 (콜백 함수 실행 대기)
    
    # 종료 시 정리 작업
    node.destroy_node()  # 노드 삭제
    rclpy.shutdown()  # ROS 2 종료
    cv2.destroyAllWindows()  # OpenCV 창 닫기

if __name__ == '__main__':
    main()  # 스크립트 실행
