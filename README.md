## Foodstagram-flask
Python으로 작성된 Foodstagram의 TensorFlow 작동부분을 JAVA로 포팅 실패.

- 시간 문제로 Python => JAVA 포팅 포기, TensorFlow 작동부분만 메인 서버로부터 분리.
- 서버가 이미지를 분석하는데 시간이 소요됨. http를 이용한 RESTful API로 구현 시 TimeOut 문제 발생 가능. 
- 이미지 인식  한정 클라이언트와의 통신 방법 소켓으로 변경.
