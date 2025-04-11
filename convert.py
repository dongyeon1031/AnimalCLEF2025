import pandas as pd

# 파일 경로
file_path = "/Users/dongyeonkim/Desktop/BiTAmin/1학기 프로젝트/333.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 컬럼 이름 변경
df.rename(columns={"individual": "identity"}, inplace=True)

# 변경된 데이터프레임 저장 (원본 파일 덮어쓰기 or 새 파일로 저장)
df.to_csv(file_path, index=False)  # 원본 덮어쓰기
# 또는 새로운 파일로 저장하고 싶다면 아래처럼 사용:
# df.to_csv("333_renamed.csv", index=False)