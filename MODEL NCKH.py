import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ==============================================================================
# BƯỚC 1: NẠP & CHUẨN HÓA DỮ LIỆU THEO KHUNG GIỜ (1H)
# ==============================================================================
file_path = "/content/household_power_consumption.csv"

# 1. Đọc file dữ liệu
df = pd.read_csv(file_path, sep=",", na_values=["?"], low_memory=False)

# 2. Xử lý cột Datetime
# Dữ liệu có dạng: 16/12/2006 và 5:24:00 PM
# Sử dụng format cụ thể để tối ưu tốc độ và tránh lỗi NaT
df["Datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

df.set_index("Datetime", inplace=True)
# Loại bỏ các dòng không thể chuyển đổi thời gian
df = df[df.index.notna()]
df.drop(columns=["Date", "Time"], inplace=True)

# 3. Ép kiểu dữ liệu về float
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"], errors="coerce"
)

# 4. Resample về độ phân giải 1 giờ (Lấy trung bình) & xóa giá trị NaN
df_hourly = df[["Global_active_power"]].resample("1H").mean().dropna()

print(f"✅ Đã chuẩn hóa dữ liệu theo giờ. Tổng số bản ghi (giờ): {len(df_hourly):,}")
display(df_hourly.head())
# ==============================================================================
# BƯỚC 2: HÀM TẠO ĐẶC TRƯNG NÂNG CAO (ADVANCED FEATURE ENGINEERING)
# ==============================================================================
def create_advanced_features(data):
  df_feat = data.copy()

  # 1. Cyclical Encoding (Mã hóa sin/cos cho giờ và ngày trong tuần)
  hour = df_feat.index.hour
  dayofweek = df_feat.index.dayofweek

  df_feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
  df_feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
  df_feat["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
  df_feat["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
  df_feat["is_weekend"] = (dayofweek >= 5).astype(int)

  # 2. Lag Features mở rộng
  lags = [1, 2, 3, 4, 5, 23, 24, 25, 168]
  for lag in lags:
    df_feat[f"lag_{lag}h"] = df_feat["Global_active_power"].shift(lag)

  # 3. Differencing Features (Tốc độ biến thiên)
  df_feat["diff_1h"] = df_feat["lag_1h"] - df_feat["lag_2h"]
  df_feat["diff_24h"] = df_feat["lag_1h"] - df_feat["lag_24h"]

  # 4. Rolling Statistics mở rộng
  for window in [3, 6, 12, 24]:
    roll_series = df_feat["Global_active_power"].shift(1).rolling(window)
    df_feat[f"rolling_mean_{window}h"] = roll_series.mean()
    df_feat[f"rolling_std_{window}h"] = roll_series.std()
    df_feat[f"rolling_max_{window}h"] = roll_series.max()
    df_feat[f"rolling_min_{window}h"] = roll_series.min()

  return df_feat.dropna()
# ==============================================================================
# BƯỚC 3: HUẤN LUYỆN MÔ HÌNH DỰ BÁO (LIGHTGBM) VÀ CÀI ĐẶT NGƯỠNG ĐỘNG
# ==============================================================================

# Tạo features mới cho dữ liệu đã resample theo giờ
df_adv = create_advanced_features(df_hourly)

# Lấy danh sách toàn bộ cột đặc trưng
feature_cols = [col for col in df_adv.columns if col != "Global_active_power"]
X = df_adv[feature_cols]
y = df_adv["Global_active_power"]

# Chia dữ liệu theo thời gian (80% Train, 20% Test)
split_idx = int(len(df_adv) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
test_indices = df_adv.index[split_idx:]

model_opt = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.015,
    num_leaves=31,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model_opt.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

# Đánh giá lại R^2 score và RMSE trên tập Test
y_pred_opt = model_opt.predict(X_test)
r2_opt = r2_score(y_test, y_pred_opt)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))

print("🚀 --- KẾT QUẢ MÔ HÌNH LIGHTGBM NÂNG CẤP --- ")
print(f"📈 R² Score: {r2_opt:.4f}")
print(f"📉 RMSE:     {rmse_opt:.4f} kW")


# ==============================================================================
# BƯỚC 4: THUẬT TOÁN PHÁT HIỆN BẤT THƯỜNG & PHÂN LOẠI CẢNH BÁO
# ==============================================================================
# 1. Tính sai số trên tập Train để xác định Ngưỡng động (Dynamic Threshold)
y_train_pred_opt = model_opt.predict(X_train)
train_errors = np.abs(y_train.values - y_train_pred_opt)

mu = np.mean(train_errors)
sigma = np.std(train_errors)
k = 3.0  # Hệ số độ lệch chuẩn (Z-Score threshold)
threshold = mu + k * sigma

print(f"\n⚙️ Ngưỡng động Z-Score (k={k}):")
print(f"   - Sai số trung bình (μ): {mu:.4f} kW")
print(f"   - Độ lệch chuẩn (σ): {sigma:.4f} kW")
print(f"   - Ngưỡng cảnh báo (Threshold = μ + k*σ): {threshold:.4f} kW\n")


# 2. Hàm kiểm tra và Phân loại Cảnh báo
def evaluate_anomalies(
    y_actual_series, y_pred_array, indices, threshold_val, top_n=10
):
  test_errors = np.abs(y_actual_series.values - y_pred_array)

  alerts = []
  for idx, actual, pred, err in zip(
      indices, y_actual_series.values, y_pred_array, test_errors
  ):
    if err > threshold_val:
      diff = actual - pred
      hour = idx.hour

      # Phân loại bất thường
      if diff > 0:
        alert_type = "🔥 TĂNG ĐỘT BIẾN (Nguy cơ rò rỉ / Quá tải thiết bị)"
      elif hour >= 0 and hour <= 5 and actual > 1.5:
        alert_type = (
            "🌙 RÒ RỈ ĐIỆN BAN ĐÊM (Tải nền cao bất thường khi đang ngủ)"
        )
      else:
        alert_type = "⚠️ GIẢM ĐỘT BIẾN (Mất nguồn / Hỏng hóc thiết bị)"

      msg = (
          f"🚨 [CẢNH BÁO {idx.strftime('%Y-%m-%d %H:%M')}]\n"
          f"   • Thực tế: {actual:.2f} kW | Dự báo chuẩn: {pred:.2f} kW\n"
          f"   • Độ lệch: {abs(diff):.2f} kW (Vượt ngưỡng {threshold_val:.2f} kW)\n"
          f"   • Chẩn đoán: {alert_type}"
      )
      alerts.append((idx, msg))

  print(
      f"🔎 Phát hiện tổng cộng {len(alerts)} giờ bị BẤT THƯỜNG trên tập"
      " Test.\n"
  )
  print(f"--- MẪU {min(top_n, len(alerts))} CẢNH BÁO ĐẦU TIÊN TÌM THẤY ---")
  for _, alert_msg in alerts[:top_n]:
    print(alert_msg)
    print("-" * 60)

  return alerts

# Chạy kiểm tra bất thường trên dữ liệu thực tế tập Test
anomalies = evaluate_anomalies(y_test, y_pred_opt, test_indices, threshold)
import joblib

# Lưu mô hình đã huấn luyện vào file
joblib.dump(model_opt, 'optimized_lightgbm_model.pkl')
print('✅ Đã lưu mô hình LightGBM tối ưu vào tệp optimized_lightgbm_model.pkl')
# Nạp lại mô hình đã lưu
loaded_model = joblib.load('optimized_lightgbm_model.pkl')
print('✅ Đã nạp lại mô hình LightGBM từ tệp optimized_lightgbm_model.pkl')
loaded_model = joblib.load('optimized_lightgbm_model.pkl')
print('✅ Đã nạp lại mô hình LightGBM từ tệp optimized_lightgbm_model.pkl')

if 'X_test' in globals():
    data_to_predict = X_test
    source = "X_test"
else:
    data_to_predict = None

if data_to_predict is not None:
    example_prediction = loaded_model.predict(data_to_predict.head(5))
    print(f'\n⚡ Dự đoán mẫu bằng mô hình đã nạp (Sử dụng dữ liệu từ {source}):')
    for i, pred_val in enumerate(example_prediction):
        print(f'   • Mẫu {i+1}: {pred_val:.4f} kW')
else:
    print('\n⚠️ Không tìm thấy dữ liệu X_test để thực hiện dự đoán mẫu.')
    # ==============================================================================
# BƯỚC 5: XỬ LÝ VÀ DỰ BÁO TRÊN FILE DỮ LIỆU THỨ 2 (household_power_consumption(2).csv)
# ==============================================================================
file_path_2 = "/content/household_power_consumption(2).csv"

# 1. Đọc file (2) - xử lý phân cách dấu phẩy và ký tự CH/SA nếu có
df2 = pd.read_csv(file_path_2, na_values=["?"], low_memory=False)

# 2. Xử lý định dạng Time và Date (Thay thế CH -> PM và SA -> AM)
df2["Time"] = (
    df2["Time"].astype(str).str.replace("CH", "PM").str.replace("SA", "AM")
)
df2["Datetime"] = pd.to_datetime(
    df2["Date"] + " " + df2["Time"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce"
)
df2.set_index("Datetime", inplace=True)
df2.drop(columns=["Date", "Time"], inplace=True)

# 3. Ép kiểu dữ liỆu về dạng số (Bỏ Scale down /1000 để khớp vối Train set)
for col in df2.columns:
  df2[col] = pd.to_numeric(df2[col], errors="coerce")

# 4. Resample theo khung 1 giờ & loại bỏ NaN
df2_hourly = df2[["Global_active_power"]].resample("1h").mean().dropna()

print(
    f"✅ Đã làm sạch dữ liệu file (2). Tổng số dòng theo giờ:"
    f" {len(df2_hourly):,}"
)

# TẠO FEATURE ENGINEERING CHO FILE (2) DÙNG HÀM create_advanced_features
df2_features = create_advanced_features(df2_hourly)

# Tách Features (X_test2) và Target thực tế (y_test2)
X_test2 = df2_features[feature_cols]
y_test2 = df2_features["Global_active_power"]
test2_indices = df2_features.index

# DÙNG MÔ HÌNH model_opt ĐỂ DỰ BÁO TRÊN FILE (2)
y_pred2 = model_opt.predict(X_test2)

# Đánh giá xem mô hình hoạt động như thế nào trên file (2)
r2_test2 = r2_score(y_test2, y_pred2)
rmse_test2 = np.sqrt(mean_squared_error(y_test2, y_pred2))

print(" --- KẾT QUẢ TEST MÔ HÌNH TRÊN FILE (2) ---")
print(f"   • R² Score: {r2_test2:.4f}")
print(f"   • RMSE: {rmse_test2:.4f} kW")

# 1. Định nghĩa từ điển chứa hành động khắc phục cho từng trường hợp
RECOMMENDATIONS = {
    "HIGH_SPIKE": [
        "Tắt bớt các thiết bị công suất lớn vừa bật (bếp từ, bình nóng lạnh, máy sấy).",
        "Nếu không chủ động bật thiết bị lớn, hãy kiểm tra ngay nguy cơ chập/rò điện.",
        "Phân bổ lại thời gian dùng thiết bị để tránh nhảy Aptomat tổng."
    ],
    "NIGHT_LEAK": [
        "Kiểm tra các thiết bị chạy nền ban đêm (tủ lạnh, máy lọc nước, đèn sân vườn).",
        "Rút phích cắm các thiết bị đang ở chế độ chờ (Standby) như TV, sạc pin, loa.",
        "Cài đặt hẹn giờ tắt tự động cho ổ cắm thông minh từ 00:00 - 05:00."
    ],
    "SUDDEN_DROP": [
        "Kiểm tra xem có bị mất điện cục bộ hoặc nhảy Aptomat nhánh nào không.",
        "Xác minh ngay các thiết bị tải nền quan trọng (tủ lạnh, camera an ninh, wifi)."
    ]
}

# 2. Cập nhật hàm phát hiện và đưa ra gợi ý hành động
def evaluate_anomalies_prescriptive(y_actual_series, y_pred_array, indices, threshold_val, top_n=5):
    test_errors = np.abs(y_actual_series.values - y_pred_array)
    alerts = []

    for idx, actual, pred, err in zip(indices, y_actual_series.values, y_pred_array, test_errors):
        if err > threshold_val:
            diff = actual - pred
            hour = idx.hour

            # Phân loại và gán mã hành động tương ứng
            if diff > 0:
                alert_code = "HIGH_SPIKE"
                alert_type = "🔥 TĂNG ĐỘT BIẾN (Nguy cơ quá tải / rò rỉ điện)"
            elif hour >= 0 and hour <= 5 and actual > 1.5:
                alert_code = "NIGHT_LEAK"
                alert_type = "🌙 RÒ RỈ ĐIỆN BAN ĐÊM (Tải nền cao bất thường)"
            else:
                alert_code = "SUDDEN_DROP"
                alert_type = "⚠️ GIẢM ĐỘT BIẾN (Mất nguồn / Thiết bị ngắt)"

            # Lấy danh sách hành động gợi ý
            actions = RECOMMENDATIONS[alert_code]
            actions_formatted = "\n".join([f"      • {act}" for act in actions])

            msg = (
                f"🚨 [CẢNH BÁO {idx.strftime('%Y-%m-%d %H:%M')}]\n"
                f"   • Thực tế: {actual:.2f} kW | Dự báo: {pred:.2f} kW (Độ lệch: {abs(diff):.2f} kW)\n"
                f"   • Chẩn đoán: {alert_type}\n"
                f"   💡 GỢI Ý HÀNH ĐỘNG NGHĨA VỤ:\n{actions_formatted}"
            )
            alerts.append((idx, msg))

    print(f"🔎 Tổng số bất thường: {len(alerts)} giờ. Mẫu {min(top_n, len(alerts))} cảnh báo đầu tiên:\n")
    for _, alert_msg in alerts[:top_n]:
        print(alert_msg)
        print("-" * 65)

    return alerts

import requests

BOT_TOKEN = "8694007977:AAFoyMTjx3S0mlyE051wLLCVn2nmfhWnZuc"
CHAT_ID = "6160425116"

def send_telegram_alert(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("✅ Đã gửi cảnh báo qua Telegram thành công!")
        else:
            print(f"⚠️ Lỗi gửi Telegram ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")

# Cập nhật hàm phát hiện và tự động bắn cảnh báo realtime
def evaluate_and_notify_telegram(y_actual_series, y_pred_array, indices, threshold_val, top_n=3):
    test_errors = np.abs(y_actual_series.values - y_pred_array)
    alerts = []
    sent_count = 0

    for idx, actual, pred, err in zip(indices, y_actual_series.values, y_pred_array, test_errors):
        if err > threshold_val:
            diff = actual - pred
            hour = idx.hour

            if diff > 0:
                alert_code = "HIGH_SPIKE"
                alert_type = "🔥 *TĂNG ĐỘT BIẾN* (Quá tải / Rò rỉ)"
            elif hour >= 0 and hour <= 5 and actual > 1.5:
                alert_code = "NIGHT_LEAK"
                alert_type = "🌙 *RÒ RỈ ĐIỆN BAN ĐÊM*"
            else:
                alert_code = "SUDDEN_DROP"
                alert_type = "⚠️ *GIẢM ĐỘT BIẾN* (Mất nguồn)"

            actions = RECOMMENDATIONS[alert_code]
            actions_formatted = "\n".join([f"   • {act}" for act in actions])

            # Đinh dạng văn bản hỗ trợ Markdown cho Telegram
            msg = (
                f"🚨 *[CẢNH BÁO {idx.strftime('%Y-%m-%d %H:%M')}]*\n"
                f"• *Thực tế:* `{actual:.2f} kW` | *Dự báo:* `{pred:.2f} kW`\n"
                f"• *Độ lệch:* `{abs(diff):.2f} kW`\n"
                f"• *Chẩn đoán:* {alert_type}\n\n"
                f"💡 *GỢI Ý HÀNH ĐỘNG:* \n{actions_formatted}"
            )
            alerts.append(msg)

            # Gửi thử nghiệm top_n cảnh báo đầu tiên về Telegram
            if sent_count < top_n:
                send_telegram_alert(msg, BOT_TOKEN, CHAT_ID)
                sent_count += 1

    return alerts

# 4. Chạy thử nghiệm gửi cảnh báo về điện thoại
alerts_sent = evaluate_and_notify_telegram(
    y_test2, y_pred2, test2_indices, threshold, top_n=3
)
import matplotlib.pyplot as plt

# ==============================================================================
# BƯỚC 6: TRỰC QUAN HÓA KẾT QUẢ DỰ BÁO TRÊN FILE (2)
# ==============================================================================
plt.figure(figsize=(12, 6))
plt.plot(y_test2.values[:100], label='Actual (File 2)', color='blue', alpha=0.6, linewidth=2)
plt.plot(y_pred2[:100], label='Predicted (File 2)', color='red', linestyle='--', alpha=0.8)
plt.title('Time-Series Forecasting for File (2): 100-Minute Window Comparison')
plt.xlabel('Time Steps (Minutes)')
plt.ylabel('Global Active Power')
plt.legend()
plt.grid(True)
plt.show()