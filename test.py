from datetime import datetime

now = datetime.now()
date = now.date()
current_time = now.strftime("%H:%M:%S")  # string format time
print(date)
print(current_time)
