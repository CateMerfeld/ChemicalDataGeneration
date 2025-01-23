from datetime import datetime, timedelta

# make a list of nums from 1-10
nums = list(range(1, 11))

# get the current time
current_time = datetime.now()

# calculate the date 5 days from now
future_date = current_time + timedelta(days=5)

print("Current time:", current_time)
print("Date in 5 days:", future_date)