from framework.data_interpolation import generate_antimony_piecewise

def generate_events():
    events = '' 
    return events

def event1():
    events = 'at (time >= 5): A_Comp1 = 10.0'
    pw_str1 = generate_antimony_piecewise(df["time"], df["B"], data_name="pw_interp1", interpolation_type="linear")
    events += "\n" + pw_str1
    pw_str2 = generate_antimony_piecewise(df["time"], df["B"], data_name="pw_interp2", interpolation_type="spline")
    events += "\n" + pw_str2
    return events

def event2():
    events = 'at (time >= 10): A_Comp1 = 5.0'
    pw_str1 = "pw_interp1 := 0.0"
    pw_str2 = "pw_interp2 := 0.0"
    events += "\n" + pw_str1
    events += "\n" + pw_str2
    return events
