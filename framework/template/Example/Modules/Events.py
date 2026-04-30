from framework.data_interpolation import generate_antimony_piecewise

def generate_no_events(replicate, df_dict):
    events = '' 
    return events

def Example_event(replicate, df_dict):
    dose = replicate["dose"]
    delay = replicate["delay"]
    events = f'at (time >= {delay}): A_Comp1 = {dose}'
    return events



