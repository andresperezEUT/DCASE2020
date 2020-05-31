'''
Postprocessing methods aimed to clean model outputs


'''


def event_filter_v1(event,frame_th_8=50):
     process=True
     if event.get_classID()==8:
        if length<frame_th_8: #false positives assigned to class_idx=8 'footsteps'
            process=False
     return process