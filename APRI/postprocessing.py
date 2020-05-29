'''
Postprocessing methods aimed to clean model outputs


'''


def event_filter_v1(event,frame_th=5,frame_th_8=50):
     process=True
     length=event.get_frames()[len(event.get_frames())-1]-event.get_frames()[0]
     if length < frame_th:  #very short events
        process=False
     elif event.get_classID()==8:
        if length<frame_th_8: #false positives assigned to class_idx=8 'footsteps'
            process=False
     return process