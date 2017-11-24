class TriggerBox():
    def __init__(self,msg,texts,callbacks):
        def show():
            import pymsgbox
            while True:
                chosen = pymsgbox.confirm(text=msg,title='triggers',buttons=texts)
                for i,t in enumerate(texts):
                    if t==chosen:
                        print(i,t,'chosen...')
                        callbacks[i]()
        import threading as th
        t = th.Thread(target=show)
        t.daemon = True
        t.start()
