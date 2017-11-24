# Pyro4 library helper

import Pyro4 as p4

p4.config.HOST = '0.0.0.0'
p4.config.COMMTIMEOUT = 3600.0 # 20 min timeout
# p4.config.MAX_RETRIES = 2
p4.config.THREADPOOL_SIZE = 1000

def pyro_connect(address,name):
    uri = 'PYRO:'+name+'@'+address
    return p4.Proxy(uri)

def pyro_expose(c,port,name):
#    def stop():
#        print('stop() called')
#        import os
#        os._exit(1)
#    from triggerbox import TriggerBox
#    tb = TriggerBox(name+' server on '+str(port),
#        ['stop server'],
#        [stop])

    c = p4.behavior(instance_mode='single')(c)
    exposed = p4.expose(c)
    p4.Daemon.serveSimple(
            {
                exposed:name
            },
            ns = False,
            port = port)

    daemon = p4.Daemon()
    uri = daemon.register(c,name)
    daemon.requestLoop()
