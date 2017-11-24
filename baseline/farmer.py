# farmer.py

# connector to the farms
from pyro_helper import pyro_connect

import threading as th
import time

# farmport = 20099

class farmlist:
    def __init__(self):
        self.list = []

    def generate(self):
        farmport = 20099
        def addressify(farmaddr,port):
            return farmaddr+':'+str(port)
        addresses = [addressify(farm[0],farmport) for farm in self.list]
        capacities = [farm[1] for farm in self.list]
        failures = [0 for i in range(len(capacities))]

        return addresses,capacities,failures

    def push(self, addr, capa):
        self.list.append((addr,capa))

fl = farmlist()

def reload_addr():
    global addresses,capacities,failures

    g = {'nothing':0}
    with open('farmlist.py','r') as f:
        farmlist_py = f.read()
    exec(farmlist_py,g)
    farmlist_base = g['farmlist_base']

    fl.list = []
    for item in farmlist_base:
        fl.push(item[0],item[1])

    addresses,capacities,failures = fl.generate()

reload_addr()

class remoteEnv:
    def pretty(self,s):
        print(('(remoteEnv) {} ').format(self.id)+str(s))

    def __init__(self,fp,id): # fp = farm proxy
        self.fp = fp
        self.id = id

    def reset(self):
        return self.fp.reset(self.id)

    def step(self,actions):
        ret = self.fp.step(self.id, actions)
        if ret == False:
            self.pretty('env not found on farm side, might been released.')
            raise Exception('env not found on farm side, might been released.')
        return ret

    def rel(self):
        while True: # releasing is important, so
            try:
                self.fp.rel(self.id)
                break
            except Exception as e:
                self.pretty('exception caught on rel()')
                self.pretty(e)
                time.sleep(3)
                pass

        self.fp._pyroRelease()

    def __del__(self):
        self.rel()

class farmer:
    def reload_addr(self):
        self.pretty('reloading farm list...')
        reload_addr()

    def pretty(self,s):
        print('(farmer) '+str(s))

    def __init__(self):
        for idx,address in enumerate(addresses):
            fp = pyro_connect(address,'farm')
            self.pretty('forced renewing... '+address)
            try:
                fp.forcerenew(capacities[idx])
                self.pretty('fp.forcerenew() success on '+address)
            except Exception as e:
                self.pretty('fp.forcerenew() failed on '+address)
                self.pretty(e)
                fp._pyroRelease()
                continue
            fp._pyroRelease()

    # find non-occupied instances from all available farms
    def acq_env(self):
        ret = False

        import random # randomly sample to achieve load averaging
        # l = list(enumerate(addresses))
        l = list(range(len(addresses)))
        random.shuffle(l)

        for idx in l:
            time.sleep(0.01)
            address = addresses[idx]
            capacity = capacities[idx]

            if failures[idx]>0:
                # wait for a few more rounds upon failure,
                # to minimize overhead on querying busy instances
                failures[idx] -= 1
                continue
            else:
                fp = pyro_connect(address,'farm')
                try:
                    result = fp.acq(capacity)
                except Exception as e:
                    self.pretty('fp.acq() failed on '+address)
                    self.pretty(e)

                    fp._pyroRelease()
                    failures[idx] += 4
                    continue
                else:
                    if result == False: # no free ei
                        fp._pyroRelease() # destroy proxy
                        failures[idx] += 4
                        continue
                    else: # result is an id
                        eid = result
                        renv = remoteEnv(fp,eid) # build remoteEnv around the proxy
                        self.pretty('got one on {} id:{}'.format(address,eid))
                        ret = renv
                        break

        # ret is False if none of the farms has free ei
        return ret

    # the following is commented out. should not use.
    # def renew(self):
    #     for idx,address in enumerate(addresses):
    #         fp = pyro_connect(address,'farm')
    #         try:
    #             fp.renew(capacities[idx])
    #         except Exception as e:
    #             print('(farmer) fp.renew() failed on '+address)
    #             print(e)
    #             fp._pyroRelease()
    #             continue
    #         print('(farmer) '+address+' renewed.')
    #         fp._pyroRelease()
