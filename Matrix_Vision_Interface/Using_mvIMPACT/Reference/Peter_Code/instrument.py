#! /usr/bin/env python 
#
# instrument.py
# A generic class for control of instruments

class instrument(object): 

    def __init__(self, info, logger=None, verbose=False):
        if 'Device' not in info.keys(): info['Device'] = 'Instrument'
        if 'Name' not in info.keys(): info['Name'] = 'Instrument'
        self._logger = logger
        self._verbose = verbose
        self._info = info
        
        # go through keys in info dictionary and make attributes
        keys = list(info.keys())
        for i in range(len(info)):
            setattr(self, keys[i], info[keys[i]])
            #print(keys[i], ' : ', info[keys[i]])
    
    def __getattr__(self, item):
        return None
        
    def _log0(self, string):
        self._log(string, level=0)

    def _log1(self, string):
        self._log(string, level=1)

    def _log2(self, string):
        self._log(string, level=2)

    def _log3(self, string):
        self._log(string, level=3)

    def _log4(self, string):
        self._log(string, level=4)

    def _log(self, string, level=0):
        if self._verbose: 
            #pass
            print("%-10s: %s" % (self._info['Name'], string))
        if self._logger is not None: 
            self._logger.log("%-10s: %s" % (self._info['Name'], string), level=level)

    def _error(self, string):
        if self._verbose: 
            print("%-10s: ERROR : %s" % (self._info['Name'], string))
        if self._logger is not None: 
            self._logger.error("%-10s: %s" % (self._info['Name'], string))

    def verbose(self, state=None):
        if state is not None: 
            if state is True or state is False: 
                self._verbose = state
            else: 
                self._error("Verbosity must be either 'True' or 'False'")
        else:
            return self._verbose

    #  updates or returns a value in/from the info dictionary
    # def value(self, key, value=None, type=str):
    #     if value is not None:    # we're setting a new value for key
    #         self._log("setting info[%s] to %s" % (key, value))
    #         self._info[key] = value
    #         return None
    #     else:
    #         if key in self._info.keys():
    #             return self._info.get(key)
    #         else:
    #             return None

    def get_info(self):
        return self._info

    def debug(self):
        for k,v in self._info.items():
            print("%s : %s = %s" % (self._info['Device'],k,v))


    
