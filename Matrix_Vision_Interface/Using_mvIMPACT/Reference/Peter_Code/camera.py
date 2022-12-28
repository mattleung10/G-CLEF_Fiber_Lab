#! /usr/bin/env python 
# 
# camera.py
#
# A 'camera' can be used to acquire images of various types: 'bias', 'dark'
# 'exp', and 'fe55'. It can also create a fake image file for testing.
#

#import sys
import os
import time
import datetime
import shutil
from astropy.io import fits
from astropy.time import Time
import numpy as np

import imgr_fits
#import eolib

# the parent class
from instrument import instrument

#----------------------------------------------------------------------------------------------------------------------
class camera(instrument):

    def __init__(self, info, sensor=None, shutter=None, display=None, archiver=None, 
                 clobber=True, logger=None, verbose=False, fakeout=False):
        super().__init__(info, verbose=verbose, logger=logger)

        # record values from initialization
        self.sensor = sensor
        self._shutter = shutter
        self._display = display
        self._archiver = archiver
        self._clobber = clobber
        self._fakeout = fakeout

        if self.name is None: self.name = 'CAMERA'
        if self.Name is None: self.Name = 'CAMERA'
        if self.adc_bits is None: self.adc_bits = 16
        if self.amplifier_gain is None: self.amplifier_gain =  0.0 # electronic gain, in dB
        if self.system_gain is None: self.system_gain = 1.0        # system gain, in e-/DN 
        if self.system_noise is None: self.system_noise = 0.0      # system read noise in e- rms
        if self.adc_offset is None: self.adc_offset = 0.0          # % of full scale
        if self.minimum_exposure_time is None: self.minimum_exposure_time = 0.0
        if self.maximum_exposure_time is None: self.maximum_exposure_time = None
        if self.fakeout is None: self.fakeout = False 

        #shutter.__init()__(info, verbose=verbose, logger=logger)
        
        # initialize default values for local variables
        self._shutter_state = 'Unknown'
        self._exposure_time = 0.0
        self.test = 'None'
        self.imtype = 'None'
        
        self._dataset = 'dark'
        self._subset = 0 
        self._seqnum = 0
        
        self.exposure_start = datetime.datetime.utcnow()
        self.exposure_stop = datetime.datetime.utcnow()
        self.darktime_start = datetime.datetime.utcnow()
        self.darktime_stop = datetime.datetime.utcnow()
        
        # TODO What's up with how we handle system gain? Both _system_gain and self.system_gain
        # and the latter comes from the configuration file for the camera and the former is set
        # ro None and then can be set by calling system_gain function? WTF?
        self._system_gain = None
        
        # we're assuming a full readout including serial extension here
        self.read_params = {'pre_rows':int(self.sensor.pre_rows), 
                            'read_rows':int(self.sensor.read_rows), 
                            'post_rows':int(self.sensor.post_rows), 
                            'over_rows':int(self.sensor.over_rows), 
                            'pre_cols':int(self.sensor.pre_cols), 
                            'read_cols': int(self.sensor.read_cols), 
                            'post_cols':int(self.sensor.post_cols), 
                            'over_cols':int(self.sensor.over_cols), 
                            'bin_rows': int(self.sensor.bin_rows), 
                            'bin_cols':int(self.sensor.bin_cols), 
                            }
         
    #------------------------------------------------------------------------------
    def read_parameter(self, item=None, value=None):
        if item is None:
            return self.read_params
        else:
            if item in self.read_params.keys():
                if isinstance(value, int):
                    self.read_params[item] = value
                    self._log("Readout parameter %s set to %s" % (item,value))
                else:
                    self._error("Readout parameters must be of type 'int' ")
            else:
                self._error("Dict read_params does not contain item: %s" % item)

    #------------------------------------------------------------------------------
    def full_image_size(self):
        # how many rows and columns will be in an image?
        rows = self.read_params['read_rows'] + self.read_params['over_rows']
        cols = self.read_params['read_cols'] + self.read_params['over_cols']
        return rows, cols     
        
    #-----------------------------------------------------------------------------------------------------------------
    # The dataset is the type of data that is being acquired such as: sbias, dark, flat, sflat, etc.
    def dataset(self, dataset=None):
        if dataset is not None: 
            #if self._verbose: print("Setting self._dataset to %s" % dataset)
            self._dataset = str(dataset)
        else:
            return self._dataset
    
    #-----------------------------------------------------------------------------------------------------------------
    # The subset is the particular set of data being taken, identified by a number. Typically these would be a set
    # of frames at a particular exposure time, wavelength, or something of that sort.
    def subset(self, subset=None):
        if subset is not None: 
            #if self._verbose: print("Setting self._subset to %d" % subset)
            self._subset = int(subset)
        else:
            return self._subset
    
    def increment_subset(self, increment=1):
        self._subset = self._subset + increment
        
    #-----------------------------------------------------------------------------------------------------------------
    # The seqnum is the image number in a set or subset. Frames with the same subset and different seqnum numbers
    # are related in some way. Could be a data set at a particular wavelength setting, but some images might be
    # flats whie others are biases, for example.
    def seqnum(self, seqnum=None):
        if seqnum is not None: 
            #if self._verbose: print("Setting self._seqnum to %d" % seqnum)
            self._seqnum = int(seqnum)
        else:
            return self._seqnum
        
    def increment_seqnum(self, increment=1):
        self._seqnum = self._seqnum + increment
        
    def reset_seqnum(self):
        self._seqnum = 0
        
    #-----------------------------------------------------------------------------------------------------------------    
    # The dark time is different from the exposure time. It is the cumulative time from the end of any charge
    # readout or clearing to the start of the next readout. It includes any time setting up the system or
    # opening or closing the shutter, etc. It is at least as long as the exposure time.
    def reset_darktime(self):
        self.darktime_start = Time.now()
        
    def data_geometry(self):
        data_rows = (self.read_params['read_rows'] + self.read_params['over_rows']) / self.read_params['bin_rows']
        data_cols = (self.read_params['read_cols'] + self.read_params['over_cols']) / self.read_params['bin_cols']
        return data_rows, data_cols

    #------------------------------------------------------------------------------
    def system_gain(self, gain=None):
        if gain is not None: 
            if self._verbose: print("Setting self._sysgain to %f" % gain)
            self._system_gain = gain
        else:
            return self._system_gain

    #------------------------------------------------------------------------------
    def debug(self):
        super().debug()
        #print("CCD: %s" % self.ccd.manufacturer()) 
        if self._shutter is not None: print("Shutter: %s" % self._shutter._name )
        if self._display is not None: print("Display: %s" % self._display._name)
        print("Clobber: %s" % self._clobber) 
        print("Fakeout: %s" % self._fakeout)

    #------------------------------------------------------------------------------
    def shutter_status(self):
        return self._shutter.state()
        
    #------------------------------------------------------------------------------
    # _show : if a display is available, toss the file up there
    def _show(self, filename):
        if (self._display != None):
            self._display.load(filename)
        return

    #------------------------------------------------------------------------------------------------------------------
    def _log_stats(self, fitsname):
        im = fits.getdata(fitsname)
        i_min = np.min(im)
        i_max = np.max(im)
        i_mean = np.mean(im)
        i_std = np.std(im)
        self._logger.stats("%50s %10.2f  %10.2f  %10.2f  %10.2f" % (fitsname,i_min,i_max,i_mean,i_std))

    def _header_set(self, hdr, keyword, value, comment=None, dtype=str):
        try:
            hdr.set(keyword, dtype(value), comment)
        except TypeError:
            pass
            #print("camera:_header_set:TypeError:", keyword, value, comment)

    def fill_primary_header(self, hdr):
        #self._log("Filling Primary HDU")
        now = Time.now()
        data_rows, data_cols = self.data_geometry()
        #print("data_rows", data_rows,"    data_cols:", data_cols)
        det_rows = int(self.sensor.total_rows)
        det_cols = int(self.sensor.total_columns)
        seg_rows = int(self.sensor.segment_rows)
        seg_cols = int(self.sensor.segment_columns)
        ser_ext = int(self.sensor.serial_extension)
        over_rows = self.read_params['over_rows']
        over_cols = self.read_params['over_cols']
        amps = int(self.sensor.amplifier_count)
        if amps == 1:
            pass
        else:
            self._header_set(hdr,'NEXTEN', int(amps), comment='Number of image extensions', dtype=int)
        self._header_set(hdr,'BUNIT', 'adu', comment='Physical unit of array values')
        self._header_set(hdr,'ORIGIN', 'SAO', comment='Institution')
        self._header_set(hdr,'INSTRUME', 'TEST STAND 1', comment='Instrument used')
        self._header_set(hdr,'DATE ', now.isot, comment='Date the file was written')
        self._header_set(hdr,'MJD-OBS', now.mjd, comment='MJD at exposure start (d)', dtype=float)
        self._header_set(hdr,'DATE-OBS', now.fits, comment='Exposure start')
        #self._header_set(hdr,'EXPTIME', '%12.3f' % float(self._exposure_time), comment='Exposure time')
        #self._header_set(hdr,'DARKTIME', '%12.3f' % float(self._exposure_time), comment='Dark signal integration time')
        self._header_set(hdr,'EXPTIME', self._exposure_time, comment='Exposure time', dtype=float)
        self._header_set(hdr,'DARKTIME', self._exposure_time, comment='Dark signal integration time', dtype=float)
        self._header_set(hdr,'TEST', '%s' % self.test, comment='Test that this data is used for')
        self._header_set(hdr,'DATASET', '%s' % self._dataset.upper(), comment='Data set file belongs to')
        self._header_set(hdr,'SUBSET', self._subset, comment='Data subset file belongs to', dtype=int)
        self._header_set(hdr,'SEQNUM', self._seqnum, comment='Frame sequence number', dtype=int)
        self._header_set(hdr,'IMGTYPE', self.imtype, comment='image type: BIAS, DARK, FLAT, etc')
        
        #self._header_set(hdr,'ORIGFILE', os.path.basename(filename), comment='Original file name')
        self._header_set(hdr,'CAMERA', self.name)
        self._header_set(hdr,'CAMMODEL', self.model)
        self._header_set(hdr,'DETECTOR', self.sensor.name, comment= 'Detector name')
        self._header_set(hdr,'CCDNAME', self.sensor.name, comment= 'Detector name')
        self._header_set(hdr,'DETTYPE', self.sensor.type, comment='Detector technology')
        self._header_set(hdr,'NAMPS', self.sensor.amplifier_count, comment='Number of Amplifiers', dtype=int)
        self._header_set(hdr,'DETSIZE', "[1:%d,1:%d]" % (det_cols,det_rows), comment='Detector size for DETSEC')
        if amps == 1:
            self._header_set(hdr,'CCDSEC',  "[%d:%d,%d:%d]" % (1, seg_cols, 1, seg_rows))
            self._header_set(hdr,'DETSEC',  "[%d:%d,%d:%d]" % (1, seg_cols, 1, seg_rows))
            self._header_set(hdr,'AMPSEC',  "[%d:%d,%d:%d]" % (ser_ext+1, ser_ext+seg_cols, 1, seg_rows))
            self._header_set(hdr,'DATASEC', "[%d:%d,%d:%d]" % (ser_ext+1, ser_ext+seg_cols, 1, seg_rows))
            self._header_set(hdr,'TRIMSEC', "[%d:%d,%d:%d]" % (ser_ext+1, ser_ext+seg_cols, 1, seg_rows))
            # add BIASSEC for serial overscan and then parallel overscan
            self._header_set(hdr,'BIASSEC', "[%d:%d,%d:%d]" % (ser_ext+seg_cols+1, data_cols, 1, seg_rows))
            #self._header_set(hdr,'BIASSEC', "[%d:%d,%d:%d]" % (ser_ext+1, ser_ext+seg_cols, seg_rows+1, data_rows))
        self._header_set(hdr,'PIXSIZE1', self.sensor.pixel_size_x, comment='Pixel size for axis 1 (microns)', dtype=float)
        self._header_set(hdr,'PIXSIZE2', self.sensor.pixel_size_y, comment='Pixel size for axis 2 (microns)', dtype=float)
        self._header_set(hdr,'SENSTYPE', self.sensor.type, comment='Sensor type: CMOS or CCD')
        self._header_set(hdr,'SENSROWS', self.sensor.total_rows, comment='Total rows', dtype=int)
        self._header_set(hdr,'SENSCOLS', self.sensor.total_columns, comment='Total columns', dtype=int)
        self._header_set(hdr,'SEGROWS', self.sensor.segment_rows , comment='number of imaging rows per segment', dtype=int)
        self._header_set(hdr,'SEGCOLS', self.sensor.segment_columns, comment='number of imaging columns per segment', dtype=int)
        self._header_set(hdr,'SEREXT', self.sensor.serial_extension, comment='pixels in the serial extension', dtype=int)
        self._header_set(hdr,'OVERCOLS', over_cols, comment='Overscan columns', dtype=int)
        self._header_set(hdr,'OVERROWS', over_rows, comment='Overscan rows', dtype=int)
        self._header_set(hdr,'PIXWELL', self.sensor.pixel_full_well, comment='e- (not used)', dtype=int)
        self._header_set(hdr,'REGWELL', self.sensor.register_full_well, comment='e- (not used)', dtype=int)
        self._header_set(hdr,'NODEWELL', self.sensor.output_node_capacity, comment='e- (not used)', dtype=int)
        self._header_set(hdr,'DARKCUR', self.sensor.dark_current, comment='e-/pixel/sec', dtype=float)
        self._header_set(hdr,'PREROWS', self.read_params['pre_rows'], dtype=int)
        self._header_set(hdr,'READROWS', self.read_params['read_rows'], dtype=int)
        self._header_set(hdr,'POSTROWS', self.read_params['post_rows'], dtype=int)
        self._header_set(hdr,'OVERROWS', self.read_params['over_rows'], dtype=int)
        self._header_set(hdr,'PRECOLS', self.read_params['pre_cols'], dtype=int)
        self._header_set(hdr,'READCOLS', self.read_params['read_cols'], dtype=int)
        self._header_set(hdr,'POSTCOLS', self.read_params['post_cols'], dtype=int)
        self._header_set(hdr,'OVERCOLS', self.read_params['over_cols'], dtype=int)
        self._header_set(hdr,'BINROWS', self.read_params['bin_rows'], dtype=int)
        self._header_set(hdr,'BINCOLS', self.read_params['bin_cols'], dtype=int)
 
        if amps == 1:
            adc_gain = ((float(self.adc_range)/float(self.amplifier_gain)) / (2 ** int(self.adc_bits))) * (10 ** 6)  # uV/DN
            system_gain = adc_gain / float(self.sensor.sensitivity)
            self._header_set(hdr,'SYSGAIN', system_gain, comment='System gain, e-/DN', dtype=float)
            self._header_set(hdr,'AMPSENS', self.sensor.sensitivity, comment='Amplifier sensitivy, uV/e-', dtype=float)
            self._header_set(hdr,'RDNOISE', self.sensor.read_noise, comment='Amplifier read noise, e-', dtype=float)
            self._header_set(hdr,'PARCTE', self.sensor.cte_parallel, dtype=float)
            self._header_set(hdr,'SERCTE', self.sensor.cte_serial, dtype=float)
        return hdr

    #------------------------------------------------------------------------------------------------------------------
    # Cameras must write their own FITs file. The archiver will update with information about
    # other equipmen tin the setup
    def _write_fits(self, filename, pixel_data):
        self._log("Writing FITS file: %s" % filename)
        #rows = np.shape(pixel_data)[0]
        #cols = np.shape(pixel_data)[1]
        hdu = fits.PrimaryHDU(pixel_data)
        #print("HDU = ", hdu)
        #hdu.header.set('BITPIX', 16)
        #hdu.header.set('BSCALE', 1)
        #hdu.header.set('BZERO', 0)
        #self._log("Rows = %d   Columns = %d" % (rows,cols))
        #hdu.header.set('NAXIS', 2)
        #hdu.header.set('NAXIS1', cols)
        #hdu.header.set('NAXIS2', rows, after='NAXIS1')
        hdu.header.set('FILENAME', os.path.basename(filename))
        hdu.header.set('ORIGFILE', os.path.basename(filename))
        hdu.header.append('CAMERA', self.name)
        hdu.header.append('MODEL', self.model)
        hdu.header = self.fill_primary_header(hdu.header)
        hdul = fits.HDUList([hdu])
        if self._clobber == True and os.path.isfile(filename) == True: os.remove(filename)
        hdul.writeto(filename)

    # #------------------------------------------------------------------------------------------------------------------
    # # Cameras are responsible for writing working FITS files. 
    # # At present, for Lumenera and MV the external
    # # program we are calling writes a FITS file for use, so we never call this in that case.
    # # Will that be true of Binspec? All cameras? Not for Spectral!
    # def _write_mef(self, filename, pixel_data):
    #     self._log("Writing Multi-extension FITS file: %s" % filename)
    #     #data_rows = np.shape(data)[0]
    #     data_rows, data_cols = self.data_geometry()
    #     #data_rows, data_cols = self.data_geometry()
    #     segs = np.shape(pixel_data)[0]
    #     rows = np.shape(pixel_data)[1]
    #     cols = np.shape(pixel_data)[2]
    #     det_rows = int(self.sensor.total_rows)
    #     det_cols = int(self.sensor.total_columns) 
    #     seg_rows = int(self.sensor.segment_rows)
    #     seg_cols = int(self.sensor.segment_columns)
    #     ser_ext = int(self.sensor.serial_extension)
    #     #over_rows = data_rows - seg_rows
    #     #over_cols = data_cols - (ser_ext + seg_cols)
    #     #read_params = self.read_params
    #     #print("Segs: %d   Rows: %d   Cols: %d" % (segs,rows,cols))
    #     # create the fits headers and add the data arrays
    #     hdul = fits.HDUList()
    #     hdul.append(fits.PrimaryHDU())
    #     hdul[0].header.set('FILENAME', os.path.basename(filename))
    #     hdul[0].header.set('ORIGFILE', os.path.basename(filename), comment='Original file name')
    #     hdul[0].header.set('BITPIX', self.adc_bits)
    #     hdul[0].header.set('BSCALE', 1)
    #     hdul[0].header.set('BZERO', 0)
    #     hdul[0].header.set('DATAROWS', rows, comment='Total number of rows')
    #     hdul[0].header.set('DATACOLS', cols, comment='Total number of columns')
    #     hdul[0].header = self.fill_primary_header(hdul[0].header)
    #     for i in range(segs):
    #         ext = i + 1
    #         hdul.append(fits.ImageHDU(pixel_data[i]))
    #         hdr = hdul[ext].header
    #         #data = hdul[ext].data
    #         #over_rows = rows - seg_rows
    #         ser_ext = int(self.sensor.serial_extension)
    #         #over_cols = cols - (ser_ext + seg_cols)
    #         self._header_set(hdr,'EXTNAME', "SEG%02d" % i)
    #         self._header_set(hdr,'IMAGEID', ext, dtype=int)
    #         self._header_set(hdr,'CCDNAME', self.sensor.name)
    #         self._header_set(hdr,'CCDSIZE', "[%d:%d,%d:%d]" % (1,det_rows,1,det_cols))
    #         self._header_set(hdr,'AMPNAME', "AMP%02d" % i)
    #         #self._header_set(hdr,'PARCTE', self.sensor.cte_parallel[i], dtype=float)
    #         #self._header_set(hdr,'SERCTE', self.sensor.cte_serial[i], dtype=float)
    #         #self._header_set(hdr,'AMPOFSET', self.lab['sensor'].bias_offset[i])
    #         #adc_gain = ((float(self.adc_range)/float(self.amplifier_gain)) / (2 ** int(self.adc_bits))) * (10 ** 6)  # uV/DN
    #         #system_gain = adc_gain / self.sensor.sensitivity[i]            # (uV/DN)/(uV/e-) = e-/DN
    #         #self._header_set(hdr,'SYSGAIN', system_gain, comment='System gain, e-/DN', dtype=float)
    #         #self._header_set(hdr,'AMPSENS', self.sensor.sensitivity[i], comment='Amplifier sensitivy, uV/e-', dtype=float)
    #         #self._header_set(hdr,'RDNOISE', self.sensor.read_noise[i], comment='Amplifier read noise, e-', dtype=float)
    #         #self._header_set(hdr,'CCDSUM', "1,1" ) # DON'T include this! Makes DS9 stop reading files correctly!
    #         if i == 0:
    #             self._header_set(hdr,'CCDSEC', "[%d:%d,%d:%d]" % (1, seg_cols+1, 1, seg_rows+1))
    #             self._header_set(hdr,'DETSEC', "[%d:%d,%d:%d]" % (1, seg_cols+1, 1, seg_rows+1))
    #         elif i == 1:
    #             self._header_set(hdr,'CCDSEC', "[%d:%d,%d:%d]" % (det_cols, seg_cols+1, 1, seg_rows+1))
    #             self._header_set(hdr,'DETSEC', "[%d:%d,%d:%d]" % (det_cols, seg_cols+1, 1, seg_rows+1))
    #         elif i == 2:
    #             self._header_set(hdr,'CCDSEC', "[%d:%d,%d:%d]" % (1, seg_cols, det_rows, seg_rows+1))
    #             self._header_set(hdr,'DETSEC', "[%d:%d,%d:%d]" % (1, seg_cols, det_rows, seg_rows+1))
    #         else:
    #             self._header_set(hdr,'CCDSEC', "[%d:%d,%d:%d]" % (det_cols, seg_cols+1, det_rows, seg_rows+1))
    #             self._header_set(hdr,'DETSEC', "[%d:%d,%d:%d]" % (det_cols, seg_cols+1, det_rows, seg_rows+1))
    #         self._header_set(hdr,'AMPSEC',  "[%d:%d,%d:%d]" % (ser_ext+1, ser_ext+seg_cols, 1, seg_rows))
    #         self._header_set(hdr,'DATASEC', "[%d:%d,%d:%d]" % (ser_ext+1, ser_ext+seg_cols, 1, seg_rows))
    #         self._header_set(hdr,'TRIMSEC', "[%d:%d,%d:%d]" % (ser_ext+1, ser_ext+seg_cols, 1, seg_rows))
    #         # add BIASSEC for serial overscan 
    #         self._header_set(hdr,'BIASSEC', "[%d:%d,%d:%d]" % (ser_ext+seg_cols+1, data_cols, 1, seg_rows))
    #     # write the fits file to disk
    #     if self._clobber == True and os.path.isfile(filename) == True: os.remove(filename)
    #     hdul.writeto(filename)

    #------------------------------------------------------------------------------
    def _show(self, pixel_data):
        print("This would be a good place to display the data somewhere!")

    #------------------------------------------------------------------------------
    def _fake_frame(self, filename, exptime=0.0, test=None, imtype=None):
        time.sleep(exptime)
        self._log("Fake file: %s" % (os.path.split(filename)[1]))
        # file = open(filename, 'wb')
        # file.close
        time.sleep(2.0) # should be camera readout time
        shutil.copyfile('Fakeout.fits', filename)

    #------------------------------------------------------------------------------
    # quad: acquire a set of four images, two biases and two identical exposures
    def quad(self, filebase='quad', path='.', test='GAIN', exptime=1.0):
        self.test = test
        self._exposure_time = exptime
        self.reset_seqnum()
        # bias1 = self.frame("%s_bias1" % filebase, exptime=0, path=path, test=test, imtype='BIAS')
        # bias2 = self.frame("%s_bias2" % filebase, exptime=0, path=path, test=test, imtype='BIAS')
        # flat1 = self.frame("%s_flat1" % filebase, exptime=exptime, path=path, test=test, imtype='FLAT')
        # flat2 = self.frame("%s_flat2" % filebase, exptime=exptime, path=path, test=test, imtype='FLAT')
        self.seqnum(0)
        bias1 = self.frame("%s_000" % filebase, exptime=0, path=path, test=test, imtype='BIAS')
        self.seqnum(1)
        bias2 = self.frame("%s_001" % filebase, exptime=0, path=path, test=test, imtype='BIAS')
        self.seqnum(2)
        flat1 = self.frame("%s_002" % filebase, exptime=exptime, path=path, test=test, imtype='FLAT')
        self.seqnum(3)
        flat2 = self.frame("%s_003" % filebase, exptime=exptime, path=path, test=test, imtype='FLAT')
        return [bias1, bias2, flat1, flat2]

    #------------------------------------------------------------------------------
    # a 'sequence' of images is a set of frames acquired in an identical fashion
    def sequence(self, fbase, exptime=0.0, path='.', test=None, imtype=None, count=1, delay=0):
        self.test = test
        self.imtype = imtype
        self._exposure_time = exptime
        self._log("%-10s: Acquiring %3d %s data set %s frames" 
                    % (str.upper(test), count, str.upper(test), str.upper(imtype)))
        fitsnames=[]
        
        for i in range(count):
            self.seqnum(i)
            filebase = "%s_%05u" % (fbase,i)
            filename = self.frame(filebase, exptime, path=path, test=test, imtype=imtype)
            fitsnames.append(filename)
            time.sleep(delay)
        return fitsnames

    #------------------------------------------------------------------------------
    # frame
    # acquire a data frame. row_delay is only used in 'ramp' frames.
    
    # TODO should check for imtype and call other things like 'sine' and target and simply pass
    # extra parameters through instead of including them here, no?
    def frame(self, filebase, exptime=0.0, row_delay=0.0, path='.', test=None, imtype=None,
              cycles=5, amplitude=1, center=0.5, phase=-90):  # used only for SINE images
   
        self.test = test
        #if seqnum is not None: self._seqnum = seqnum
        self.imtype = imtype
        self._exposure_time = exptime
        min_exposure = float(self.minimum_exposure_time) # this does not work if NONE
        if min_exposure is None: min_exposure = 0.00
        max_exposure = float(self.maximum_exposure_time) # this does not work if NONE
        if max_exposure is None: max_exposure = 36000.00
        #self._log("Camera frame(): filebase = %s  exptime = %f  imtype = %s" % (filebase,exptime,imtype)) 
        if exptime < min_exposure:
            self._logger.warning("Exposure time is less than minimum. Using %6.3f seconds." % min_exposure)
            exptime = min_exposure
        if exptime > max_exposure:
            self._logger.warning("Exposure time greater than maximum. Using %10.3f seconds" % max_exposure)
            exptime = max_exposure
        self._exposure_time = exptime
        imtype = str.upper(imtype)
        if not imtype in ['BIAS', 'DARK', 'EXP', 'EXPOSE', 'FLAT', 'FE55', 'XRAY', 'OBJ', 
                          'RAMP', 'TDI', 'SPOT', 'SINE', 'OTHER', 'FAKE']:      
            self._logger.error("%-10s: Invalid image type.")
            return None
        if imtype in ['BIAS', 'DARK', 'FE55', 'FAKE' ] and self._shutter is not None: 
            if str.upper(self._shutter.state()) != 'CLOSED': self._shutter.close()
        if imtype == 'BIAS': exptime = 0.0  # ignore non-zero exptime for bias images
        fitsname = "%s/%s_%s.fits" % (os.path.abspath(path), filebase, time.strftime("%Y%m%d%H%M%S"))
        if self._clobber == True and os.path.isfile(fitsname) == True: os.remove(fitsname)
        
        if imtype in ['SINE']:
            self.acquire_frame(fitsname, exptime=exptime, test=test, imtype=imtype,
                               cycles=cycles, amplitude=amplitude, center=center, phase=phase)
        else:
            self.acquire_frame(fitsname, exptime=exptime, row_delay=row_delay, test=test, imtype=imtype) 
        
        self._log("%-10s: %-10s: %s "  % (str.upper(test), str.upper(imtype), os.path.basename(fitsname)))
        
        if self._archiver is not None:
            self._archiver.header_update(fitsname, exptime=exptime, test=test, imtype=imtype) 
            self._archiver.store_backup(fitsname)
        # i_min, i_max, i_mean, i_median, i_std = imgr_fits.stats(fitsname, region='data', stats='all')
        # self._logger.log("%-10s: %-50s  %-8s  %8.4f %6d %6d %8.2f %8.2f %8.2f " % 
        #     ('IMAGE',os.path.basename(fitsname), imtype, exptime, i_min, i_max, i_mean, i_median, i_std))
        # self._log_stats(fitsname)
        #if self._display is not None:
        #        self._show(pixel_data)
        return fitsname
        
    #------------------------------------------------------------------------------
    # functions below this should be overridden in actual camera interface code
    #------------------------------------------------------------------------------
    def acquire_frame(self, filename, exptime=0.0, row_delay=0, test=None, imtype=None, seqnum=None,
                      cycles=5, amplitude=1, center=0.5, phase=-90):
        self.test = test
        #if seqnum is not None: self._seqnum =seqnum
        self.imtype = imtype
        self._exposure_time = exptime
        if self._fakeout or imtype == 'FAKE': 
            self._fake_frame(filename, exptime=exptime, test=test, imtype=imtype)
        else: 
            self._log("No Camera: Pretending to acquire a frame")
            self._fake_frame(filename, exptime=exptime, test=test, imtype=imtype)
        
        #print(self._seqnum)
        #self.increment_seqnum()
        #print(self._seqnum)
        return 
    
    # def acquire_sine(self, filename, exptime=0.0, cycles=5, amplitude=1, center=0.5, phase=-90,
    #                  test=None, imtype='SINE', seqnum=None):
    #     self.test = test
    #     #if seqnum is not None: self._seqnum =seqnum
    #     self.imtype = imtype
    #     self._exposure_time = exptime
    #     if self._fakeout or imtype == 'FAKE': 
    #         self._fake_frame(filename, exptime=exptime, test=test, imtype=imtype)
    #     else: 
    #         self._log("No Camera: Pretending to acquire a frame")
    #         self._fake_frame(filename, exptime=exptime, test=test, imtype=imtype)
        
        #print(self._seqnum)
        #self.increment_seqnum()
        #print(self._seqnum)
        return 

    #------------------------------------------------------------------------------
    def open_shutter(self):
        if self._shutter is not None: 
            self._log("No Camera: Pretending to open shutter")
            self._shutter.open()
        self._shutter_status = 'Open'

    #------------------------------------------------------------------------------
    def close_shutter(self):
        if self._shutter is not None: 
            self._log("No Camera: Pretending to close shutter")
            self._shutter.close()
        self._shutter_status = 'Closed'

    #------------------------------------------------------------------------------
    def clear_ccd(self):
        self._log("No Camera: Pretending to clear CCD")
        self.darktime_start = datetime.datetime.utcnow()

    #------------------------------------------------------------------------------
    def read_ccd(self):
        self._log("No Camera: Pretending to read CCD")
        self.darktime_start = datetime.datetime.utcnow()

    # def exposure_time(self, milliseconds):
    #     if milliseconds is not None:
    #         self._log("No Camera: Setting exposure time to %f" % milliseconds)
    #         self._exposure_time = milliseconds
    #     else:
    #         return self._exposure_time 

    
        



    
